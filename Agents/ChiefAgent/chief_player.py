from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from game_net.behavior_clone import BehaviorClone
import pandas as pd
import numpy as np
import random
from copy import deepcopy
from scipy.stats import entropy

STARTING_COLUMNS_MOVETRACKING = {"move_idx":[], "move": [], "observable game state":[], "card ids":[], "hand knowledge":[], "agent distribution":[], "conditional probabilities":[], "MLE probabilities":[], "generated samples":[]}
BOLTZMANN_CONSTANT = 4

CardChoices = []

for i in range(25):
        CardChoices.append(i)

# Note: hand is formmated as [(color, number)] - can use indices for color and indices + 1 for number

class Sample(object):
        def __init__(self, card_vals, conditional_probs):
                self.card_vals = card_vals
                self.conditional_probs = conditional_probs

        def consistent(self, knowledge):
                for i, card_id in enumerate(self.card_vals):
                        card = self.card_vals[card_id]

                        if knowledge[i][card[0]][card[1]-1] <= 0:
                                return False

                return True

class ChiefPlayer(Player):
        def __init__(self, name, pnr, pool_ids, num_samples=10, avoid_knowledge_rollback=False):
                self.name = name
                self.pnr = pnr
                self.pool_ids = pool_ids
                self.partner_nr = (pnr + 1) % 2
                self.move_tracking_table = pd.DataFrame(data=STARTING_COLUMNS_MOVETRACKING)
                self.move_tracking_table = self.move_tracking_table.set_index("move_idx")
                self.card_ids = dict()
                self.new_card_id = 0
                self.move_idx = 0
                self.total_card_knowledge = dict()
                self.drawn_dict = dict()
                self.hints_to_partner = []
                self.game_state_before_move = None
                self.player_model_before_move = None
                self.played_or_discarded_card = None
                self.num_samples = num_samples
                self.avoid_knowledge_rollback = avoid_knowledge_rollback

                self.sequential_gamestates_chief_persp = []
                self.sequential_playermodels_chief_persp = []
                self.sequential_partnerknowledgemodels_chief_persp = []

        def get_action(self, game_state, player_model, action_default=None):
                print("Starting get action")

                self.sequential_gamestates_chief_persp.append(game_state)
                self.sequential_playermodels_chief_persp.append(player_model)
                self.sequential_partnerknowledgemodels_chief_persp.append(self._make_partner_knowledge_model(game_state))

                if action_default is None:
                        action = self._get_action_helper()
                else:
                        action = action_default

                if len(self.card_ids) == 0:
                        for i, k in enumerate(player_model.knowledge):
                                self.card_ids[self.new_card_id] = i
                                self.drawn_dict[self.new_card_id] = self.move_idx
                                self.total_card_knowledge[self.new_card_id] = k
                                self.new_card_id += 1

                # code for shifting card ids for keeping track of positions in case we play/discard
                # this is used for our agents which are meant to model the teammate's point of view
                if action.type == PLAY or action.type == DISCARD:
                        new_card_ids = dict()

                        for cid in self.card_ids:
                                if self.card_ids[cid] < action.cnr:
                                        new_card_ids[cid] = self.card_ids[cid]
                                elif self.card_ids[cid] > action.cnr:
                                        new_card_ids[cid] = self.card_ids[cid] - 1
                                else:
                                        self.played_or_discarded_card = [cid, [], self.drawn_dict[cid]] # will fill in knowledge during inform
                                        del self.drawn_dict[cid]

                        new_card_ids[self.new_card_id] = len(player_model.knowledge) - 1
                        self.drawn_dict[self.new_card_id] = None
                        self.new_card_id += 1
                        self.card_ids = new_card_ids
                else:
                        self.hints_to_partner.append((self.pnr, action))


                return action

        def inform(self, action, player, game_state, player_model):
                print("Starting inform")

                changed_cards = []
                drawn_move = []

                #############################################################
                ### Update agents in pool if chief agent just took action ###
                #############################################################
                if player == self.pnr:
                        self.game_state_before_move = deepcopy(game_state)
                        self.player_model_before_move = deepcopy(player_model)

                        if action.type == PLAY:
                                temp = np.zeros(shape=(5,5), dtype=np.int32)
                                card = game_state.card_changed
                                temp[card[0]][card[1] - 1] = 1
                                self.played_or_discarded_card[1] = temp.tolist()
                        elif action.type == DISCARD:
                                temp = np.zeros(shape=(5,5), dtype=np.int32)
                                card = game_state.trash[-1]
                                temp[card[0]][card[1] - 1] = 1
                                self.played_or_discarded_card[1] = temp.tolist()

                        return


                if self.game_state_before_move == None: # CHIEF going second this means
                        if action == PLAY or action == DISCARD:
                                hand = []

                                for i in range(5):
                                        if i < action.cnr:
                                                hand.append(game_state.hands[self.partner_nr][i])
                                        elif i > action.cnr:
                                                hand.append(game_state.hands[self.partner_nr][i-1])
                                        else:
                                                hand.append(game_state.card_changed)
                        else:
                                hand = game_state.hands[self.partner_nr]

                        hands = [[]]*2
                        hands[self.partner_nr] = hand

                        self.game_state_before_move = GameState(self.partner_nr, hands, [], [], [(c, 0) for c in ALL_COLORS], 3, game_state.valid_actions, 8, [[initial_knowledge()]*5]*2)
                        self.player_model_before_move = BasePlayerModel(self.pnr, [initial_knowledge()]*5, [], [])

                if len(self.card_ids) == 0:
                        for i, k in enumerate(player_model.knowledge):
                                self.card_ids[self.new_card_id] = i
                                self.drawn_dict[self.new_card_id] = self.move_idx
                                self.total_card_knowledge[self.new_card_id] = k
                                self.new_card_id += 1

                ###########################################################
                ## Detecting relevant changes to all-time card knowledge ##
                ###########################################################
                mark_delete = []

                for c in self.card_ids:
                        if self.card_ids[c] >= len(player_model.get_knowledge()):
                                mark_delete.append(c)
                                continue

                        new_k = player_model.get_knowledge()[self.card_ids[c]]

                        if self.drawn_dict[c] == None:
                                self.drawn_dict[c] = self.move_idx
                                self.total_card_knowledge[c] = new_k
                        elif new_k != self.total_card_knowledge[c]:
                                changed_cards.append((c, new_k))
                                drawn_move.append(self.drawn_dict[c])

                        self.total_card_knowledge[c] = new_k

                ## If a card was "picked up" by the internal system, but not in the game engine, correct issue
                for marked_c in mark_delete:
                        del self.card_ids[marked_c]


                #########################################################################
                ### Updating historical data based on all-time card knowledge changes ###
                #########################################################################
                if ((len(changed_cards) > 0) or self.played_or_discarded_card != None) and not self.avoid_knowledge_rollback:
                        if (self.played_or_discarded_card):
                                changed_cards.append(tuple(self.played_or_discarded_card[:-1]))

                        if len(drawn_move) == 0:
                                starting_move = self.played_or_discarded_card[2]
                        elif self.played_or_discarded_card == None:
                                starting_move = min(drawn_move)
                        else:
                                starting_move = min(min(drawn_move), self.played_or_discarded_card[2])

                        self._rollforward_move_tracker(starting_move, self.move_idx, changed_cards)
                        self.played_or_discarded_card = None


                ##################################################################
                ### Creating initial row for this move index in tracking table ###
                ##################################################################
                new_row = dict()
                new_row["move"] = self._action_to_key(action)

                ## storing game state before move for get_action
                modified_game_state = deepcopy(self.game_state_before_move)
                modified_game_state.hands = [None if a == [] else [] for a in self.game_state_before_move.hands]
                modified_game_state.current_player = self.partner_nr
                modified_game_state.hinted_indices = []
                modified_game_state.card_changed = None
                VA = []

                for a in game_state.valid_actions:
                        if a.type in [PLAY, DISCARD] and a.cnr >= len(game_state.all_knowledge[self.partner_nr]):
                                continue
                        else:
                                VA.append(a)

                modified_game_state.valid_actions = deepcopy(VA)

                partners_hints = self.hints_to_partner
                modified_player_model = BasePlayerModel(self.partner_nr, self.game_state_before_move.all_knowledge[self.partner_nr], self.hints_to_partner, self.player_model_before_move.get_actions())
                new_row["observable game state"] = (modified_game_state, modified_player_model)

                new_row["card ids"] = self.card_ids
                new_row["hand knowledge"] = player_model.get_knowledge()
                new_row["generated samples"] = [None]*self.num_samples
                new_row["conditional probabilities"] = [0]*len(self.pool_ids)
                new_row["agent distribution"] = [0]*len(self.pool_ids)
                new_row["MLE probabilities"] = [0]*len(self.pool_ids)

                # add incomplete row to make use of functions below
                self.move_tracking_table = self.move_tracking_table.append(pd.Series(data=new_row, name=self.move_idx)) # https://stackoverflow.com/questions/39998262/append-an-empty-row-in-dataframe-using-pandas

                # Generate samples with corresponding conditionals
                self._hand_sampler(self.move_idx, self.num_samples)

                # generate average conditional
                sample_conditionals = [sample.conditional_probs for sample in self.move_tracking_table.loc[self.move_idx, "generated samples"]]
                new_conditional = np.mean(np.array(sample_conditionals), axis=0)

                self.move_tracking_table.at[self.move_idx, "conditional probabilities"] = new_conditional

                #### SCALING ####
                pmax = np.max(new_conditional)

                if pmax < 1/(2 * len(new_conditional)):
                        new_conditional = np.ones(len(new_conditional))
                
                # get prior using the average conditional and previous prior if available
                if self.move_idx == 0:
                        self.move_tracking_table.at[self.move_idx,"agent distribution"] = self._makeprob(new_conditional)
                        self.move_tracking_table.at[self.move_idx,"MLE probabilities"] = new_conditional
                else:
                        prev_row = self.move_tracking_table.loc[self.move_idx - 1]
                        prior = prev_row["agent distribution"]
                        prior2 = prev_row["MLE probabilities"]
                        updated_prior = new_conditional*prior
                        self.move_tracking_table.at[self.move_idx,"agent distribution"] = self._makeprob(updated_prior)
                        self.move_tracking_table.at[self.move_idx,"MLE probabilities"] = (new_conditional + prior2*self.move_idx)/(self.move_idx + 1)

                self.move_idx += 1


        
        def _get_action_helper(self):
                '''
                Using the behavior clones from CHIEF's perspective, choose an action
                Compute weighted average based on probabilities of each strategy for teammate
                Ideal case: 100% certain of right strategy, just do what matching strategy entails
                '''

                action_to_play = np.zeros(20)
                agent_ids = sorted(self.pool_ids)

                if len(self.move_tracking_table) > 0:
                        agent_weights = self.move_tracking_table.iloc[-1]["agent distribution"]
                else:
                        agent_weights = np.ones(len(agent_ids))

                for i, agent_id in enumerate(agent_ids):
                        bc_output = BehaviorClone.sequential_predict(agent_id,
                                                                     self.sequential_gamestates_chief_persp,
                                                                     self.sequential_playermodels_chief_persp,
                                                                     self.sequential_partnerknowledgemodels_chief_persp)
                        actionvalues = np.zeros(20)

                        print(max(bc_output, key=bc_output.get))

                        for action in bc_output:
                                actionvalues[self._action_to_key(action)] = float(bc_output[action])
                        
                        action_to_play += self._makeprob(actionvalues)*agent_weights[i]

                action_to_play = self._makeprob(action_to_play)
                action_key = np.argmax(action_to_play)
                return self._action_from_key(action_key, self.partner_nr)


        def _make_partner_knowledge_model(self, game_state):
                '''
                Copy of the function in hanabi.py
                '''

                partner_knowledge_model = dict()

                for possible_action in game_state.get_valid_actions():
                        if possible_action.type in [HINT_COLOR, HINT_NUMBER]:
                                partner_knowledge_model[possible_action] = apply_hint_to_knowledge(
                                        possible_action, game_state.hands, game_state.all_knowledge
                                )
                return partner_knowledge_model

        def _weighted_sample(self, choices, probs):
                '''
                Compute a weighted sample of arguments, handling floating point errors

                choices: list of objects to choose from
                probs: list of respective probabilities for each item in "choices"
                '''

                x = random.random()

                for idx, p in enumerate(probs):
                        x -= p

                        if x < 0:
                                return choices[idx]

                return choices[-1]

        def _makeprob(self, L):
                '''
                Generated a normalized probability distribution

                L: list of values to normalize into a probability distribution
                '''

                x = np.array(L).flatten()
                p = x/np.sum(x)

                if (np.sum(x) == 0):
                        return np.around(np.ones(len(x))/len(x), decimals=5)

                return p

        def _action_to_key(self, action):
                '''
                Generate a numerical key from "action" (Action object)
                '''


                if action.type == PLAY:
                        j = action.cnr
                elif action.type == DISCARD:
                        j = action.cnr
                elif action.type == HINT_NUMBER:
                        j = action.num - 1
                else:
                        j = action.col

                return action.type*5 + j

        def _action_from_key(self, action_key, partner_nr):
                '''
                Generate an Action object from "action_key"

                action_key: Assumed to be generated by _action_to_key
                partner_nr: Partner's number in the game
                        --> Used to make sure Action arguments are consistent
                '''

                action_type = action_key//5
                action_idx = action_key%5
                pnr = 1 - partner_nr if action_type in [PLAY, DISCARD] else partner_nr

                return Action(action_type, pnr, action_idx, action_idx + 1, action_idx)

        def _sample_hash(self, sample):
                temp = sample.card_vals
                return str(temp)

        def _new_sample(self, new_knowledge):
                output_dict = dict()

                for card in new_knowledge:
                        card_idx = self._weighted_sample(CardChoices, self._makeprob(new_knowledge[card])) # https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
                        card_val = (card_idx//5, card_idx%5 + 1)
                        output_dict[card] = card_val

                return Sample(output_dict, None)

        def _values_to_probs(self, actionvalues):
                values = min(actionvalues.values())*np.ones(20)

                for action in actionvalues:
                        values[self._action_to_key(action)] = actionvalues[action]

                values +=  0 - min(values)
                values /= max(values) - min(values)
                E = np.exp(values * BOLTZMANN_CONSTANT)
                return E/np.sum(E)

        def _gen_hand(self, samp_values, iloc_idx):
                hand_ids = self.move_tracking_table.iloc[iloc_idx]["card ids"]
                hand = [None]*len(hand_ids)

                for card_id in hand_ids:
                        hand[hand_ids[card_id]] = samp_values[card_id]

                return hand

        def _agent_probs(self, samp_values, move_idx): ## THIS ASSUMES THAT WE ONLY HAVE ONE TEAMMATE
                action_idx = int(self.move_tracking_table.loc[move_idx,"move"])

                game_states = []
                base_player_models = []

                for idx in range(move_idx+1):
                        game_state_ref, base_player_model_ref = self.move_tracking_table.iloc[idx]["observable game state"]
                        game_state = deepcopy(game_state_ref)
                        base_player_model = deepcopy(base_player_model_ref)

                        for i in range(len(game_state.hands)):
                                if game_state.hands[i] is None:
                                        game_state.hands[i] = self._gen_hand(samp_values, idx)

                        game_states.append(game_state)
                        base_player_models.append(base_player_model)

                probs = []
                agent_ids = sorted(self.pool_ids)

                for agent_id in agent_ids:
                        bc_output = BehaviorClone.sequential_predict(agent_id, game_states, base_player_models, [self._make_partner_knowledge_model(gs) for gs in game_states])
                        actionvalues = np.zeros(20)

                        for action in bc_output:
                                actionvalues[self._action_to_key(action)] = bc_output[action]

                        probs.append(self._makeprob(actionvalues)[action_idx])

                return np.array(probs)

        def _hand_sampler(self, move_idx, number_needed):
                existing_samples = self.move_tracking_table.loc[move_idx,"generated samples"]
                new_samples = []
                stored_samples = dict()
                new_knowledge = self.total_card_knowledge

                for sample in existing_samples:
                        if sample == None:
                                continue

                        if sample.consistent(new_knowledge):
                                new_samples.append(deepcopy(sample))
                                stored_samples[self._sample_hash(sample)] = sample.conditional_probs
                
                n = len(new_samples)

                for i in range(number_needed - n):
                        # generate unique sample
                        new_samp = self._new_sample(new_knowledge)
                        h = self._sample_hash(new_samp)

                        # compute conditional game state
                        if h in stored_samples:
                                new_conditional = stored_samples[h]
                        else:
                                new_conditional = self._agent_probs(new_samp.card_vals, move_idx)
                                stored_samples[h] = new_conditional

                        new_samp.conditional_probs = new_conditional
                        new_samples.append(new_samp)

                self.move_tracking_table.at[move_idx,"generated samples"] = new_samples


        def _rollforward_move_tracker(self, move_index, most_recent_move, cards_affected_with_new_knowledge):
                for table_idx in range(move_index, most_recent_move):                   
                        # update hand knowledge
                        for card, new_k in cards_affected_with_new_knowledge:
                                if card in self.move_tracking_table.at[table_idx, "card ids"]:
                                        pos_idx = self.move_tracking_table.loc[table_idx,"card ids"][card]
                                        self.move_tracking_table.at[table_idx,"hand knowledge"][pos_idx] = new_k


                        # use _hand_sampler to get new samples where needed
                        ### num_possibilities = np.prod(np.sum(self.move_tracking_table.loc[table_idx]["hand knowledge"], axis=(1,2)))
                        agent_copies = self._hand_sampler(table_idx, self.num_samples)


                        # compute new average conditional probabilities
                        sample_conditionals = [sample.conditional_probs for sample in self.move_tracking_table.loc[table_idx]["generated samples"]]
                        new_conditional = np.mean(np.array(sample_conditionals), axis=0)
                        self.move_tracking_table.at[table_idx,"conditional probabilities"] = new_conditional

                        #### SCALING ####
                        pmax = np.max(new_conditional)

                        if pmax < 1/(2 * len(new_conditional)):
                                new_conditional = np.ones(len(new_conditional))

                        # get prior distribution
                        prev_idx = table_idx - 1

                        if prev_idx >= 0:
                                prev_prior = self.move_tracking_table.loc[prev_idx]["agent distribution"]
                                prev_prior2 = self.move_tracking_table.loc[prev_idx]["MLE probabilities"]
                        else:
                                pool_size = len(self.pool_ids)
                                prev_prior = np.ones(pool_size)/pool_size
                                prev_prior2 = np.zeros(pool_size)


                        # update current agent distribution using updated conditional and previous prior
                        new_prior_pre = new_conditional * prev_prior
                        new_prior = self._makeprob(new_prior_pre)
                        self.move_tracking_table.at[table_idx,"agent distribution"] = new_prior
                        self.move_tracking_table.at[table_idx,"MLE probabilities"] = (new_conditional + prev_prior2*table_idx)/(table_idx + 1)