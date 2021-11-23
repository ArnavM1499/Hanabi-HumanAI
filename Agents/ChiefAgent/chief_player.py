from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from Agents.ChiefAgent.player_pool import PlayerPool
from game_net.behavior_clone import BehaviorClone
import pandas as pd
import numpy as np
import random
from copy import deepcopy
from scipy.stats import entropy

STARTING_COLUMNS_MOVETRACKING = {"move_idx":[], "move": [], "observable game state":[], "card ids":[], "hand knowledge":[], "agent distribution":[], "conditional probabilities":[], "MLE probabilities":[], "generated samples":[], "agent state copies":[]}
NUM_SAMPLES = 20
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
	def __init__(self, name, pnr, pool_file):
		self.name = name
		self.pnr = pnr
		self.partner_nr = (pnr + 1) % 2
		self.move_tracking_table = pd.DataFrame(data=STARTING_COLUMNS_MOVETRACKING)
		self.move_tracking_table = self.move_tracking_table.set_index("move_idx")
		self.player_pool = PlayerPool("pool_agent", self.partner_nr, pool_file)
		self.card_ids = dict()
		self.new_card_id = 0
		self.move_idx = 0
		self.total_card_knowledge = dict()
		self.drawn_dict = dict()
		self.hints_to_partner = []
		self.game_state_before_move = None
		self.player_model_before_move = None
		self.played_or_discarded_card = None

	def get_action(self, game_state, player_model, action_default=None):
		if action_default is None:
			action = self.get_action_helper(game_state, player_model)
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


	def get_action_helper(self, game_state, player_model):
		best_team_reward = -5
		best_action = 0

		print("Deciding action...")

		for va in game_state.valid_actions:
			team_reward = 0

			for i in range(NUM_SAMPLES):	
				new_samp = self.new_sample(self.total_card_knowledge)
				sampled_vals = self.new_sample_original(player_model.get_knowledge())

				for cid in self.card_ids:
					if cid in self.total_card_knowledge:
						sampled_vals[self.card_ids[cid]] = new_samp.card_vals[cid]

				game_state_next, player_model_next, chief_reward = self.simulate_move(game_state, player_model, va, sampled_vals)
				teammate_action = self.get_prediction(game_state_next, player_model_next, new_samp, sampled_vals)
				team_reward += self.eval_action(game_state_next.board, game_state.hands[self.partner_nr], teammate_action) + chief_reward

			print("Trying action", va, "reward is", team_reward)

			if team_reward > best_team_reward:
				best_team_reward = team_reward
				best_action = va

		return best_action

	def eval_action(self, pred_board, curr_team_hand, action):
		if action//5 == PLAY:
			if curr_team_hand[action%5][1] == pred_board[curr_team_hand[action%5][0]][1]+1:
				return 1
			else:
				return -1

		return 0


	def inform(self, action, player, game_state, player_model):
		changed_cards = []
		drawn_move = []

		#############################################################
		### Update agents in pool if chief agent just took action ###
		#############################################################
		if player == self.pnr:
			self.game_state_before_move = deepcopy(game_state)
			self.player_model_before_move = deepcopy(player_model)

			if action.type == PLAY:
				temp = np.zeros(shape=(5,5))
				card = game_state.card_changed
				temp[card[0]][card[1] - 1] = 1
				self.played_or_discarded_card[1] = temp
			elif action.type == DISCARD:
				temp = np.zeros(shape=(5,5))
				card = game_state.trash[-1]
				temp[card[0]][card[1] - 1] = 1
				self.played_or_discarded_card[1] = temp

			for agent in self.player_pool.get_agents():
				modified_game_state = deepcopy(game_state)

				for a in modified_game_state.valid_actions:
					a.pnr = self.pnr

				modified_game_state.hands = [self.new_sample_original(player_model.get_knowledge()) if a == [] else [] for a in modified_game_state.hands]
				partners_hints = deepcopy(self.hints_to_partner)
				modified_player_model = BasePlayerModel(self.partner_nr, game_state.all_knowledge[self.partner_nr], self.hints_to_partner, player_model.get_actions())
				agent.inform(action, self.pnr, modified_game_state, modified_player_model)

			return

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
		if (len(changed_cards) > 0) or self.played_or_discarded_card != None:
			if (self.played_or_discarded_card):
				changed_cards.append(tuple(self.played_or_discarded_card[:-1]))

			if len(drawn_move) == 0:
				starting_move = self.played_or_discarded_card[2]
			elif self.played_or_discarded_card == None:
				starting_move = min(drawn_move)
			else:
				starting_move = min(min(drawn_move), self.played_or_discarded_card[2])

			self.rollforward_move_tracker(starting_move, self.move_idx, changed_cards)
			self.played_or_discarded_card = None


		##################################################################
		### Creating initial row for this move index in tracking table ###
		##################################################################
		new_row = dict()
		new_row["move"] = self.action_to_key(action)

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
		new_row["agent state copies"] = self.player_pool.copies()
		new_row["generated samples"] = [None]*NUM_SAMPLES
		new_row["conditional probabilities"] = [0]*self.player_pool.get_size()
		new_row["agent distribution"] = [0]*self.player_pool.get_size()
		new_row["MLE probabilities"] = [0]*self.player_pool.get_size()


		## Informing player pool of "their own" actions
		modified_game_state2 = deepcopy(game_state) # https://stackoverflow.com/questions/48338847/how-to-copy-a-class-instance-in-python
		modified_game_state2.hands[self.pnr] = []

		for a in modified_game_state2.valid_actions:
			a.pnr = self.pnr

		partners_hints = self.hints_to_partner
		modified_player_model2 = BasePlayerModel(self.partner_nr, game_state.all_knowledge[self.partner_nr], self.hints_to_partner, player_model.get_actions())

		for agent in self.player_pool.get_agents():
			game_state_input = deepcopy(modified_game_state2)
			game_state_input.hands = [self.new_sample_original(player_model.get_knowledge()) if a == [] else [] for a in modified_game_state2.hands]
			prev_game = deepcopy(modified_game_state)
			prev_game.hands = [self.new_sample_original(player_model.get_knowledge()) if a == None else [] for a in modified_game_state.hands]
			d = agent.get_action(prev_game, modified_player_model)
			agent.inform(action, player, game_state_input, modified_player_model2)

		# add incomplete row to make use of functions below
		self.move_tracking_table = self.move_tracking_table.append(pd.Series(data=new_row, name=self.move_idx)) # https://stackoverflow.com/questions/39998262/append-an-empty-row-in-dataframe-using-pandas

		# Generate samples with corresponding conditionals
		self.hand_sampler(self.move_idx, NUM_SAMPLES)

		# generate average conditional
		sample_conditionals = [sample.conditional_probs for sample in self.move_tracking_table.loc[self.move_idx, "generated samples"]]
		new_conditional = np.mean(np.array(sample_conditionals), axis=0)

		#### SCALING ####
		pmax = np.max(new_conditional)

		if pmax < 1/(2 * len(new_conditional)):
			new_conditional = np.ones(len(new_conditional))
		
		self.move_tracking_table.at[self.move_idx, "conditional probabilities"] = new_conditional
		print("NEW CONDITIONAL:", np.vectorize(lambda a: round(a,2))(new_conditional))
		
		# get prior using the average conditional and previous prior if available
		if self.move_idx == 0:
			self.move_tracking_table.at[self.move_idx,"agent distribution"] = self.makeprob(new_conditional)
			self.move_tracking_table.at[self.move_idx,"MLE probabilities"] = new_conditional
		else:
			prev_row = self.move_tracking_table.loc[self.move_idx - 1]
			prior = prev_row["agent distribution"]
			prior2 = prev_row["MLE probabilities"]
			updated_prior = new_conditional*prior
			print(np.vectorize(lambda a: round(a,2))(updated_prior), np.vectorize(lambda a: round(a,2))([prior]))
			self.move_tracking_table.at[self.move_idx,"agent distribution"] = self.makeprob(updated_prior)
			self.move_tracking_table.at[self.move_idx,"MLE probabilities"] = (new_conditional + prior2*self.move_idx)/(self.move_idx + 1)

		self.move_idx += 1

	def get_prediction(self, game_state, player_model, new_samp, sampled_vals):		
		modified_game_state = deepcopy(game_state)
		modified_player_model = deepcopy(player_model)

		##
		# Generate samples like hand_sampler but try to guess action this time
		##
		agent_ids = sorted(self.player_pool.get_player_dict().keys())

		game_states = []
		base_player_models = []

		for idx in range(len(self.move_tracking_table)+1):
			if idx == len(self.move_tracking_table):
				game_state_ref, base_player_model_ref = modified_game_state, modified_player_model
			else:
				game_state_ref, base_player_model_ref = self.move_tracking_table.iloc[idx]["observable game state"]
			
			game_state = deepcopy(game_state_ref)
			base_player_model = deepcopy(base_player_model_ref)

			for i in range(len(game_state.hands)):
				if game_state.hands[i] is None:
					if idx == len(self.move_tracking_table):
						game_state.hands[i] = sampled_vals
					else:
						game_state.hands[i] = self.gen_hand(new_samp.card_vals, idx)

			game_states.append(game_state)
			base_player_models.append(base_player_model)

		probs = np.zeros(20)
		agent_ids = sorted(self.player_pool.get_player_dict().keys())

		if len(self.move_tracking_table) > 0:
			agent_weights = self.move_tracking_table.iloc[-1]["agent distribution"]
		else:
			agent_weights = np.ones(len(agent_ids))

		for i, agent_id in enumerate(agent_ids):
			bc_output = BehaviorClone.sequential_predict(agent_id, game_states, base_player_models, [self._make_partner_knowledge_model(gs) for gs in game_states])
			actionvalues = np.zeros(20)

			for action in bc_output:
				actionvalues[self.action_to_key(action)] = float(bc_output[action])
			
			probs += self.makeprob(actionvalues)*agent_weights[i]

		prediction_vector = self.makeprob(probs)

		return np.argmax(prediction_vector)

	def _make_partner_knowledge_model(self, game_state):
		partner_knowledge_model = dict()

		for possible_action in game_state.get_valid_actions():
			if possible_action.type in [HINT_COLOR, HINT_NUMBER]:
				partner_knowledge_model[possible_action] = apply_hint_to_knowledge(
					possible_action, game_state.hands, game_state.all_knowledge
				)
		return partner_knowledge_model


	def entropy_of_knowledge(self, back_moves=1):
		# from https://en.wikipedia.org/wiki/Entropy_(information_theory)#Further_properties - total entropy of independent variables is just sum of entropies
		entropysum = 0
		back_moves_modified = min(len(self.move_tracking_table), back_moves)

		if len(self.move_tracking_table) == 0:
			return -1

		for hand in self.move_tracking_table.iloc[-1*back_moves_modified]["hand knowledge"]:
			entropysum += entropy(np.array(hand).flatten())

		return entropysum

	def entropy_of_pool(self, back_moves=1):
		back_moves_modified = min(len(self.move_tracking_table), back_moves)

		if len(self.move_tracking_table) == 0:
			return -1

		return entropy(self.move_tracking_table.iloc[-1*back_moves_modified]["agent distribution"])

	def weighted_sample(self, choices, probs): # making this because numpy.random.choice has annoying errors with floating point errors
		x = random.random()

		for idx, p in enumerate(probs):
			x -= p

			if x < 0:
				return choices[idx]

		return choices[-1] # in case there were any issues with floating points/should add logging for this at some point

	def makeprob(self, L):
		x = np.array(L).flatten()
		p = x/np.sum(x)

		if (np.sum(x) == 0):
			return np.around(np.ones(len(x))/len(x), decimals=5)

		return p

	def action_to_key(self, action):
		if action.type == PLAY:
			j = action.cnr
		elif action.type == DISCARD:
			j = action.cnr
		elif action.type == HINT_NUMBER:
			j = action.num - 1
		else:
			j = action.col

		return action.type*5 + j

	def sample_hash(self, sample):
		temp = sample.card_vals
		return str(temp)

	def new_sample_original(self, new_knowledge):
		hand = []

		for card in new_knowledge:
			card_idx = self.weighted_sample(CardChoices, self.makeprob(card)) # https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
			card_val = (card_idx//5, card_idx%5 + 1)
			hand.append(card_val)

		return hand

	def new_sample(self, new_knowledge):
		output_dict = dict()

		for card in new_knowledge:
			card_idx = self.weighted_sample(CardChoices, self.makeprob(new_knowledge[card])) # https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
			card_val = (card_idx//5, card_idx%5 + 1)
			output_dict[card] = card_val

		return Sample(output_dict, None)

	def values_to_probs(self, actionvalues):
		values = min(actionvalues.values())*np.ones(20)

		for action in actionvalues:
			values[self.action_to_key(action)] = actionvalues[action]

		values +=  0 - min(values)
		values /= max(values) - min(values)
		E = np.exp(values * BOLTZMANN_CONSTANT)
		return E/np.sum(E)

	def gen_hand(self, samp_values, iloc_idx):
		hand_ids = self.move_tracking_table.iloc[iloc_idx]["card ids"]
		hand = [None]*len(hand_ids)

		for card_id in hand_ids:
			hand[hand_ids[card_id]] = samp_values[card_id]

		return hand

	def agent_probs(self, samp_values, move_idx): ## THIS ASSUMES THAT WE ONLY HAVE ONE TEAMMATE
		action_idx = int(self.move_tracking_table.loc[move_idx,"move"])

		game_states = []
		base_player_models = []

		for idx in range(move_idx+1):
			game_state_ref, base_player_model_ref = self.move_tracking_table.iloc[idx]["observable game state"]
			game_state = deepcopy(game_state_ref)
			base_player_model = deepcopy(base_player_model_ref)

			for i in range(len(game_state.hands)):
				if game_state.hands[i] is None:
					# print("hello")
					game_state.hands[i] = self.gen_hand(samp_values, idx)

			game_states.append(game_state)
			base_player_models.append(base_player_model)

		probs = []
		agent_ids = sorted(self.player_pool.get_player_dict().keys())

		for agent_id in agent_ids:
			bc_output = BehaviorClone.sequential_predict(agent_id, game_states, base_player_models, [self._make_partner_knowledge_model(gs) for gs in game_states])
			actionvalues = np.zeros(20)

			for action in bc_output:
				actionvalues[self.action_to_key(action)] = bc_output[action]

			probs.append(self.makeprob(actionvalues)[action_idx])

		return np.array(probs)

	def hand_sampler(self, move_idx, number_needed):
		existing_samples = self.move_tracking_table.loc[move_idx,"generated samples"]
		new_samples = []
		stored_samples = dict()
		new_knowledge = self.total_card_knowledge

		for sample in existing_samples:
			if sample == None:
				continue

			if sample.consistent(new_knowledge):
				new_samples.append(deepcopy(sample))
				stored_samples[self.sample_hash(sample)] = sample.conditional_probs
		
		n = len(new_samples)

		for i in range(number_needed - n):
			# generate unique sample
			new_samp = self.new_sample(new_knowledge)
			h = self.sample_hash(new_samp)

			# compute conditional game state
			if h in stored_samples:
				new_conditional = stored_samples[h]
			else:
				new_conditional = self.agent_probs(new_samp.card_vals, move_idx)
				stored_samples[h] = new_conditional

			new_samp.conditional_probs = new_conditional
			new_samples.append(new_samp)

		self.move_tracking_table.at[move_idx,"generated samples"] = new_samples


	def rollforward_move_tracker(self, move_index, most_recent_move, cards_affected_with_new_knowledge):
		for table_idx in range(move_index, most_recent_move):			
			# update hand knowledge
			for card, new_k in cards_affected_with_new_knowledge:
				if card in self.move_tracking_table.at[table_idx, "card ids"]:
					pos_idx = self.move_tracking_table.loc[table_idx,"card ids"][card]
					self.move_tracking_table.at[table_idx,"hand knowledge"][pos_idx] = new_k


			# use hand_sampler to get new samples where needed
			### num_possibilities = np.prod(np.sum(self.move_tracking_table.loc[table_idx]["hand knowledge"], axis=(1,2)))
			agent_copies = self.hand_sampler(table_idx, NUM_SAMPLES)


			# compute new average conditional probabilities
			sample_conditionals = [sample.conditional_probs for sample in self.move_tracking_table.loc[table_idx]["generated samples"]]
			new_conditional = np.mean(np.array(sample_conditionals), axis=0)

			#### SCALING ####
			pmax = np.max(new_conditional)

			if pmax < 1/(2 * len(new_conditional)):
				new_conditional = np.ones(len(new_conditional))

			self.move_tracking_table.at[table_idx,"conditional probabilities"] = new_conditional
			print("NEW CONDITIONAL for MOVE", table_idx, ":", np.vectorize(lambda a: round(a,2))(new_conditional))


			# get prior distribution
			prev_idx = table_idx - 1

			if prev_idx >= 0:
				prev_prior = self.move_tracking_table.loc[prev_idx]["agent distribution"]
				prev_prior2 = self.move_tracking_table.loc[prev_idx]["MLE probabilities"]
			else:
				pool_size = self.player_pool.get_size()
				prev_prior = np.ones(pool_size)/pool_size
				prev_prior2 = np.zeros(pool_size)


			# update current agent distribution using updated conditional and previous prior
			new_prior_pre = new_conditional * prev_prior
			new_prior = self.makeprob(new_prior_pre)
			print(np.vectorize(lambda a: round(a,3))(new_prior))
			self.move_tracking_table.at[table_idx,"agent distribution"] = new_prior
			self.move_tracking_table.at[table_idx,"MLE probabilities"] = (new_conditional + prev_prior2*table_idx)/(table_idx + 1)

	def simulate_move(self, game_state, player_model, action, sampled_vals):
		hint_indices = []
		hint_lis = []
		reward = 0

		game_state = deepcopy(game_state)
		player_model = deepcopy(player_model)

		if action.type == DISCARD:
			card_discarded = sampled_vals[action.cnr]
			game_state.trash.append(card_discarded)
			game_state.all_knowledge[game_state.current_player][action.cnr] = initial_knowledge()
			game_state.card_changed = card_discarded
		elif action.type == PLAY:
			card_played = sampled_vals[action.cnr]
			game_state.card_changed = card_played
			game_state.all_knowledge[game_state.current_player][action.cnr] = initial_knowledge()

			if card_playable(card_played, game_state.board):
				game_state.played.append(card_played)
				game_state.board[card_played[0]] = (card_played[0], card_played[1])
				reward = 1
			else:
				game_state.trash.append(card_played)
				game_state.hits -= 1
				reward = -1
		elif action.type == HINT_NUMBER:
			game_state.num_hints -= 1
			hint_lis.append((game_state.current_player,action))
			slot_index = 0

			for (col, num), knowledge in zip(game_state.hands[action.pnr], game_state.all_knowledge[action.pnr]):
				if num == action.num:
					hint_indices.append(slot_index)
					for k in knowledge:
						for i in range(len(COUNTS)):
							if i + 1 != num:
								k[i] = 0
				else:
					for k in knowledge:
						k[action.num - 1] = 0

				slot_index += 1
		else:
			game_state.num_hints -= 1
			hint_lis.append((game_state.current_player,action))
			slot_index = 0

			for (col, num), knowledge in zip(
					game_state.hands[action.pnr], game_state.all_knowledge[action.pnr]
				):
				if col == action.col:
					hint_indices.append(slot_index)
					for i, k in enumerate(knowledge):
						if i != col:
							for i in range(len(k)):
								k[i] = 0
				else:
					for i in range(len(knowledge[action.col])):
						knowledge[action.col][i] = 0

				slot_index += 1

		player_model.actions[player_model.nr].append(action)
		player_model.knowledge = game_state.all_knowledge[1 - player_model.nr]
		player_model.nr = 1 - player_model.nr
		player_model.hints = self.hints_to_partner + hint_lis

		game_state.current_player += 1
		game_state.current_player %= len(game_state.hands)
		game_state.hinted_indices = hint_indices
		game_state.hands[player_model.nr] = []
		game_state.hands[1 - player_model.nr] = sampled_vals

		valid = []

		for i in range(len(game_state.hands[game_state.current_player])):
			valid.append(Action(PLAY, cnr=i))
			valid.append(Action(DISCARD, cnr=i))

		if game_state.num_hints > 0:
			for i in range(len(game_state.hands)):
				if i != game_state.current_player:
					for col in set(map(lambda colnum: colnum[0], game_state.hands[i])):
						valid.append(Action(HINT_COLOR, pnr=i, col=col))

					for num in set(map(lambda colnum: colnum[1], game_state.hands[i])):
						valid.append(Action(HINT_NUMBER, pnr=i, num=num))

		game_state.valid_actions = valid

		return game_state, player_model, reward