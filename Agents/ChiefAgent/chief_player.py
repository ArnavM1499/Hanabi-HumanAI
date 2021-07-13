from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from Agents.ChiefAgent.player_pool import PlayerPool
import pandas as pd
import numpy as np
import random
from copy import deepcopy

STARTING_COLUMNS_MOVETRACKING = {"move_idx":[], "move": [], "observable game state":[], "card ids":[], "hand knowledge":[], "agent distribution":[], "conditional probabilities":[], "MLE probabilities":[], "generated samples":[], "agent state copies":[]}
NUM_SAMPLES = 25
BOLTZMANN_CONSTANT = 30

CardChoices = []

for i in range(25):
	CardChoices.append(i)

# Note: hand is formmated as [(color, number)] - can use indices for color and indices + 1 for number

class Sample(object):
	def __init__(self, hand, conditional_probs, MLE_probs):
		self.hand = hand
		self.conditional_probs = conditional_probs
		self.MLE_probs = MLE_probs

	def consistent(self, knowledge):
		for i, card in enumerate(self.hand):
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
		self.prev_knowledge = dict()
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
				self.prev_knowledge[self.new_card_id] = k
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
					del self.prev_knowledge[cid]

			new_card_ids[self.new_card_id] = len(player_model.knowledge) - 1
			self.drawn_dict[self.new_card_id] = None
			self.prev_knowledge[self.new_card_id] = None
			self.new_card_id += 1
			self.card_ids = new_card_ids
		else:
			self.hints_to_partner.append((self.pnr, action))


		return action


	def get_action_helper(self, game_state, player_model):
		pass

	def inform(self, action, player, game_state, player_model):
		changed_cards = []
		drawn_move = []

		if player == self.pnr: # chief agent just took action
			self.game_state_before_move = deepcopy(game_state)
			self.player_model_before_move = deepcopy(player_model)

			if action.type == PLAY:
				temp = np.zeros(shape=(5,5))
				card = game_state.played[-1]
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

				modified_game_state.hands = [self.new_sample(player_model.get_knowledge()).hand if a == [] else [] for a in game_state.hands]
				partners_hints = deepcopy(self.hints_to_partner)
				modified_player_model = BasePlayerModel(self.partner_nr, game_state.all_knowledge[self.partner_nr], self.hints_to_partner, player_model.get_actions())
				agent.inform(action, self.pnr, modified_game_state, modified_player_model)

			return

		# Detecting if any card information has changed that can be used to update previous data
		mark_delete = []

		for c in self.card_ids:
			if self.card_ids[c] >= len(player_model.get_knowledge()):
				mark_delete.append(c)
				continue

			new_k = player_model.get_knowledge()[self.card_ids[c]]

			if self.drawn_dict[c] == None:
				self.drawn_dict[c] = self.move_idx
				self.prev_knowledge[c] = new_k
			elif new_k != self.prev_knowledge[c]:
				changed_cards.append((c, new_k))
				drawn_move.append(self.drawn_dict[c])

		for marked_c in mark_delete:
			del self.card_ids[marked_c] # new card picked up when game didn't actually have one


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

		# Creating new row for move tracking table
		new_row = dict()
		new_row["move"] = self.action_to_key(action)
		# print(action, self.action_to_key(action))

		## storing game state before move for get_action
		modified_game_state = deepcopy(self.game_state_before_move)
		modified_game_state.hands = [None if a == [] else [] for a in self.game_state_before_move.hands]
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
		
		for a in modified_game_state2.valid_actions:
			a.pnr = self.pnr

		partners_hints = self.hints_to_partner
		modified_player_model2 = BasePlayerModel(self.partner_nr, game_state.all_knowledge[self.partner_nr], self.hints_to_partner, player_model.get_actions())

		for agent in self.player_pool.get_agents():
			game_state_input = deepcopy(modified_game_state2)
			game_state_input.hands = [self.new_sample(player_model.get_knowledge()).hand if a == [] else [] for a in game_state.hands]
			prev_game = deepcopy(modified_game_state)
			prev_game.hands = [self.new_sample(player_model.get_knowledge()).hand if a == None else [] for a in modified_game_state.hands]
			d = agent.get_action(prev_game, modified_player_model)
			# print(max(d, key=d.get), type(agent).__name__, prev_game.hands)
			agent.inform(action, player, game_state_input, modified_player_model2)

		# add incomplete row to make use of functions below
		self.move_tracking_table = self.move_tracking_table.append(pd.Series(data=new_row, name=self.move_idx)) # https://stackoverflow.com/questions/39998262/append-an-empty-row-in-dataframe-using-pandas

		# Generate samples with corresponding conditionals
		self.hand_sampler(self.move_idx, NUM_SAMPLES)

		# generate average conditional
		sample_conditionals = [sample.conditional_probs for sample in self.move_tracking_table.loc[self.move_idx, "generated samples"]]
		new_conditional = np.mean(np.array(sample_conditionals), axis=0)
		self.move_tracking_table.at[self.move_idx, "conditional probabilities"] = new_conditional

		# generate average MLE prob
		sample_MLEs = [sample.MLE_probs for sample in self.move_tracking_table.loc[self.move_idx, "generated samples"]]
		new_MLE = np.mean(np.array(sample_MLEs), axis=0)
		
		# get prior using the average conditional and previous prior if available
		if self.move_idx == 0:
			self.move_tracking_table.at[self.move_idx,"agent distribution"] = new_conditional
			self.move_tracking_table.at[self.move_idx,"MLE probabilities"] = new_MLE
		else:
			prev_row = self.move_tracking_table.loc[self.move_idx - 1]
			prior = prev_row["agent distribution"]
			prior2 = prev_row["MLE probabilities"]
			updated_prior = new_conditional*prior
			self.move_tracking_table.at[self.move_idx,"agent distribution"] = self.makeprob(updated_prior)
			self.move_tracking_table.at[self.move_idx,"MLE probabilities"] = (new_MLE + prior2*self.move_idx)/(self.move_idx + 1)

		self.move_idx += 1

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
			i = 0
			j = action.cnr
		elif action.type == DISCARD:
			i = 1
			j = action.cnr
		elif action.type == HINT_NUMBER:
			i = 2
			j = action.num - 1
		else:
			i = 3
			j = action.col

		return i*5 + j

	def sample_hash(self, sample):
		temp = sample.hand
		return str(temp)

	def new_sample(self, new_knowledge):
		new_samp = []

		for card in new_knowledge:
			card_idx = self.weighted_sample(CardChoices, self.makeprob(card)) # https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
			card = (card_idx//5, card_idx%5 + 1)
			new_samp.append(card)

		return Sample(new_samp, None, None)

	# def values_to_probs(self, actionvalues):
	# 	values = np.zeros(20)

	# 	for action in actionvalues:
	# 		values[self.action_to_key(action)] = actionvalues[action]

	# 	nonnegative_vals = values - min(values)
	# 	return nonnegative_vals/np.sum(nonnegative_vals)

	def values_to_probs(self, actionvalues):
		values = min(actionvalues.values())*np.ones(20)

		for action in actionvalues:
			values[self.action_to_key(action)] = actionvalues[action]

		E = np.exp(values * BOLTZMANN_CONSTANT)
		return E/np.sum(E)

	def agent_probs(self, hand, move_idx): ## THIS ASSUMES THAT WE ONLY HAVE ONE TEAMMATE
		game_state_ref, base_player_model_ref = self.move_tracking_table.loc[move_idx,"observable game state"]
		agent_copies = self.move_tracking_table.loc[move_idx,"agent state copies"]
		action_idx = int(self.move_tracking_table.loc[move_idx,"move"])

		game_state = deepcopy(game_state_ref)
		base_player_model = deepcopy(base_player_model_ref)

		for i in range(len(game_state.hands)):
			if game_state.hands[i] is None:
				game_state.hands[i] = hand

		probs = []
		MLEs = []

		for agent in agent_copies:
			temp_agent = deepcopy(agent)
			values = temp_agent.get_action(game_state, base_player_model)

			# print(type(agent).__name__, game_state.hands, game_state.board, [round(a,2) for a in values.values()])

			probs.append(self.values_to_probs(values)[action_idx])
			prob_array = np.array(self.values_to_probs(values))
			MLEs.append(float((prob_array[action_idx] == max(prob_array))/(np.sum(prob_array == max(prob_array)))))

		return np.array(probs), np.array(MLEs)

	def hand_sampler(self, move_idx, number_needed):
		existing_samples = self.move_tracking_table.loc[move_idx,"generated samples"]
		new_samples = []
		stored_samples = dict()
		new_knowledge = self.move_tracking_table.loc[move_idx,"hand knowledge"]

		for sample in existing_samples:
			if sample == None:
				continue

			if sample.consistent(new_knowledge):
				new_samples.append(deepcopy(sample))
				stored_samples[self.sample_hash(sample)] = (sample.conditional_probs, sample.MLE_probs)
		
		n = len(new_samples)

		for i in range(number_needed - n):
			# generate unique sample
			new_samp = self.new_sample(new_knowledge)
			h = self.sample_hash(new_samp)

			# compute conditional game state
			if h in stored_samples:
				new_conditional, new_MLE = stored_samples[h]
			else:
				new_conditional, new_MLE = self.agent_probs(new_samp.hand, move_idx)
				stored_samples[h] = (new_conditional, new_MLE)

			new_samp.conditional_probs = new_conditional
			new_samp.MLE_probs = new_conditional
			# print(new_samp.hand, new_samp.MLE_probs)
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
			sample_MLEs = [sample.MLE_probs for sample in self.move_tracking_table.loc[table_idx]["generated samples"]]
			new_MLE = np.mean(np.array(sample_MLEs), axis=0)
			self.move_tracking_table.at[table_idx,"conditional probabilities"] = new_conditional


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
			self.move_tracking_table.at[table_idx,"agent distribution"] = new_prior
			self.move_tracking_table.at[table_idx,"MLE probabilities"] = (new_MLE + prev_prior2*table_idx)/(table_idx + 1)