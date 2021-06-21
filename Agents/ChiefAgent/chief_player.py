from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from Agents.ChiefAgent.player_pool import PlayerPool
import pandas as pd
import numpy as np
import random
from copy import deepcopy

STARTING_COLUMNS_MOVETRACKING = {"move_idx":[], "move": [], "observable game state":[], "card ids":[], "hand knowledge":[], "agent distribution":[], "conditional probabilities":[], "generated samples":[], "agent state copies":[]}
NUM_SAMPLES = 10

CardChoices = []

for i in range(25):
	CardChoices.append(i)

# Note: hand is formmated as [(color, number)] - can use indices for color and indices + 1 for number

class Sample(object):
	def __init__(self, hand, conditional_probs):
		self.hand = hand
		self.conditional_probs = conditional_probs

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
		self.player_pool = PlayerPool(name, pnr, pool_file)
		self.card_ids = dict()
		self.new_card_id = 0
		self.move_idx = 0
		self.store_agent_copies_flag = False
		self.prev_knowledge = dict()
		self.drawn_dict = dict()
		self.hints_to_partner = []

	def get_action(self, game_state, player_model, action_default=None):
		if action_default is None:
			action = self.get_action_helper(game_state, player_model)
		else:
			action = action_default

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
					del self.drawn_dict[cid]
					del self.prev_knowledge[cid]

			new_card_ids[self.new_card_id] = len(player_model.knowledge) - 1
			self.drawn_dict[self.new_card_id] = None
			self.prev_knowledge[self.new_card_id] = None
			self.new_card_id += 1
			self.card_ids = new_card_ids
			store_agent_copies_flag = True
		else:
			self.hints_to_partner.append((self.pnr, action))

		return action


	def get_action_helper(self, game_state, player_model):
		pass

	def inform(self, action, player, game_state, player_model):
		changed_cards = []
		drawn_move = []

		# Detecting if any card information has changed that can be used to update previous data
		for c in self.card_ids:
			new_k = player_model.get_knowledge()[self.card_ids[c]]

			if self.drawn_dict[c] == None:
				self.drawn_dict[c] = self.move_idx
				self.prev_knowledge[c] = new_k
			elif new_k != self.prev_knowledge[c]:
				changed_cards.append((c, new_k))
				drawn_move.append(self.drawn_dict[c])

		if (len(changed_cards) > 0):
			self.rollforward_move_tracker(min(drawn_move), self.move_idx, changed_cards)


		# Creating new row for move tracking table
		new_row = dict()
		new_row["move"] = self.action_to_key(action)

		modified_game_state = deepcopy(game_state) # https://stackoverflow.com/questions/48338847/how-to-copy-a-class-instance-in-python
		modified_game_state.hands = [None if a != [] else [] for a in game_state.hands]
		partners_hints = self.hints_to_partner
		modified_player_model = BasePlayerModel(self.partner_nr, game_state.all_knowledge[self.partner_nr], self.hints_to_partner, player_model.get_actions())
		new_row["observable game state"] = (modified_game_state, modified_player_model)

		new_row["card ids"] = self.card_ids
		new_row["hand knowledge"] = player_model.get_knowledge()

		# store agent copies if flagged
		if self.store_agent_copies_flag:
			self.store_agent_copies_flag = False
			new_row["agent state copies"] = self.player_pool.copies()
		else:
			new_row["agent state copies"] = None

		new_row["generated samples"] = []

		# add incomplete row to make use of functions below
		self.move_tracking_table = self.move_tracking_table.append(pd.Series(data=new_row, name=self.move_idx)) # https://stackoverflow.com/questions/39998262/append-an-empty-row-in-dataframe-using-pandas

		# Generate samples with corresponding conditionals
		current_row = self.move_tracking_table.loc[self.move_idx]
		copies = self.player_pool.copies()
		self.hand_sampler(current_row, NUM_SAMPLES, copies)

		# generate average conditional
		sample_conditionals = [sample.conditional_probs for sample in current_row["generated samples"]]
		new_conditional = np.mean(np.array(sample_conditionals), axis=0)
		current_row["conditional probabilities"] = new_conditional
		
		# get prior using the average conditional and previous prior if available
		if self.move_idx == 0:
			current_row["agent distribution"] = new_conditional
		else:
			prev_row = self.move_tracking_table.loc[self.move_idx - 1]
			prior = prev_row["agent distribution"]
			updated_prior = new_conditional*prior
			current_row["agent distribution"] = self.makeprob(updated_prior)

		self.move_idx += 1

	def makeprob(self, L):
		x = np.array(L).flatten()
		p = x/np.sum(x)

		if np.sum(p) != 1:
			p[-1] = 1 - np.sum(p[:-1])

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
			card_idx = np.random.choice(CardChoices, p=self.makeprob(card)) # https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
			card = (card_idx//5, card_idx%5 + 1)
			new_samp.append(card)

		return Sample(new_samp, None)

	def values_to_probs(self, actionvalues):
		values = np.zeros(20)

		for action in actionvalues:
			values[self.action_to_key(action)] = actionvalues[action]

		nonnegative_vals = values + min(values)
		return nonnegative_vals/np.sum(nonnegative_vals)

	def agent_probs(self, hand, row_ref, agent_copies): ## THIS ASSUMES THAT WE ONLY HAVE ONE TEAMMATE
		game_state, base_player_model = row_ref["observable game state"]
		action_idx = int(row_ref["move"])

		for i in range(len(game_state.hands)):
			if game_state.hands[i] is None:
				game_state.hands[i] = hand

		probs = []
		updated_agents = []

		for agent in agent_copies:
			temp_agent = deepcopy(agent)
			values = temp_agent.get_action(game_state, base_player_model)
			probs.append(self.values_to_probs(values)[action_idx])
			updated_agents.append(temp_agent)

		return np.array(probs), updated_agents

	def hand_sampler(self, row_ref, number_needed, agent_copies):
		existing_samples = row_ref["generated samples"]
		new_samples = []
		stored_samples = dict()
		new_knowledge = row_ref["hand knowledge"]

		for sample in existing_samples:
			if sample.consistent(new_knowledge):
				new_samples.append(deepcopy(sample))
				stored_samples[self.sample_hash(sample)] = sample.conditional_probs
		
		n = len(new_samples)
		updated_copies = agent_copies

		for i in range(number_needed - n):
			# generate unique sample
			new_samp = self.new_sample(new_knowledge)
			h = self.sample_hash(new_samp)

			# compute conditional game state
			if h in stored_samples:
				new_conditional = stored_samples[h]
			else:
				new_conditional, updated_copies = self.agent_probs(new_samp.hand, row_ref, updated_copies)
				stored_samples[h] = new_conditional

			new_samp.conditional_probs = new_conditional
			new_samples.append(new_samp)

		row_ref["generated samples"] = new_samples

		if updated_copies is None:
			_, updated_copies = self.agent_probs(new_samples[0].hand, row_ref, agent_copies)

		return updated_copies


	def rollforward_move_tracker(self, move_index, most_recent_move, cards_affected_with_new_knowledge):
		agent_copies = self.move_tracking_table.loc[move_index]["agent state copies"]

		for table_idx in range(move_index, most_recent_move):
			current_row = self.move_tracking_table.loc[table_idx] # This is passed around everywhere as reference (including into functions)
			
			# update hand knowledge
			for card, new_k in cards_affected_with_new_knowledge:
				pos_idx = current_row["card ids"][card]
				current_row["hand knowledge"][pos_idx] = new_k


			# use hand_sampler to get new samples where needed
			### num_possibilities = np.prod(np.sum(current_row["hand knowledge"], axis=(1,2)))
			agent_copies = self.hand_sampler(current_row, NUM_SAMPLES)


			# compute new average conditional probabilities
			sample_conditionals = [sample.conditional_probs for sample in current_row["generated samples"]]
			new_conditional = np.mean(np.array(sample_conditionals), axis=0)
			current_row["conditional probabilities"] = new_conditional


			# get prior distribution
			prev_idx = table_idx - 1

			if prev_idx >= 0:
				prev_prior = self.move_tracking_table.loc[prev_idx]["agent distribution"]
			else:
				pool_size = self.player_pool.get_size()
				prev_prior = np.ones(pool_size)/pool_size


			# update current agent distribution using updated conditional and previous prior
			new_prior_pre = new_conditional * prev_prior
			new_prior = self.makeprob(new_prior_pre)
			current_row["agent distribution"] = new_prior
