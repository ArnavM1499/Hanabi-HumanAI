from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from Agents.ChiefAgent.player_pool import PlayerPool
import pandas as pd
import numpy as np
import random

STARTING_COLUMNS_MOVETRACKING = {"move_idx":[], "move": [], "observable game state":[], "card ids":[], "hand knowledge":[], "prior distribution":[], "conditional distribution":[], "generated samples":[], "agent state copies"[]}
NUM_SAMPLES = 10

CardChoices = []

for i in range(5):
	for j in range(5):
		CardChoices.append((i,j+1))

# Note: hand is formmated as [(color, number)] - can use indices for color and indices + 1 for number

class Sample(object):
	def __init__(self, hand, conditional_distrib):
		self.hand = hand
		self.conditional_distrib = conditional_distrib

	def consistent(self, knowledge):
		for i, card in enumerate(self.hand):
			if knowledge[i][card[0]][card[1]-1] <= 0:
				return False

		return True

class ChiefPlayer(Player):
	def __init__(self, name, pnr):
		self.name = name
		self.pnr = pnr
		self.move_tracking_table = pd.DataFrame(data=STARTING_COLUMNS_MOVETRACKING)
		self.move_tracking_table = self.move_tracking_table.set_index("move_idx")
		self.player_pool = PlayerPool("agent_pool.json")
		self.card_ids = []
		self.new_card_id = 0
		self.move_idx = 0
		self.store_agent_copies_flag = False

	def get_action(self, game_state, player_model, for_testing_only=False, action_default=None):
		if action_default is None:
			action = self.get_action_helper(game_state, player_model)
		else:
			action = action_default

		# code for shifting card ids for keeping track of positions in case we play/discard
		# this is used for our agents which are meant to model the teammate's point of view
		if action.type == PLAY or action.type == DISCARD or for_testing_only:
			new_card_ids = []

			for idx in range(len(self.card_ids)):
				if idx != action.cnr:
					new_card_ids.append(self.card_ids[idx])

			new_card_ids.append(self.new_card_id)
			self.new_card_id += 1
			self.card_ids = new_card_ids
			store_agent_copies_flag = True

		return action


	def get_action_helper(self, game_state, player_model):
		pass

	def inform(self, action, player, game_state, player_model):
		new_row = dict()

		new_row["move"] = action_to_key(action)

		modified_game_state = 
		modified_player_model =
		new_row["observable game state"] = (modified_game_state, modified_player_model)

		new_row["card ids"] = self.card_ids
		new_row["hand knowledge"] = player_model.get_knowledge()

		# store agent copies if flagged
		if self.store_agent_copies_flag:
			self.store_agent_copies_flag = False
			new_row["agent_copies"] = self.player_pool.copies()
		else:
			new_row["agent_copies"] = None

		# add incomplete row to make use of functions below
		self.move_tracking_table.append(pd.Series(data=new_row, name=self.move_idx)) # https://stackoverflow.com/questions/39998262/append-an-empty-row-in-dataframe-using-pandas

		# Generate samples with corresponding conditionals
		current_row = self.move_tracking_table.loc[self.move_idx]
		copies = new_row["agent_copies"]
		hand_sampler(current_row, NUM_SAMPLES, copies)

		# generate average conditional
		sample_conditionals = [sample.conditional_distrib for sample in current_row["generated samples"]]
		new_conditional = np.mean(np.array(sample_conditionals), axis=0)
		new_conditional = makeprob(new_conditional)
		current_row["conditional distribution"] = new_conditional
		
		# get prior using the average conditoinal and previous prior if available
		if self.move_idx == 0:
			current_row["prior distribution"] = new_conditional
		else:
			prev_row = self.move_tracking_table.loc[self.move_idx - 1]
			prior = prev_row["prior distribution"]
			updated_prior = new_conditional*prior
			current_row["prior distribution"] = make_prob(updated_prior)

		self.move_idx += 1


	def action_to_key(action):
		return  (action.type, action.cnr, action.col, action.num)

	def sample_hash(sample):
		temp = sample.hand.flatten()
		return str(temp)

	def makeprob(L):
		x = np.flatten(L)
		p = x/np.sum(x)

		if np.sum(p) != 1:
			p[-1] = 1 - np.sum(p[:-1])

		return p

	def new_sample(new_knowledge):
		new_samp = []

		for card in new_knowledge:
			new_samp.append(random.choice(CardChoices, p=makeprob(card))) # https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice

		return new_samp

	def values_to_probs(values):
		nonnegative_vals = values + min(values)
		return nonnegative_vals/np.sum(nonnegative_vals)

	def agent_probs(hand, row_ref, agent_copies): ## THIS ASSUMES THAT WE ONLY HAVE ONE TEAMMATE
		game_state, base_player_model = row_ref["observable game state"]
		move_idx = row_ref["move"]

		for i in range(len(game_state.hands)):
			if game_state.hands[i] is None:
				game_state.hands[i] = hand

		probs = []
		updated_agents = []

		for agent in agent_copies:
			temp_agent = agent.copy()
			values = temp_agent.get_action(game_state, base_player_model)
			probs.append(values_to_probs(values)[move_idx])
			updated_agents.append(temp_agent)

		return np.array(probs), updated_agents

	def hand_sampler(row_ref, number_needed, agent_copies):
		existing_samples = row_ref["generated samples"]
		new_samples = []
		stored_samples = dict()
		new_knowledge = row_ref["hand knowledge"]

		for sample in existing_samples:
			if sample.consistent(new_knowledge):
				new_samples.append(sample.copy())
				stored_samples[sample_hash(sample)] = sample.conditional_distrib
		
		n = len(new_samples)
		updated_copies = None

		for i in range(number_needed - n):
			# generate unique sample
			new_samp = new_sample(new_knowledge)
			h = sample_hash(new_samp)

			# compute conditional game state
			if h in stored_samples:
				new_conditional = stored_samples[h]
			else:
				new_conditional, updated_copies = agent_probs(new_samp.hand, row_ref, agent_copies)
				stored_samples[h] = new_conditional

			new_samp.conditional_distrib = new_conditional
			new_samples.append(new_samp)


		row_ref["generated samples"] = new_samples

		if updated_copies is None:
			_, updated_copies = agent_probs(new_samples[0].hand, row_ref, agent_copies)

		return updated_copies


	def rollforward_move_tracker(self, move_index, most_recent_move, cards_affected_with_new_knowledge):
		agent_copies = self.move_tracking_table.loc[move_index]["agent state copies"]

		for table_idx in range(move_index, most_recent_move + 1):
			current_row = self.move_tracking_table.loc[table_idx] # This is passed around everywhere as reference (including into functions)
			
			# update hand knowledge
			for card, new_k in cards_affected_with_new_knowledge:
				pos_idx = current_row["card ids"][card]
				current_row["hand knowledge"][pos_idx] = new_k


			# use hand_sampler to get new samples where needed
			### num_possibilities = np.prod(np.sum(current_row["hand knowledge"], axis=(1,2)))
			agent_copies = hand_sampler(current_row, NUM_SAMPLES)


			# compute new average conditional distribution
			sample_conditionals = [sample.conditional_distrib for sample in current_row["generated samples"]]
			new_conditional = np.mean(np.array(sample_conditionals), axis=0)
			current_row["conditional distribution"] = new_conditional


			# get previous prior distribution
			prev_idx = table_idx - 1

			if prev_idx >= 0:
				prev_prior = self.move_tracking_table.loc[prev_idx]["prior distribution"]
			else:
				pool_size = self.player_pool.get_size()
				prev_prior = np.ones(pool_size)/pool_size


			# update current prior distribution using updated conditional and previous prior
			new_prior_pre = new_conditional * prev_prior
			new_prior = new_prior_pre/np.sum(new_prior_pre)
			current_row["prior distribution"] = new_prior
