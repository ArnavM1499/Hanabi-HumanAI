from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from Agents.ChiefAgent.player_pool import PlayerPool
import pandas as pd
import numpy as np
import random

STARTING_COLUMNS_MOVETRACKING = {"move": [], "observable game state":[], "card ids":[], "hand knowledge":[], "prior distribution":[], "conditional distribution":[], "generated samples":[], "agent state copies"[]}
NUM_SAMPLES = 10

CardChoices = []

for i in range(5):
	for j in range(5):
		CardChoices.append((i,j+1))

# Note: hand is formmated as [(color, number)] - can use indices for color and indices + 1 for number

class Sample(object):
	def __init__(self, hand, conditional_distrib, first_usage):
		self.hand = hand
		self.conditional_distrib = conditional_distrib
		self.first_usage = first_usage

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
		self.move_tracking_table.set_index("move")
		self.player_pool = PlayerPool("agent_pool.json")

	def get_action(self, game_state, player_model):
		pass

	def inform(self, action, player, game_state, player_model):
		pass

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

		for i in range(len(game_state.hands)):
			if game_state.hands[i] is None:
				game_state.hands[i] = hand

		probs = []
		updated_agents = []

		for agent in agent_copies:
			temp_agent = agent.copy()
			values = temp_agent.get_action(game_state, base_player_model)
			probs.append(values_to_probs(values))
			updated_agents.append(temp_agent)

		return np.array(probs), updated_agents

	def hand_sampler(row_ref, number_needed, agent_copies):
		existing_samples = row_ref["generated samples"]
		new_samples = []
		uniqueness_hash = set()
		new_knowledge = row_ref["hand knowledge"]

		for sample in existing_samples:
			if sample.consistent(new_knowledge):
				new_samples.append(sample.copy())
				uniqueness_hash.add(sample_hash(sample))
		
		n = len(new_samples)
		updated_copies = None

		for i in range(number_needed - n):
			# generate unique sample
			new_samp = new_sample(new_knowledge)
			h = sample_hash(new_samp)

			while(h in uniqueness_hash):
				new_samp = new_sample(new_knowledge)
				h = sample_hash(new_samp)

			uniqueness_hash.add(h)

			# compute conditional game state
			new_conditional, updated_copies = agent_probs(new_samp.hand, row_ref, agent_copies)
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
			num_possibilities = np.prod(np.sum(current_row["hand knowledge"], axis=(1,2)))
			agent_copies = hand_sampler(current_row, min(NUM_SAMPLES, num_possibilities)) # If NUM_SAMPLES > number possible, just cap it


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
