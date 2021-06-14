from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from Agents.ChiefAgent.player_pool import PlayerPool
import pandas as pd
import numpy as np

STARTING_COLUMNS_MOVETRACKING = {"move": [], "observable game state":[], "card ids":[], "hand knowledge":[], "prior distribution":[], "conditional distribution":[], "generated samples":[], "agent state copies"[]}
NUM_SAMPLES = 10

class Sample(object):
	def __init__(self, hand, conditional_distrib, first_usage):
		self.hand = hand
		self.conditional_distrib = conditional_distrib
		self.first_usage = first_usage


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

	def hand_sampler(self, row_ref, number_needed):
		existing_samples = row_ref["generated samples"]
		new_knowledge = row_ref["hand knowledge"]

		


	def rollforward_move_tracker(self, move_index, most_recent_move, cards_affected_with_new_knowledge):
		for table_idx in range(move_index, most_recent_move + 1):
			current_row = self.move_tracking_table.loc[table_idx] # This is passed around everywhere as reference (including into functions)
			
			# update hand knowledge
			for card, new_k in cards_affected_with_new_knowledge:
				pos_idx = current_row["card ids"][card]
				current_row["hand knowledge"][pos_idx] = new_k


			# use hand_sampler to get new samples where needed
			num_possibilities = np.prod(np.sum(current_row["hand knowledge"], axis=(1,2)))
			self.hand_sampler(current_row, min(NUM_SAMPLES, num_possibilities)) # If NUM_SAMPLES > number possible, just cap it


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
