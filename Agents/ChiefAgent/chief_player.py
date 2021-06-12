from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from Agents.ChiefAgent.player_pool import PlayerPool
import pandas as pd
import numpy as np

STARTING_COLUMNS_MOVETRACKING = {"move": [], "observable game state":[], "hand knowledge":[], "prior distribution":[], "conditional distribution":[], "generated samples":[], "agent state copies"[]}
NUM_SAMPLES = 10

class Sample(object):
	def __init__(self, hand, conditional_distrib, first_usage):
		self.hand = hand
		self.conditional_distrib = conditional_distrib
		self.first_usage = first_usage


class ChiefPlayer(Player):cards_affected_with_new_knowledge
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

	def hand_sampler(self, existing_samples, number_needed, hand_knowledge):
		pass

	def rollforward_move_tracker(self, move_index, most_recent_move, cards_affected_with_new_knowledge):
		for table_idx in range(move_index, most_recent_move + 1):
			
			# update hand knowledge
			


			# use hand_sampler to get new samples where needed
			self.hand_sampler(NUM_SAMPLES, new_knowledge) # If NUM_SAMPLES > number possible, just cap it


			# compute new average conditional distribution
			sample_conditionals = [sample.conditional_distrib for sample in self.move_tracking_table.loc[table_idx]["generated samples"]]
			new_conditional = np.mean(np.array(sample_conditionals), axis=0)
			self.move_tracking_table.loc[table_idx]["conditional distribution"] = new_conditional


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
			self.move_tracking_table.loc[table_idx]["prior distribution"] = new_prior
