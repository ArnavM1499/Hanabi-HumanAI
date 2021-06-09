from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from Agents.ChiefAgent.player_pool import PlayerPool
import pandas as pd
import numpy as np

STARTING_COLUMNS_MOVETRACKING = {"observable game state":[], "hand knowledge":[], "prior distribution":[], "conditional distribution":[], "generated samples":[], "agent state copies"[]}

class ChiefPlayer(Player):
	def __init__(self, name, pnr):
		self.name = name
		self.pnr = pnr
		self.move_tracking_table = pd.DataFrame(data=STARTING_COLUMNS_MOVETRACKING)

	def get_action(self, game_state, player_model):
		pass

	def inform(self, action, player, game_state, player_model):
		pass

	def rollforward_move_tracker(self, move_index, cards_affected_with_new_knowledge)