from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action
from Agents.ChiefAgent.player_pool import PlayerPool


class ChiefPlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.pnr = pnr

    def get_action(self, game_state, player_model):
        pass

    def inform(self, action, player, game_state, player_model):
        pass
