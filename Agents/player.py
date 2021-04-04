from common_game_functions import *
from hanabi import *


class Action(object):
    def __init__(self, type, pnr=None, col=None, num=None, cnr=None):
        self.type = type
        self.pnr = pnr
        self.col = col
        self.num = num
        self.cnr = cnr

    def __str__(self):
        if self.type == HINT_COLOR:
            return (
                "hints "
                + str(self.pnr)
                + " about all their "
                + COLORNAMES[self.col]
                + " cards"
            )
        if self.type == HINT_NUMBER:
            return (
                "hints "
                + str(self.pnr)
                + " about all their "
                + str(self.num)
                + " cards"
            )
        if self.type == PLAY:
            return "plays their " + str(self.cnr)
        if self.type == DISCARD:
            return "discards their " + str(self.cnr)

    def __eq__(self, other):
        return (self.type, self.pnr, self.col, self.num, self.cnr) == (
            other.type,
            other.pnr,
            other.col,
            other.num,
            other.cnr,
        )

    def __hash__(self):
        return hash(str(self))


class Player(object):
    def __init__(self, name, pnr):
        self.name = name
        self.pnr = pnr
        self.explanation = []

    def get_nr(self):
        return self.pnr

    def get_action(self, game_state, base_player_model):
        return random.choice(valid_actions)

    def inform(self, action, player, new_state, new_model):
        pass

    def get_explanation(self):
        return self.explanation

    def new_game(self, hands):  # All the hands visible to the player
        pass
