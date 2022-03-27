import random
import pickle
from common_game_functions import *
import Agents


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
        return isinstance(other, Action) and (self.encode() == other.encode())

    def __hash__(self):
        return hash(str(self))

    def encode(self):
        if self.type == HINT_COLOR:
            t = (HINT_COLOR, self.col)
        elif self.type == HINT_NUMBER:
            t = (HINT_NUMBER, self.num - 1)
        elif self.type == PLAY:
            t = (PLAY, self.cnr)
        elif self.type == DISCARD:
            t = (DISCARD, self.cnr)
        else:
            raise NotImplementedError
        return t[0] * 5 + t[1]

    @staticmethod
    def from_encoded(encoded, pnr):
        action_type = encoded // 5
        action_idx = encoded % 5
        if action_type == HINT_COLOR:
            return Action(HINT_COLOR, col=action_idx, pnr=1 - pnr)
        elif action_type == HINT_NUMBER:
            return Action(HINT_NUMBER, num=1 + action_idx, pnr=1 - pnr)
        elif action_type == PLAY:
            return Action(PLAY, cnr=action_idx, pnr=pnr)
        elif action_type == DISCARD:
            return Action(DISCARD, cnr=action_idx, pnr=pnr)
        else:
            raise NotImplementedError


class Player(object):
    def __init__(self, name, pnr):
        self.name = name
        self.pnr = pnr
        self.explanation = []

    @staticmethod
    def from_dict(name, pnr, json_dict):
        json_dict["name"] = name
        json_dict["pnr"] = pnr
        return getattr(Agents, json_dict["player_class"])(**json_dict)

    def get_nr(self):
        return self.pnr

    def get_action(self, game_state, base_player_model):
        return random.choice(game_state.get_valid_actions())

    def inform(self, action, player, new_state, new_model):
        pass

    def get_explanation(self):
        return self.explanation

    def new_game(self, hands):  # All the hands visible to the player
        pass

    def snapshot(self, data_file=None):
        if data_file:
            pickle.dump(self.__dict__, open(data_file, "wb"))
