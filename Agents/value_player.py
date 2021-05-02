from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import *
import time
import copy
import random


def count_card_list(knowledge, ls):
    for card in ls:
        remove_card(card, knowledge)


def count_board(knowledge, board):
    for i in range(len(board)):
        for j in range(1, board[i][1]):
            remove_card((i, j), knowledge)


def remove_card(card, knowledge):
    for slot in knowledge:
        slot[card[0]][card[1] - 1] = max(0, slot[card[0]][card[1] - 1] - 1)


def weight_knowledge(knowledge, weights):
    new_knowledge = copy.deepcopy(weights)
    for slot in range(len(new_knowledge)):
        for col in range(5):
            for num in range(5):
                new_knowledge[slot][col][num] *= knowledge[slot][col][num]
    return new_knowledge


def update_weights(weights, weight, board, hint_indices):
    if not hint_indices:
        return
    priority = hint_indices[-1]
    for col in range(5):
        for nr in range(5):
            if card_playable((col, nr + 1), board):
                weights[priority][col][nr] *= weight
    return weights


# all info (knowledge, weights, partner info, etc.) is updated every time we are informed of an action
# weights/partner weights are maintained; everything else is just replaced
# this keeps get_action light/fast, but informing takes more time

class ValuePlayer(Player):
    def __init__(self, name, pnr, **kwargs):
        super().__init__(name, pnr)
        self.partner_nr = 1 - self.pnr  # hard code for two players
        self.turn = 0

        # self knowledge
        self.knowledge = []
        self.weights = [
            [[1 for _ in range(5)] for _ in range(5)] for _ in range(5)
        ]
        self.weighted_knowledge = None

        # partner knowledge
        self.partner_hand = None
        self.partner_knowledge = []
        self.partner_weights = [
            [[1 for _ in range(5)] for _ in range(5)] for _ in range(5)
        ]
        self.partner_weighted_knowledge = None

        # state/model information state; maintained by simply copying whenever we receive new ones via inform
        self.state = None
        self.model = None

        # parameters and default values
        self.hint_weight = 1000.0
        self.discard_type = "first"
        self.card_count = True
        self.card_count_partner = True
        self.get_action_values = False
        self.default_hint = "high"
        self.play_threshold = 0.95
        self.discard_threshold = 0.95
        self.play_bias = 1.0
        self.disc_bias = 0.9
        self.hint_bias = 0.8
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _update_info(self, state, model):
        self.state = state
        self.model = model
        self.knowledge = copy.deepcopy(model.get_knowledge())
        self.partner_hand = state.get_hands()[self.partner_nr]
        self.partner_knowledge = state.get_all_knowledge()[self.partner_nr]

    def _count_cards(self):
        if self.card_count:
            count_card_list(self.knowledge, self.state.get_trash())
            # count_card_list(self.knowledge, self.state.get_hands()[self.partner_nr])
            count_board(self.knowledge, self.state.get_board())

    def _count_partner_cards(self, partner_knowledge):
        if self.card_count_partner:
            count_card_list(partner_knowledge, self.state.get_trash())
            count_board(partner_knowledge, self.state.get_board())

    def _eval_play(self, action):
        assert(action.type == PLAY)
        pct = slot_playable_pct(
                    self.weighted_knowledge[action.cnr], self.state.get_board()
                )
        print(action)
        print(pct)
        if pct > self.play_threshold:
            return pct
        elif pct > 0.5:
            return 0
        else:
            return -1

    def _eval_discard(self, action):
        assert(action.type == DISCARD)
        pct = slot_discardable_pct(
                    self.weighted_knowledge[action.cnr], self.state.get_board()
                )
        if pct > self.discard_threshold:
            return pct
        elif pct > 0.5:
            return 0
        else:
            return -1

    def _eval_hint(self, action):
        assert(action.type in [HINT_COLOR, HINT_NUMBER])
        target = get_multi_target(action, self.partner_hand, self.partner_weighted_knowledge,
                                  self.state.get_board(), self.play_threshold, self.discard_threshold)
        if target == -1:
            return 0
        if target_possible(action, target, self.partner_weighted_knowledge, self.state.get_board()):
            if card_playable(self.partner_hand[target], self.state.get_board()):
                return 0.8
            else:
                return -1
        return 0

    def eval_action(self, action):
        if action.type == PLAY:
            return self.play_bias * self._eval_play(action)
        elif action.type == DISCARD:
            return self.disc_bias * self._eval_discard(action)
        return self.hint_bias * self._eval_hint(action)

    def get_action(self, game_state, player_model):
        self.turn += 1
        # because of valid_action's implementation in hanabi.py we need to update this here as well
        # to get the correct legal moves
        self._update_info(game_state, player_model)

        # count cards
        self._count_cards()

        # compute weighted knowledge, weighted partner knowledge
        self.weighted_knowledge = weight_knowledge(self.knowledge, self.weights)
        self.partner_weighted_knowledge = weight_knowledge(self.partner_knowledge, self.partner_weights)

        # evaluate all moves and return maximum
        best_action = None
        max_value = -1
        for action in self.state.get_valid_actions():
            value = self.eval_action(action)
            if value > max_value:
                best_action = action
                max_value = value
        return best_action

    # hint_indices is [] if the action is not a hint
    def inform(self, action, player, new_state, new_model):
        self._update_info(new_state, new_model)
        if player == self.pnr:
            if action.type in [PLAY, DISCARD]:
                # reset weights for specific slot
                del self.weights[action.cnr]
                if len(self.knowledge) != len(self.weights):
                    self.weights.append([
                        [1 for _ in range(5)] for _ in range(5)
                    ])
            else:
                # on hint, update partner weights accordingly
                update_weights(
                    self.partner_weights, self.hint_weight, new_state.get_board(), new_state.get_hinted_indices())
            return

        # for 2 player games there's only 1 other player
        assert player == self.partner_nr
        if action.type in [HINT_COLOR, HINT_NUMBER]:
            update_weights(self.weights, self.hint_weight, new_state.get_board(), new_state.get_hinted_indices())

        elif action.type in [PLAY, DISCARD]:
            # reset weights for specific slot
            del self.partner_weights[action.cnr]
            if len(self.partner_weights) != len(
                    new_state.get_all_knowledge()[self.partner_nr]
            ):
                self.partner_weights.append([
                    [1 for _ in range(5)] for _ in range(5)
                ])