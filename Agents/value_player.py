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


def update_weights(weights, weight, board, target):
    if target != -1:
        for col in range(5):
            for nr in range(5):
                if card_playable((col, nr + 1), board):
                    weights[target][col][nr] *= weight
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
        # discard_type and default_hint don't do anything at the moment; not sure how to implement them
        # with this approach except for multiplying by a bonus weight (which feels like cheating)
        self.discard_type = "first"
        self.default_hint = "high"

        self.card_count = True
        self.card_count_partner = True
        self.get_action_values = False
        self.play_threshold = 0.95
        self.discard_threshold = 0.5
        self.play_bias = 1.0
        self.disc_bias = 0.7
        self.hint_bias = 0.9
        self.hint_biases = [0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.0, 1.0]
        self.play_biases = [0, 0.7, 0.9, 1.0]
        # TODO: make a function for discard risk
        self.hint_risk_weight = 1.0
        self.play_risk_weight = 1.0
        self.dynamic_bias = True
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _update_info(self, state, model):
        self.state = state
        self.model = model
        self.knowledge = copy.deepcopy(model.get_knowledge())
        self.partner_hand = state.get_hands()[self.partner_nr]
        self.partner_knowledge = state.get_all_knowledge()[self.partner_nr]

    def _update_bias(self):
        if self.dynamic_bias:
            self.hint_risk_weight = self.hint_biases[self.state.get_num_hints()]
            self.play_risk_weight = self.play_biases[self.state.get_hits()]

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
        if pct > self.play_threshold:
            return pct
        else:
            # playing low-confidence is *really bad*, so we need to multiply by something small
            return pct * 0.1

    def _eval_discard(self, action):
        assert(action.type == DISCARD)
        pct = slot_discardable_pct(
                    self.weighted_knowledge[action.cnr], self.state.get_board(), self.state.get_trash()
                )
        if pct > self.discard_threshold:
            return pct
        else:
            # discarding low-confidence isn't as bad as low-confidence play, so this should be higher than
            # whatever value we use in low confidence play
            return pct * 0.5

    def _eval_hint(self, action):
        assert(action.type in [HINT_COLOR, HINT_NUMBER])
        target = get_multi_target(action, self.partner_hand, self.partner_weighted_knowledge,
                                  self.state.get_board(), self.play_threshold, self.discard_threshold)
        if target == -1:
            return 0.25
        if target_possible(action, target, self.partner_weighted_knowledge, self.state.get_board()):
            if card_playable(self.partner_hand[target], self.state.get_board()):
                # TODO: differentiate between valid hints
                return 0.8
            else:
                return 0.1
        return 0.25

    def eval_action(self, action):
        if action.type == PLAY:
            return self.play_bias * self.play_risk_weight * self._eval_play(action)
        elif action.type == DISCARD:
            return self.disc_bias * self._eval_discard(action)
        return self.hint_bias * self.hint_risk_weight * self._eval_hint(action)

    def get_action(self, game_state, player_model):
        self.turn += 1
        # because of valid_action's implementation we need to update this here as well to get the correct legal moves
        self._update_info(game_state, player_model)
        self._update_bias()

        # count cards
        self._count_cards()

        # compute weighted knowledge, weighted partner knowledge
        self.weighted_knowledge = weight_knowledge(self.knowledge, self.weights)
        self.partner_weighted_knowledge = weight_knowledge(self.partner_knowledge, self.partner_weights)

        value_dict = {}
        # evaluate all moves and return maximum
        best_action = None
        # all values are in [0, 1], so this is lower than all possible values
        max_value = -1.0
        for action in self.state.get_valid_actions():
            # print(action)
            value = self.eval_action(action)
            # print(value)
            if value > max_value:
                best_action = action
                max_value = value
            if self.get_action_values:
                value_dict[action] = value
        if self.get_action_values:
            return value_dict
        return best_action

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
                target = -1
                hint_indices = copy.deepcopy(new_state.get_hinted_indices())
                while hint_indices:
                    potential_target = hint_indices[-1]
                    if slot_playable_pct(self.partner_weighted_knowledge[potential_target], new_state.get_board()) \
                            <= self.play_threshold:
                        target = potential_target
                        break
                    del hint_indices[-1]
                update_weights(
                    self.partner_weights, self.hint_weight, new_state.get_board(), target)
            return

        # for 2 player games there's only 1 other player
        assert player == self.partner_nr
        if action.type in [HINT_COLOR, HINT_NUMBER]:
            target = -1
            self.weighted_knowledge = weight_knowledge(self.knowledge, self.weights)
            hint_indices = copy.deepcopy(new_state.get_hinted_indices())
            while hint_indices:
                potential_target = hint_indices[-1]
                if slot_playable_pct(self.weighted_knowledge[potential_target], new_state.get_board()) <= self.play_threshold:
                    target = potential_target
                    break
                del hint_indices[-1]
            update_weights(self.weights, self.hint_weight, new_state.get_board(), target)

        elif action.type in [PLAY, DISCARD]:
            # reset weights for specific slot
            del self.partner_weights[action.cnr]
            if len(self.partner_weights) != len(
                    new_state.get_all_knowledge()[self.partner_nr]
            ):
                self.partner_weights.append([
                    [1 for _ in range(5)] for _ in range(5)
                ])