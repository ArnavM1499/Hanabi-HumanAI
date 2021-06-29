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


def slot_pct(knowledge, list):
    total_combos = 0.0
    satisf_combos = 0.0
    for col in range(len(knowledge)):
        # there are 5 possible numbers
        for num in range(5):
            total_combos += knowledge[col][num]
            if (col, num + 1) in list:
                satisf_combos += knowledge[col][num]
    if total_combos < 1:
        print("slot_pct error")
    return satisf_combos / total_combos


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
        self.last_hint = None

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
        self.protect = []
        self.discard = []
        self.play = []

        # whether we return a dictionary of all actions/values, or just the best action
        self.get_action_values = False

        # parameters and default values
        self.hint_weight = 1000.0

        # left, right, best
        self.play_preference = "best"
        self.discard_preference = "best"

        # doesn't actually do anything at the moment; still trying to parameterize
        self.default_hint = "high"

        # card counting
        self.card_count = True
        self.card_count_partner = True

        # we want to assign a nonzero value to an arbitrary (average) discard:
        self.discard_base_value = 0.2

        # how much we care about protecting our high cards for future play
        self.protect_importance = 0.4

        # multiplier for low confidence (i.e. below threshold) plays/discards
        # perhaps it would be more appropriate to replace with a continuous map e.g. x^2
        self.play_low_multiplier = 0.1
        self.discard_low_multiplier = 0.5
        self.play_threshold = 0.95
        self.discard_threshold = 0.55

        # how much we like playing, discarding, hinting in general
        self.play_bias = 1.0
        self.disc_bias = 0.8
        self.hint_bias = 0.9

        # if dynamic bias is true, then hint and play biases are further multiplied
        # by the following values as the game goes on
        self.dynamic_bias = True

        # [0 hints left, 1 hint, 2 hint, etc.]
        self.hint_biases = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

        # [0 lives left, 1 life, etc.]
        self.play_biases = [0, 0.7, 0.9, 1.0]
        # TODO: make a function for discard risk

        # default when dynamic bias is off
        self.hint_risk_weight = 1.0
        self.play_risk_weight = 1.0
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _update_protect_discard(self):
        self.protect = []
        self.discard = []
        self.play = []
        trash = self.state.get_trash()
        # print(trash)
        # if trash:
        #     print(trash[0])
        #     print(trash.count(trash[0]))
        for col in range(5):
            nr = self.state.get_board()[col][1]
            if nr < 5:
                self.play.append((col, nr + 1))
            for i in range(1, nr + 1):
                self.discard.append((col, i))
            trash_mode = False
            for j in range(nr + 1, 6):
                if trash_mode:
                    self.discard.append((col, j))
                elif trash.count((col, j)) == COUNTS[j - 1] - 1:
                    self.protect.append((col, j))
                elif trash.count((col, j)) == COUNTS[j - 1]:
                    trash_mode = True
        # print(self.play)
        # print(self.protect)
        # print(self.discard)

    def _update_bias(self):
        if self.dynamic_bias:
            self.hint_risk_weight = self.hint_biases[self.state.get_num_hints()]
            self.play_risk_weight = self.play_biases[self.state.get_hits()]

    def _update_info(self, state, model):
        self.state = state
        self.model = model
        self.knowledge = copy.deepcopy(model.get_knowledge())
        self.partner_hand = state.get_hands()[self.partner_nr]
        self.partner_knowledge = state.get_all_knowledge()[self.partner_nr]
        self._update_protect_discard()
        self._update_bias()

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
        pct = slot_pct(self.weighted_knowledge[action.cnr], self.play)
        # pct = slot_playable_pct(
        #             self.weighted_knowledge[action.cnr], self.state.get_board()
        #         )
        if self.play_preference == "left":
            pct *= ([1.2, 1.1, 1.0, 0.9, 0.8][action.cnr])
        elif self.play_preference == "right":
            pct *= ([0.8, 0.9, 1.0, 1.1, 1.2][action.cnr])
        if pct > self.play_threshold:
            return pct
        else:
            return pct * self.play_low_multiplier

    def _eval_discard(self, action):
        assert(action.type == DISCARD)
        value = self.discard_base_value + slot_pct(self.weighted_knowledge[action.cnr], self.discard)
        value -= self.protect_importance * slot_pct(self.weighted_knowledge[action.cnr], self.protect)
        # no negatives
        value = max(value, 0)
        # pct = slot_discardable_pct(
        #     self.weighted_knowledge[action.cnr], self.state.get_board(), self.state.get_trash()
        # )
        if self.discard_preference == "left":
            value *= ([1.2, 1.1, 1.0, 0.9, 0.8][action.cnr])
        elif self.discard_preference == "right":
            value *= ([0.8, 0.9, 1.0, 1.1, 1.2][action.cnr])
        if value > self.discard_threshold:
            return value
        else:
            return value * self.discard_low_multiplier

    def _eval_hint(self, action):
        # if self.last_hint is not None and action == self.last_hint:
        #     return 0
        # assert(action.type in [HINT_COLOR, HINT_NUMBER])

        target = get_multi_target(action, self.partner_hand, self.partner_weighted_knowledge,
                                  self.state.get_board(), self.play_threshold, self.discard_threshold)
        # copy_weights = copy.deepcopy(self.partner_weights)
        # new_partner_weights = update_weights(copy_weights, self.hint_weight, self.state.get_board(), target)
        # copy_knowledge = copy.deepcopy(self.partner_knowledge)

        # # update knowledge ourselves as part of simulation
        # if action.type == HINT_COLOR:
        #     for i in range(len(copy_knowledge)):
        #         if self.partner_hand[i][0] == action.col:
        #             for j in range(0, 5):
        #                 if j != action.col:
        #                     copy_knowledge[i][j] = [0, 0, 0, 0, 0]
        #         else:
        #             copy_knowledge[i][action.col] = [0, 0, 0, 0, 0]
        # elif action.type == HINT_NUMBER:
        #     for i in range(len(copy_knowledge)):
        #         if self.partner_hand[i][1] == action.num:
        #             for j in range(0, 5):
        #                 if j != action.cnr:
        #                     copy_knowledge[i][j][action.num - 1] = 0
        #         else:
        #             for j in range(0, 5):
        #                 copy_knowledge[i][j][action.num - 1] = 0

        # new_weighted_knowledge = weight_knowledge(copy_knowledge, new_partner_weights)

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

        # print(self.name)
        # for hint in self.model.get_hints():
        #     print(hint[1])

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
        # print(self.name)
        # print(self.hint_risk_weight)
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
        self._count_cards()
        if player == self.pnr:
            if action.type in [PLAY, DISCARD]:
                # reset weights for specific slot
                del self.weights[action.cnr]
                if len(self.knowledge) != len(self.weights):
                    self.weights.append([
                        [1 for _ in range(5)] for _ in range(5)
                    ])
            else:
                self.last_hint = action
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
            self.weighted_knowledge = weight_knowledge(self.knowledge, self.weights)
            self.partner_weighted_knowledge = weight_knowledge(self.partner_knowledge, self.partner_weights)
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
        self.weighted_knowledge = weight_knowledge(self.knowledge, self.weights)
        self.partner_weighted_knowledge = weight_knowledge(self.partner_knowledge, self.partner_weights)