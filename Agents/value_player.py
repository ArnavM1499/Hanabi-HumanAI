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


# This is actually bugged at the moment -- it can't handle when one player has
# less than 5 cards (close to the end of the game). it doesn't crash, but
# things like self.todo won't behave properly


class ValuePlayer(Player):
    def __init__(self, name, pnr, **kwargs):
        super().__init__(name, pnr)
        self.partner_nr = 1 - self.pnr  # hard code for two players
        self.turn = 0
        # same as last_state.get_knowledge(), but done for coding ease for now
        # if we need to optimize speed/memory we can remove it
        self.knowledge = []
        self.hint_weights = [
            [[1 for _ in range(5)] for _ in range(5)] for _ in range(5)
        ]
        self.weighted_knowledge = None
        self.partner_hand = None
        self.partner_knowledge = []
        self.partner_hint_weights = [
            [[1 for _ in range(5)] for _ in range(5)] for _ in range(5)
        ]
        self.partner_weighted_knowledge = None
        self.last_state = None
        self.last_model = None
        self.protect = []
        self.hinted = []
        self.log = []
        self.nr_cards = 5
        # below are default values for parameters
        self.hint_weight = 1000.0
        self.discard_type = "first"
        self.card_count = True
        self.card_count_partner = True
        self.get_action_values = False
        self.default_hint = "high"
        self.play_threshold = 0.95
        self.discard_threshold = 0.95
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _count_cards(self):
        if self.card_count:
            count_card_list(self.knowledge, self.last_state.get_trash())
            count_card_list(self.knowledge, self.last_state.get_hands()[self.partner_nr])
            count_board(self.knowledge, self.last_state.get_board())

    def _count_partner_cards(self, partner_knowledge):
        if self.card_count_partner:
            count_card_list(partner_knowledge, self.last_state.get_trash())
            count_board(partner_knowledge, self.last_state.get_board())

    def _eval_play(self, action):
        assert(action.type == PLAY)
        pct = slot_playable_pct(
                    self.weighted_knowledge[action.cnr], self.last_state.get_board()
                )
        if pct > self.play_threshold:
            return pct
        elif pct > 0.5:
            return 0
        else:
            return -1

    def _eval_discard(self, action):
        assert(action.type == DISCARD)
        pct = slot_discardable_pct(
                    self.weighted_knowledge[action.cnr], self.last_state.get_board()
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
                                  self.last_state.get_board(), self.play_threshold, self.discard_threshold)
        if target == -1:
            return 0
        if target_possible(action, target, self.partner_weighted_knowledge, self.last_state.get_board()):
            if card_playable(self.partner_hand[target], self.last_state.get_board()):
                return 1
            else:
                return -1
        return 0

    def eval_action(self, action):
        if action.type == PLAY:
            return self._eval_play(action)
        elif action.type == DISCARD:
            return self._eval_discard(action)
        return self._eval_hint(action)

    def get_action(self, game_state, player_model):
        self.turn += 1

        # first turn
        if self.last_model is None:
            self.last_model = player_model
        if self.last_state is None:
            self.last_state = game_state
        self.knowledge = copy.deepcopy(self.last_model.get_knowledge())
        #print("player " + str(self.pnr) + " knowledge: " + str(self.knowledge))
        if self.card_count:
            self._count_cards()
        #print("player " + str(self.pnr) + " knowledge: " + str(self.knowledge))
        #print("partner hand:" + str(self.last_state.get_hands()[self.partner_nr]))
        #time.sleep(3)
        self.weighted_knowledge = weight_knowledge(self.knowledge, self.hint_weights)
        best_action = None
        max_value = -1
        for action in self.last_state.get_valid_actions():
            value = self.eval_action(action)
            if value > max_value:
                best_action = action
                max_value = value
        return best_action

    # for 2 player the only hints we need to consider are hints about our cards
    # this will need to be revisited if we generalize to more players
    def _receive_hint(self, action, player, new_state, new_model, hint_indices):
        self.last_model = new_model
        self.last_state = new_state
        # assert action.type in [HINT_COLOR, HINT_NUMBER] and player == self.partner_nr
        # assert(not self.partner_todo)

        new_board = new_state.get_board()
        self.knowledge = copy.deepcopy(new_model.get_knowledge())

        # empty list: no new info gained; bad hint
        if not hint_indices:
            return

        priority_index = hint_indices[-1]

        # is there a more efficient way of doing this?
        for slot in range(len(self.knowledge)):
            for col in range(5):
                for num in range(5):
                    if slot == priority_index and card_playable(
                        (col, num + 1), new_board
                    ):
                        self.hint_weights[slot][col][num] *= self.hint_weight

    # don't need the hint indices, of course
    def _receive_play(self, action, player, new_state, new_model):
        self.last_model = new_model
        self.last_state = new_state
        if len(self.partner_hint_weights) == len(
            new_state.get_all_knowledge()[self.partner_nr]
        ):
            self.partner_hint_weights[action.cnr] = [
                [1 for _ in range(5)] for _ in range(5)
            ]
        else:
            del self.partner_hint_weights[action.cnr]

    def _receive_discard(self, action, player, new_state, new_model):
        self.last_model = new_model
        self.last_state = new_state
        if len(self.partner_hint_weights) == len(
            new_state.get_all_knowledge()[self.partner_nr]
        ):
            self.partner_hint_weights[action.cnr] = [
                [1 for _ in range(5)] for _ in range(5)
            ]
        else:
            del self.partner_hint_weights[action.cnr]

    # hint_indices is [] if the action is not a hint
    def inform(self, action, player, new_state, new_model):
        hint_indices = new_state.get_hinted_indices()
        if player == self.pnr:
            # maybe this part should be moved up to before playing?
            # reset knowledge if we played or discarded
            if action.type == PLAY or action.type == DISCARD:
                self.knowledge = copy.deepcopy(new_model.get_knowledge())
                if len(self.knowledge) == len(self.hint_weights):
                    self.hint_weights[action.cnr] = [
                        [1 for _ in range(5)] for _ in range(5)
                    ]
                else:
                    del self.hint_weights[action.cnr]
            elif action.type in [HINT_COLOR, HINT_NUMBER]:
                new_board = new_state.get_board()
                partner_knowledge = copy.deepcopy(
                    new_state.get_all_knowledge()[self.partner_nr]
                )

                # empty list: no new info gained; bad hint
                if not hint_indices:
                    return

                self._count_partner_cards(partner_knowledge)
                priority_index = hint_indices[-1]

                # is there a more efficient way of doing this?
                for slot in range(len(partner_knowledge)):
                    for col in range(5):
                        for num in range(5):
                            if slot == priority_index and card_playable(
                                (col, num + 1), new_board
                            ):
                                self.partner_hint_weights[slot][col][
                                    num
                                ] *= self.hint_weight
            return

        # for 2 player games there's only 1 other player
        # assert player == self.partner_nr
        if action.type in [HINT_COLOR, HINT_NUMBER]:
            self._receive_hint(action, player, new_state, new_model, hint_indices)

        elif action.type == PLAY:
            self._receive_play(action, player, new_state, new_model)

        # discard
        else:
            self._receive_discard(action, player, new_state, new_model)