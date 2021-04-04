from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import *
import time
import copy


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


class ExperimentalPlayer(Player):
    def __init__(self, name, pnr, **kwargs):
        super().__init__(name, pnr)
        self.partner_nr = 1 - self.pnr  # hard code for two players
        self.turn = 0
        self.todo = []
        # same as last_state.get_knowledge(), but done for coding ease for now
        # if we need to optimize speed/memory we can remove it
        self.knowledge = []
        self.hint_weights = [
            [[1 for _ in range(5)] for _ in range(5)] for _ in range(5)
        ]
        self.partner_todo = []  # same format
        self.partner_hint_weights = [
            [[1 for _ in range(5)] for _ in range(5)] for _ in range(5)
        ]
        self.last_state = None
        self.last_model = None
        self.protect = []
        self.hinted = []
        self.log = []
        self.nr_cards = 5
        # should be accepted as a parameter
        self.hint_weight = 1000.0
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _sort_todo(self):
        pass

    def _decide(self):
        return "_execute"

    def _execute(self, force=False):

        # execute something in the todo list

        while self.todo:
            index = self.todo[-1]
            weighted_knowledge = weight_knowledge(self.knowledge, self.hint_weights)
            if (
                slot_playable_pct(
                    weighted_knowledge[index], self.last_state.get_board()
                )
                > 0.95
            ):
                return Action(PLAY, cnr=index)
            elif (
                slot_discardable_pct(
                    weighted_knowledge[index], self.last_state.get_board()
                )
                > 0.95
            ):
                return Action(DISCARD, cnr=index)
            else:
                del self.todo[-1]

        return self._hint(False)

    def _hint(self, force=False):

        if self.last_state.get_num_hints() == 0:
            return self._discard(True)

        partner_hand = self.last_state.get_hands()[self.partner_nr]
        partner_knowledge = copy.deepcopy(self.last_state.get_all_knowledge())[
            self.partner_nr
        ]

        # check for playable card
        playable = []
        for i in range(len(partner_hand)):
            if card_playable(partner_hand[i], self.last_state.get_board()):
                playable.append(i)

        weighted_partner_knowledge = weight_knowledge(
            partner_knowledge, self.partner_hint_weights
        )

        # if card is playable, hint it with the most info gain
        while playable:
            newest_playable = playable[-1]
            if newest_playable in self.partner_todo:
                del playable[-1]
                continue
            if newest_playable >= 0:
                hint_type = best_hint_type(
                    partner_hand, newest_playable, weighted_partner_knowledge
                )
                if hint_type is None:
                    del playable[-1]
                    continue
                elif hint_type == HINT_COLOR:
                    return Action(
                        HINT_COLOR,
                        self.partner_nr,
                        col=partner_hand[newest_playable][0],
                    )
                else:
                    return Action(
                        HINT_NUMBER,
                        self.partner_nr,
                        num=partner_hand[newest_playable][1],
                    )

        # check for discardable card

        discardable = []
        for i in range(len(partner_hand)):
            if card_discardable(partner_hand[i], self.last_state.get_board()):
                discardable.append(i)

        # if card is discardable, hint it with the most info gain
        # make sure it cannot be "confused" with playable
        while discardable:
            newest_discardable = discardable[-1]
            if newest_discardable in self.partner_todo:
                del discardable[-1]
                continue
            if newest_discardable >= 0:
                hint_type = best_hint_type(
                    partner_hand, newest_discardable, weighted_partner_knowledge
                )
                #hint_type = best_discard_hint_type(
                #    partner_hand, newest_discardable, weighted_partner_knowledge, self.last_state.get_board()
                #)
                if hint_type is None:
                    del discardable[-1]
                    continue
                elif hint_type == HINT_COLOR:
                   return Action(
                        HINT_COLOR,
                        self.partner_nr,
                        col=partner_hand[newest_discardable][0],
                    )
                else:
                    return Action(
                        HINT_NUMBER,
                        self.partner_nr,
                        num=partner_hand[newest_discardable][1],
                    )

        if force:
            nums = [card[1] for card in partner_hand]
            return Action(HINT_NUMBER, self.partner_nr, num=min(nums))
        else:
            return self._discard(True)

    def _discard(self, force=False):

        # discard something
        if self.last_state.get_num_hints() == 8:
            return self._hint(True)

        discard_index = -1
        highest_discard_probability = 0.0
        # discard highest probability discardable
        weighted_knowledge = weight_knowledge(self.knowledge, self.hint_weights)

        for i in range(len(weighted_knowledge)):
            discard_probability = slot_discardable_pct(weighted_knowledge[i], self.last_state.get_board())
            if discard_probability > highest_discard_probability:
                highest_discard_probability = discard_probability
                discard_index = i

        if discard_index != -1:
            return Action(DISCARD, cnr=discard_index)
        # discard oldest
        return Action(DISCARD, cnr=0)

    def get_action(self, game_state, player_model):
        self.turn += 1

        # first turn
        if self.last_model is None:
            self.last_model = player_model
            self.knowledge = copy.deepcopy(self.last_model.get_knowledge())
        if self.last_state is None:
            self.last_state = game_state

        action_type = self._decide()
        action = getattr(self, action_type)()
        # time.sleep(1)
        return action

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

        weighted_knowledge = weight_knowledge(self.knowledge, self.hint_weights)

        for i in range(len(weighted_knowledge)):
            updated_pct = slot_playable_pct(weighted_knowledge[i], new_board)
            # hint to play
            if updated_pct > 0.95:
                self.todo.append(i)
            discardable_pct = slot_discardable_pct(weighted_knowledge[i], new_board)
            if discardable_pct > 0.95:
                self.todo.append(i)

    # don't need the hint indices, of course
    def _receive_play(self, action, player, new_state, new_model):
        self.last_model = new_model
        self.last_state = new_state
        self.partner_todo = [i for i in self.partner_todo if i != action.cnr]
        if len(self.partner_hint_weights) == len(new_state.get_all_knowledge()[self.partner_nr]):
            self.partner_hint_weights[action.cnr] = [
                [1 for _ in range(5)] for _ in range(5)
            ]
        else:
            del self.partner_hint_weights[action.cnr]

    def _receive_discard(self, action, player, new_state, new_model):
        self.last_model = new_model
        self.last_state = new_state
        self.partner_todo = [i for i in self.partner_todo if i != action.cnr]
        if len(self.partner_hint_weights) == len(new_state.get_all_knowledge()[self.partner_nr]):
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
                # delete index from todo list
                self.todo = [i for i in self.todo if i != action.cnr]
            elif action.type in [HINT_COLOR, HINT_NUMBER]:
                new_board = new_state.get_board()
                partner_knowledge = copy.deepcopy(
                    new_state.get_all_knowledge()[self.partner_nr]
                )

                # empty list: no new info gained; bad hint
                if not hint_indices:
                    return

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

                weighted_knowledge = weight_knowledge(
                    partner_knowledge, self.partner_hint_weights
                )

                for i in range(len(weighted_knowledge)):
                    updated_pct = slot_playable_pct(weighted_knowledge[i], new_board)
                    # hint to play
                    if updated_pct > 0.95:
                        self.partner_todo.append(i)
                    discardable_pct = slot_discardable_pct(
                        weighted_knowledge[i], new_board
                    )
                    if discardable_pct > 0.95:
                        self.partner_todo.append(i)
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
