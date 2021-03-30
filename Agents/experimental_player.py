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
# things like self.todo will behave incorrectly


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
        # discard from old to new

    #       to_discard = sorted(
    #            [(a, t) for a, t in self.todo if a.type != PLAY], key=lambda x: x[-1]
    #        )
    #        # play from highest confidence
    #        to_play = sorted(
    #            [(a, t) for a, t in self.todo if a.type == PLAY],
    #            key=lambda x: slot_playable_pct(
    #                self.last_model.get_knowledge()[x[0].cnr],
    #                self.last_state.get_board()
    #            ),
    #        )

    # play has higher priority than discard
    #        return to_discard + to_play

    #    def _update_index(self, idx):
    #        return#

    #        i = 0
    #        while i < len(self.todo):
    #            action, turn = self.todo[i]
    #            if action.cnr == idx:
    #                del self.todo[i]
    #            else:
    #                if action.cnr > idx:
    #                    action.cnr -= 1
    #                i += 1#

    #        i = 0
    #        while i < len(self.protect):
    #            if self.protect[i] == idx:
    #                del self.protect[i]
    #            else:
    #                if self.protect[i] > idx:
    #                    self.protect[i] -= 1
    #                i += 1#

    #        i = 0
    #        while i < len(self.hinted):
    #            if self.hinted[i] == idx:
    #                del self.hinted[i]
    #            else:
    #                if self.hinted[i] > idx:
    #                    self.hinted[i] -= 1
    #                i += 1

    def _decide(self):

        return "_execute"
        num_hints = self.last_state.get_num_hints()

        # decide what to do next (general category of action)
        if self.todo != [] and self.partner_todo != []:
            self.log.append(1)
            return "_execute"
        if self.todo != [] and num_hints >= 2:
            self.log.append(2)
            return "_execute"
        if self.partner_todo == [] and num_hints >= 1:
            self.log.append(3)
            return "_hint"
        if self.todo == []:
            self.log.append(4)
            return "_hint"
        if self.todo == []:
            self.log.append(5)
            return "_discard"

        # default actions
        if self.todo != []:
            return "_execute"
        elif num_hints < 8:
            return "_discard"
        else:
            return "_hint"

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
        partner_knowledge = copy.deepcopy(self.last_model.get_all_knowledge())[
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

        weighted_partner_knowledge = weight_knowledge(
            partner_knowledge, self.partner_hint_weights
        )

        # if card is playable, hint it with the most info gain
        while discardable:
            newest_discardable = discardable[-1]
            if newest_discardable in self.partner_todo:
                del discardable[-1]
                continue
            if newest_discardable >= 0:
                hint_type = best_hint_type(
                    partner_hand, newest_discardable, weighted_partner_knowledge
                )
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

    #        partner = self.last_state.get_hands()[self.partner_nr]
    #        board = self.last_state.get_board()
    #        trash = self.last_state.get_trash()#

    #        # hint the partner#

    #        # TODO hint to override#

    #        # choose which card to hint
    #        play = []
    #        protect = set()
    #        for i, (col, num) in enumerate(partner):
    #            # playable
    #            if board[col][1] + 1 == num:
    #                play.append((i, (col, num)))
    #            # TODO need to protect
    #            # if num == 5 or (num != 1 and (col, num) in trash) or (num == 1 and trash.count(col, num) > 1):
    #            #     if not i in self.partner_protect:
    #            #         protect.add(i)#

    #        if len(play) != 0:
    #            # hint smallest unambiguous number
    #            nums = sorted({card[1] for i, card in play})
    #            for candidate in nums:
    #                flag = True
    #                for i, (col, num) in enumerate(partner):
    #                    if num == candidate and (not (col, num) in play):
    #                        flag = False
    #                        break
    #                if flag:
    #                    for i, (col, num) in play[::-1]:
    #                        if num == candidate:
    #                            self.partner_play.append(i)
    #                            break
    #                    return Action(HINT_NUMBER, self.partner_nr, num=candidate)
    #            # if no unambiguous number, try unambiguous color
    #            cols = {card[0] for i, card in play}
    #            for candidate in cols:
    #                flag = True
    #                for i, (col, num) in enumerate(partner):
    #                    if col == candidate and (not (col, num) in play):
    #                        flag = False
    #                        break
    #                if flag:
    #                    for i, (col, num) in play[::-1]:
    #                        if col == candidate:
    #                            self.partner_play.append(i)
    #                            break
    #                    return Action(HINT_COLOR, self.partner_nr, col=candidate)
    #            # TODO assume play from newest#

    #            # hint the newest playable
    #            if len(play) > 0:
    #                index, (col, num) = play[0]
    #                self.partner_play.append(index)
    #                return Action(HINT_NUMBER, self.partner_nr, num=num)#

    #        # hint to protect
    #        if 5 in [x[1] for x in partner]:
    #            return Action(HINT_NUMBER, self.partner_nr, num=5)#

    #        # default action
    #        if force or self.last_state.get_num_hints() == 8:
    #            # hint smallest number
    #            nums = [card[1] for card in partner]
    #            return Action(HINT_NUMBER, self.partner_nr, num=min(nums))
    #        else:
    #            return self._discard(True)

    def _discard(self, force=False):

        # discard something

        if self.last_state.get_num_hints() == 8:
            return self._hint(True)

        # discard known discardable
        # for action, turn in self.todo:
        #    if action.type == DISCARD:
        #        return action

        # discard unhinted
        # for i in range(self.nr_cards):
        #    if (not i in self.protect) and (not i in self.hinted):
        #        return Action(DISCARD, cnr=i)

        # discard unprotected
        # for i in range(self.nr_cards):
        #    if not i in self.protect:
        #        return Action(DISCARD, cnr=i)

        # discard oldest
        return Action(DISCARD, cnr=0)

    def get_action(self, game_state, player_model):
        self.turn += 1

        # print(player_model.get_all_knowledge())

        # first turn
        if self.last_model is None:
            self.last_model = player_model
            self.knowledge = copy.deepcopy(self.last_model.get_knowledge())
        if self.last_state is None:
            self.last_state = game_state

        action_type = self._decide()
        action = getattr(self, action_type)()
        # if action.type in [PLAY, DISCARD]:
        #    self._update_index(action.cnr)
        # time.sleep(1)
        return action

    # for 2 player the only hints we need to consider are hints about our cards
    # this will need to be revisited if we generalize to more players
    def _receive_hint(self, action, player, new_state, new_model, hint_indices):
        self.last_model = new_model
        self.last_state = new_state
        assert action.type in [HINT_COLOR, HINT_NUMBER] and player == self.partner_nr
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

        # get card index hinted
        # index_hinted = []

    #        for i, possible in index2possible.items():
    #            flag = True
    #            for (col, num) in possible:
    #                if (new_hint.type == HINT_COLOR and col != new_hint.color) or (
    #                        new_hint.type == HINT_NUMBER and num != new_hint.num
    #                ):
    #                    flag = False
    #                    break
    #            if flag:
    #                index_hinted.append(i)#

    #        # check need to protect
    #        for i, possible in index2possible.items():
    #            flag = True
    #            # check for 5s
    #            for (col, nr) in possible:
    #                if nr != 5:
    #                    flag = False
    #                    break
    #            if flag:
    #                # check for trash
    #                for (col, nr) in possible:
    #                    nr_in_trash = new_state.get_trash().count((col, nr))
    #                    if (nr == 1 and nr_in_trash <= 1) or (
    #                            nr != 1 and nr_in_trash < 1
    #                    ):
    #                        flag = False
    #                        break
    #            if flag and (i not in self.protect):
    #                self.protect.append(i)#

    #        # check playable
    #        for i, possible in index2possible.items():
    #            if playable(possible, board):
    #                self.todo.append((Action(PLAY, cnr=i), self.turn))
    #            elif i in index_hinted and potentially_playable(possible, board):
    #                self.todo.append((Action(PLAY, cnr=i), self.turn))#

    #        # check discardable
    #        for i in index_hinted:
    #            if discardable(index2possible[i], new_state.get_board()):
    #                self.todo.append((Action(DISCARD, cnr=i), self.turn))#

    #        for i in index_hinted:
    #            if not i in self.hinted:
    #                self.hinted.append(i)
    # don't need the hint indices, of course
    def _receive_play(self, action, player, new_state, new_model):
        self.last_model = new_model
        self.last_state = new_state
        self.partner_todo = [i for i in self.partner_todo if i != action.cnr]
        self.partner_hint_weights[action.cnr] = [
            [1 for _ in range(5)] for _ in range(5)
        ]

    def _receive_discard(self, action, player, new_state, new_model):
        self.last_model = new_model
        self.last_state = new_state
        self.partner_todo = [i for i in self.partner_todo if i != action.cnr]
        self.partner_hint_weights[action.cnr] = [
            [1 for _ in range(5)] for _ in range(5)
        ]

    # hint_indices is [] if the action is not a hint
    def inform(self, action, player, new_state, new_model):
        hint_indices = new_state.get_hinted_indices()
        if player == self.pnr:
            # maybe this part should be moved up to before playing?
            # reset knowledge if we played or discarded
            if action.type == PLAY or action.type == DISCARD:
                # print(action.cnr)
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
                    new_model.get_all_knowledge()[self.partner_nr]
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
        assert player == self.partner_nr
        last_action = action

        if last_action.type in [HINT_COLOR, HINT_NUMBER]:
            self._receive_hint(action, player, new_state, new_model, hint_indices)

        elif last_action.type == PLAY:
            self._receive_play(action, player, new_state, new_model)

        # discard
        else:
            self._receive_discard(action, player, new_state, new_model)
