from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import *


class HardcodePlayer(Player):
    def __init__(self, name, pnr, **kwargs):
        self.name = name
        self.pnr = pnr
        self.partner_nr = 1 - self.pnr  # hard code for two players
        self.turn = 0
        self.todo = []  # [(ACTION, turns)]
        self.partner_play = []  # [index to play]
        self.partner_protect = []  # [index to be protected]
        self.last_state = None
        self.last_model = None
        self.protect = []
        self.hinted = []
        self.log = []
        self.nr_cards = 5
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _sort_todo(self):

        # discard from old to new
        to_discard = sorted(
            [(a, t) for a, t in self.todo if a.type != PLAY], key=lambda x: x[-1]
        )
        # play from highest confidence
        to_play = sorted(
            [(a, t) for a, t in self.todo if a.type == PLAY],
            key=lambda x: slot_playable_pct(
                self.last_model.get_knowledge()[x[0].cnr],
                self.last_state.get_board()
            ),
        )

        # play has higher priority than discard
        return to_discard + to_play

    def _update_index(self, idx):

        i = 0
        while i < len(self.todo):
            action, turn = self.todo[i]
            if action.cnr == idx:
                del self.todo[i]
            else:
                if action.cnr > idx:
                    action.cnr -= 1
                i += 1

        i = 0
        while i < len(self.protect):
            if self.protect[i] == idx:
                del self.protect[i]
            else:
                if self.protect[i] > idx:
                    self.protect[i] -= 1
                i += 1

        i = 0
        while i < len(self.hinted):
            if self.hinted[i] == idx:
                del self.hinted[i]
            else:
                if self.hinted[i] > idx:
                    self.hinted[i] -= 1
                i += 1

    def _interpret(self, new_state, new_model):

        # update self.todo based on partner's action

        if new_model.get_actions()[self.partner_nr] != []:

            last_action = new_model.get_actions()[self.partner_nr][-1]

            if last_action.type in [HINT_COLOR, HINT_NUMBER]:
                # if there is a new hint
                new_hint = last_action
                board = new_state.get_board()
                knowledge = new_model.get_knowledge()
                index2possible = {i: get_possible(k) for i, k in enumerate(knowledge)}

                # get card index hinted
                index_hinted = []
                for i, possible in index2possible.items():
                    flag = True
                    for (col, num) in possible:
                        if (new_hint.type == HINT_COLOR and col != new_hint.color) or (
                            new_hint.type == HINT_NUMBER and num != new_hint.num
                        ):
                            flag = False
                            break
                    if flag:
                        index_hinted.append(i)

                # check need to protect
                for i, possible in index2possible.items():
                    flag = True
                    # check for 5s
                    for (col, nr) in possible:
                        if nr != 5:
                            flag = False
                            break
                    if flag:
                        # check for trash
                        for (col, nr) in possible:
                            nr_in_trash = new_state.get_trash().count((col, nr))
                            if (nr == 1 and nr_in_trash <= 1) or (
                                nr != 1 and nr_in_trash < 1
                            ):
                                flag = False
                                break
                    if flag and (i not in self.protect):
                        self.protect.append(i)

                # check playable
                for i, possible in index2possible.items():
                    if playable(possible, board):
                        self.todo.append((Action(PLAY, cnr=i), self.turn))
                    elif i in index_hinted and potentially_playable(possible, board):
                        self.todo.append((Action(PLAY, cnr=i), self.turn))

                # check discardable
                for i in index_hinted:
                    if discardable(index2possible[i], new_state.get_board()):
                        self.todo.append((Action(DISCARD, cnr=i), self.turn))

                for i in index_hinted:
                    if not i in self.hinted:
                        self.hinted.append(i)
            else:
                # if there is no new hint, update todo for partner
                i = last_action.cnr
                self.partner_play = [
                    x if x < i else x - 1 for x in self.partner_play if x != i
                ]
                self.partner_protect = [
                    x if x < i else x - 1 for x in self.partner_protect if x != i
                ]

        self.last_model = new_model
        self.last_state = new_state

        self._sort_todo()

    def _decide(self):

        num_hints = self.last_state.get_num_hints()

        # decide what to do next (general category of action)
        if self.todo != [] and self.partner_play != []:
            self.log.append(1)
            return "_execute"
        if self.todo != [] and num_hints >= 2:
            self.log.append(2)
            return "_execute"
        if self.partner_play == [] and num_hints >= 1:
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

        if self.todo != []:
            action, turn = self.todo.pop()
            if action.type == PLAY and (
                0.4 > percent_playable(
                    get_possible(self.last_model.get_knowledge()[action.cnr]),
                    self.last_state.get_board(),
                )
            ):
                action.type = DISCARD
            return action
        else:
            return self._hint(True)

    def _hint(self, force=False):

        if self.last_state.get_num_hints() == 0:
            return self._discard(True)

        partner = self.last_state.get_hands()[self.partner_nr]
        board = self.last_state.get_board()
        trash = self.last_state.get_trash()

        # hint the partner

        # TODO hint to override

        # choose which card to hint
        play = []
        protect = set()
        for i, (col, num) in enumerate(partner):
            # playable
            if board[col][1] + 1 == num:
                play.append((i, (col, num)))
            # TODO need to protect
            # if num == 5 or (num != 1 and (col, num) in trash) or (num == 1 and trash.count(col, num) > 1):
            #     if not i in self.partner_protect:
            #         protect.add(i)

        if len(play) != 0:
            # hint smallest unambiguous number
            nums = sorted({card[1] for i, card in play})
            for candidate in nums:
                flag = True
                for i, (col, num) in enumerate(partner):
                    if num == candidate and (not (col, num) in play):
                        flag = False
                        break
                if flag:
                    for i, (col, num) in play[::-1]:
                        if num == candidate:
                            self.partner_play.append(i)
                            break
                    return Action(HINT_NUMBER, self.partner_nr, num=candidate)
            # if no unambiguous number, try unambiguous color
            cols = {card[0] for i, card in play}
            for candidate in cols:
                flag = True
                for i, (col, num) in enumerate(partner):
                    if col == candidate and (not (col, num) in play):
                        flag = False
                        break
                if flag:
                    for i, (col, num) in play[::-1]:
                        if col == candidate:
                            self.partner_play.append(i)
                            break
                    return Action(HINT_COLOR, self.partner_nr, col=candidate)
            # TODO assume play from newest

            # hint the newest playable
            if len(play) > 0:
                index, (col, num) = play[0]
                self.partner_play.append(index)
                return Action(HINT_NUMBER, self.partner_nr, num=num)

        # hint to protect
        if 5 in [x[1] for x in partner]:
            return Action(HINT_NUMBER, self.partner_nr, num=5)

        # default action
        if force or self.last_state.get_num_hints() == 8:
            # hint smallest number
            nums = [card[1] for card in partner]
            return Action(HINT_NUMBER, self.partner_nr, num=min(nums))
        else:
            return self._discard(True)

    def _discard(self, force=False):

        # discard something

        if self.last_state.get_num_hints() == 8:
            return self._hint(True)

        # discard known discardable
        for action, turn in self.todo:
            if action.type == DISCARD:
                return action

        # discard unhinted
        for i in range(self.nr_cards):
            if (not i in self.protect) and (not i in self.hinted):
                return Action(DISCARD, cnr=i)

        # discard unprotected
        for i in range(self.nr_cards):
            if not i in self.protect:
                return Action(DISCARD, cnr=i)

        # discard oldest
        return Action(DISCARD, cnr=0)

    def get_action(self, game_state, player_model):

        self.turn += 1
        self._interpret(game_state, player_model)
        action_type = self._decide()
        action = getattr(self, action_type)()
        if action.type in [PLAY, DISCARD]:
            self._update_index(action.cnr)
        return action
