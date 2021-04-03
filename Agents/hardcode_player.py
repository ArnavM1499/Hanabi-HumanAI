from copy import deepcopy
from pprint import pprint
import common_game_functions as cgf
import Agents.common_player_functions as cpf
from Agents.player import Player, Action


class HardcodePlayer2(Player):
    def __init__(self, name, pnr, **kwargs):

        # Basic Info
        self.name = name
        self.pnr = pnr
        self.turn = 0
        self.debug = False

        # TOFIX Hardcoded for 2 players
        self.partner_nr = 1 - self.pnr
        self.card_nr = 5

        # Records
        self.last_state = None
        self.last_model = None
        self.knowledge = None

        self.index_play = []
        self.index_play_candidate = []
        self.index_discard = []
        self.index_discard_candidate = []
        self.index_protect = []
        self.index_hinted = []

        self.partner_play = []
        self.partner_play_candidate = []
        self.partner_discard = []
        self.partner_protect = []
        self.partner_hinted = []

        # Decision pattern matches
        # [(func : Player -> bool, action : Action Type)]
        self.decision_protocol = [
            (lambda p, s, m: p.index_play != [] and p.partner_play != [], "_execute"),
            (lambda p, s, m: p.index_play != [] and s.get_num_hints() > 1, "_execute"),
            # (lambda p, s, m: p.index_play != [], "_execute"),
            (lambda p, s, m: p.partner_play == [] and s.get_num_hints() > 0, "_hint"),
            (lambda p, s, m: p.index_play == [], "_hint"),
        ]

        # Parameters
        self.risk_play = {0: 0.5, 5: 0, 12: 0.4, 30: 1}

        for k, v in kwargs.items():
            setattr(self, k, v)

    def inform(self, action, player, new_state, new_model):

        self._update_state(new_state, new_model)
        if action.pnr != self.pnr:
            return

        board = new_state.get_board()
        trash = new_state.get_trash()
        knowledge = new_model.get_knowledge()

        if self.turn == 0:
            if action.type == cgf.HINT_COLOR:
                hinted_indices = new_state.get_hinted_indices()
                self.index_play.append(hinted_indices[0])
                self.index_discard.extend(hinted_indices[1:])
            elif action.type == cgf.HINT_NUMBER:
                if action.num == 1:
                    self.index_play.extend(new_state.get_hinted_indices())
                else:
                    for i in range(self.card_nr):
                        for j in range(5):
                            for k in range(action.num - 1):
                                self.knowledge[i][j][k] = 0
            if self.debug:
                pprint(self.__dict__)
                print("\n\n\n")
            return

        if action.type in [cgf.HINT_COLOR, cgf.HINT_NUMBER]:

            hinted_indices = new_state.get_hinted_indices()
            assert hinted_indices != []
            hinted_indices.sort()

            (
                self.index_play,
                self.index_play_candidate,
                self.index_discard,
                self.index_protect,
            ) = self._interpret(
                hinted_indices,
                knowledge,
                board,
                trash,
                self.index_play,
                self.index_play_candidate,
                self.index_discard,
                self.index_protect,
                is_five=(action.type == cgf.HINT_NUMBER and action.num == 5),
            )

            if self.debug:
                pprint(self.__dict__)
                print("\n\n\n")

            for idx in hinted_indices:
                if idx not in self.index_hinted:
                    self.index_hinted.append(idx)

        elif action.type in [cgf.PLAY, cgf.DISCARD]:

            self._update_index(action.cnr, partner=True)

            if action.type == cgf.DISCARD and new_state.get_num_hints() > 1:
                for i in range(self.card_nr):
                    if not (
                        (i in self.index_play)
                        or (i in self.index_play_candidate)
                        or (i in self.index_protect)
                    ):
                        self.index_discard_candidate.append(i)

        else:
            assert False

    def _interpret(
        self,
        hinted_indices,
        knowledge,
        board,
        trash,
        cur_play,
        cur_play_candidate,
        cur_discard,
        cur_protect,
        is_five=False,
    ):

        play = cur_play.copy()
        play_candidate = cur_play_candidate.copy()
        discard = cur_discard.copy()
        protect = cur_protect.copy()

        flag = False
        for idx in hinted_indices:
            card = knowledge[idx]
            if cpf.slot_playable_pct(card, board) > 0.8:
                play.append(idx)
                flag = True
            elif cpf.slot_discardable_pct(card, board, trash) > 0.98:
                discard.append(idx)
            elif is_five:
                protect.append(idx)

        if not flag:
            newest = hinted_indices[-1]
            card = knowledge[newest]
            if cpf.slot_playable_pct(card, board) > 0:
                play.append(newest)
            else:
                protect.append(newest)
            for idx in hinted_indices[:-1]:
                card = knowledge[idx]
                if cpf.slot_playable_pct(card, board) > 0:
                    play_candidate.append(idx)

        i = 0
        while i < len(play):
            card = knowledge[play[i]]
            if cpf.slot_playable_pct(card, board) < 0.02:
                del play[i]
            else:
                i += 1
        i = 0
        while i < len(play_candidate):
            card = knowledge[play_candidate[i]]
            if cpf.slot_playable_pct(card, board) < 0.02:
                del play_candidate[i]
            else:
                i += 1
        i = 0
        while i < len(discard):
            card = knowledge[discard[i]]
            if cpf.slot_discardable_pct(card, board, trash) < 0.02:
                del discard[i]
            else:
                i += 1
        i = 0
        while i < len(protect):
            card = knowledge[protect[i]]
            if cpf.slot_discardable_pct(card, board, trash) > 0.5:
                del protect[i]
            else:
                i += 1

        return sorted(play), sorted(play_candidate), sorted(discard), sorted(protect)

    def get_action(self, state, model):

        self.turn += 1

        # Always hint the smallest number in the first turn
        if self.turn == 1 and self.last_state is None:
            partner = state.get_hands()[self.partner_nr]
            min_num = min([x[1] for x in partner])

            if self.debug:
                print("first turn")

            return Action(cgf.HINT_NUMBER, self.partner_nr, num=min_num)

        self._update_state(state, model)

        # Pattern matcing [self._decide() in version 1]
        chosen_action = None
        for i, (func, action) in enumerate(self.decision_protocol):
            if func(self, state, model):
                force = False
                if action.endswith("_force"):
                    action = action[:-6]
                chosen_action = getattr(self, action)(force=force)
                if self.debug:
                    print("executing pattern ", i)
                print(chosen_action)
                break

        # Default action
        if chosen_action is None:
            if self.debug:
                print("Using Defualt action!")
            chosen_action = self._execute()
        # TODO change to other class for less agressive play

        # Post processes
        if chosen_action.type in [cgf.PLAY, cgf.DISCARD]:
            self._update_index(chosen_action.cnr)

        return chosen_action

    def _execute(self, force=False):

        board = self.last_state.get_board()
        knowledge = self.knowledge

        self.index_play.sort()
        while self.index_play != []:
            idx = self.index_play.pop()
            card = knowledge[idx]
            if cpf.slot_playable_pct(card, board) > 0:
                return Action(cgf.PLAY, cnr=idx)
            else:
                self.index_discard.append(idx)

        # play at risk
        risk_threshold = 0
        for turn, thresh in self.risk_play.items():
            if turn >= self.turn:
                risk_threshold = 1 - thresh
                break
        idx = None
        max_pct = 0
        for i in self.index_play_candidate:
            card = knowledge[i]
            if cpf.slot_playable_pct(card, board) > max_pct:
                idx = i
                max_pct = cpf.slot_playable_pct(card, board)
        if idx and max_pct > risk_threshold:
            return Action(cgf.PLAY, cnr=idx)

        # TODO add more conditions for more aggressive play

        if force:
            self.index_play_candidate.sort()
            while self.index_play_candidate != []:
                idx = self.index_play_candidate.pop()
                card = knowledge[idx]
                if cpf.slot_playable(card, board) > 0.02:
                    return Action(cgf.PLAY, cnr=idx)
                else:
                    self.index_discard.append(idx)
            return Action(cgf.PLAY, cnr=self.card_nr - 1)
        else:
            if self.last_state.get_num_hints() > 1:
                return self._hint(force=True)
            else:
                return self._discard(force=True)

    def _discard(self, force=False):

        if self.index_discard != []:
            return Action(cgf.DISCARD, cnr=min(self.index_discard))

        if self.index_discard_candidate != []:
            return Action(cgf.DISCARD, cnr=min(self.index_discard_candidate))

        for i in range(self.card_nr):
            if i not in self.index_protect:
                return Action(cgf.DISCARD, cnr=i)

        return Action(cgf.DISCARD, cnr=0)

    def _evaluate_partner(
        self, hands, predicted_play, predicted_play_candidate, predicted_discard
    ):

        # having playable cards in play list is GOOD +3pt
        # having unplayable cards in play list is BAD -2pt
        # having unplayable card at the front of play list is VERY BAD -5pt
        # having discardable cards in discard list is GOOD +1pt
        # having playable cards in play candidates is GOOD +1pt
        # having unplayable cards in play candidates is BAD -0.8pt

        board = self.last_state.get_board()
        trash = self.last_state.get_trash()

        score = 0
        for i in predicted_play:
            if cpf.card_playable(hands[i], board):
                score += 3
            else:
                score -= 2
        if predicted_play != [] and (
            not cpf.card_playable(hands[max(predicted_play)], board)
        ):
            score -= 5

        for i in predicted_play_candidate:
            if cpf.card_playable(hands[i], board):
                score += 1
            else:
                score -= 0.8

        for i in predicted_discard:
            if cpf.card_discardable(hands[i], board, trash):
                score += 1

        return score

    def _hint(self, force=False):

        partner_hand = self.last_state.get_hands()[self.partner_nr]
        partner_knowledge = self.last_state.get_all_knowledge()[self.partner_nr]
        board = self.last_state.get_board()
        trash = self.last_state.get_trash()

        max_score = self._evaluate_partner(
            partner_hand,
            self.partner_play,
            self.partner_play_candidate,
            self.partner_discard,
        )
        best_action = None

        if self.debug:
            print("comparing hints")

        for action in self.last_state.get_valid_actions():

            if action.type not in [cgf.HINT_NUMBER, cgf.HINT_COLOR]:
                continue
            pred_play = self.partner_play.copy()
            pred_play_candidate = self.partner_play_candidate.copy()
            pred_discard = self.partner_discard.copy()
            pred_protect = self.partner_protect.copy()

            hinted = []
            for i, card in enumerate(partner_hand):
                if (action.type == cgf.HINT_NUMBER and card[1] == action.num) or (
                    action.type == cgf.HINT_COLOR and card[0] == action.col
                ):
                    hinted.append(i)

            (
                pred_play,
                pred_play_candidate,
                pred_discard,
                pred_protect,
            ) = self._interpret(
                hinted,
                partner_knowledge,
                board,
                trash,
                pred_play,
                pred_play_candidate,
                pred_discard,
                pred_protect,
                is_five=(action.type == cgf.HINT_NUMBER and action.num == 5),
            )

            score = self._evaluate_partner(
                partner_hand,
                pred_play,
                pred_play_candidate,
                pred_discard,
            )

            if self.debug:
                print("Action: ", str(action), " evaluates to: ")
                print("  score: ", score)
                print("  play: ", self.partner_play, " >> ", pred_play)
                print(
                    "  candidate: ",
                    self.partner_play_candidate,
                    " >> ",
                    pred_play_candidate,
                )
                print("  discard: ", self.partner_discard, " >> ", pred_discard)
                print("")

            if score > max_score:
                max_score = score
                best_action = action

        if best_action is None:
            return self._discard(force=True)

        return best_action

    def _update_index(self, idx, partner=False):

        if self:
            prefix = "index_"
            del self.knowledge[idx]
            self.knowledge.append([cgf.COUNTS.copy() for _ in range(5)])
        else:
            prefix = "partner_"

        for attr in dir(self):
            if attr.startswith(prefix):
                L = getattr(self, attr)
                if not isinstance(L, list):
                    continue
                setattr(self, attr, [x if x < idx else x - 1 for x in L if x != idx])

    def _update_state(self, new_state, new_model):

        self.last_model = deepcopy(new_model)
        self.last_state = deepcopy(new_state)
        new_knowledge = new_model.get_knowledge()
        if self.knowledge:
            merged = []
            for old, new in zip(self.knowledge, new_knowledge):
                temp = []
                for i in range(5):
                    temp.append([])
                    for j in range(5):
                        temp[-1].append(min(old[i][j], new[i][j]))
                merged.append(temp)
            self.knowledge = merged
        else:
            self.knowledge = deepcopy(new_model.get_knowledge())
