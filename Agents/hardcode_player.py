from pprint import pprint
from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Player, Action


class HardcodePlayer2(Player):
    def __init__(self, name, pnr, **kwargs):

        # Basic Info
        self.name = name
        self.pnr = pnr
        self.turn = 0
        self.debug = True

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
            # (lambda p, s, m: p.index_play != [] and p.partner_play != [], "_execute"),
            # (lambda p, s, m: p.index_play != [] and s.get_num_hints() > 1, "_execute"),
            (lambda p, s, m: p.index_play != [], "_execute"),
            (lambda p, s, m: p.partner_play == [] and s.get_num_hints() > 0, "_hint"),
            (lambda p, s, m: p.index_play == [], "_hint"),
        ]

        # Parameters
        self.risk_play = {10: 0, 30: 0.4, 50: 1}

        for k, v in kwargs.items():
            setattr(self, k, v)

    def inform(self, action, player, new_state, new_model):

        self._update_state(new_state, new_model)
        if action.pnr == self.pnr:
            return

        board = new_state.get_board()
        knowledge = new_model.get_knowledge()

        if self.turn == 1:
            if action.type == HINT_COLOR:
                hinted_indices = action.get_hinted_indices()
                self.index_play.append(hinted_indices[0])
                self.index_discard.extend(hinted_indices[1:])
            elif action.type == HINT_NUMBER:
                if action.num == 1:
                    self.index_play.extend(new_state.get_hinted_indices())
                # else:
                #     for i in range(self.card_nr):
                #         for j in range(5):
                #             for k in range(action.num - 1):
                #                 self.knowledge[i][j][k] = 0
            if self.debug:
                pprint(self.__dict__)
            return

        if action.type in [HINT_COLOR, HINT_NUMBER]:

            hinted_indices = new_state.get_hinted_indices()
            assert hinted_indices != []
            hinted_indices.sort()

            new_play, new_play_candidate, new_discard, new_protect = self._interpret(
                hinted_indices,
                knowledge,
                board,
                is_five=(action.type == HINT_NUMBER and action.num == 5),
            )
            for i in new_play:
                if i not in self.index_play:
                    self.index_play.append(i)
            for i in new_play_candidate:
                if i not in self.index_play_candidate:
                    self.index_play_candidate.append(i)
            for i in new_discard:
                if i not in self.index_discard:
                    self.index_discard.append(i)
            for i in new_protect:
                if i not in self.index_protect:
                    self.index_protect.append(i)
            self.index_play.sort()
            self.index_play_candidate.sort()
            self.index_discard.sort()
            self.index_protect.sort()

            if self.debug:
                pprint(self.__dict__)

            # flag = False
            # for idx in hinted_indices:
            #     card = knowledge[idx]
            #     if slot_playable_pct(card, board) > 0.8:
            #         self.index_play.append(idx)
            #         flag = True
            #     elif slot_discardable_pct(card, board) > 0.98:
            #         self.index_discard.append(idx)
            #     elif action.type == HINT_NUMBER and action.num == 5:
            #         self.index_protect.append(idx)
            #         flag = True

            # if not flag:
            #     newest = hinted_indices[-1]
            #     card = knowledge[newest]
            #     if slot_playable_pct(card, board) > 0:
            #         self.index_play.append(newest)
            #     else:
            #         self.index_protect.append(newest)
            #     # TODO add rest of the cards to candidate play list for more
            #     # aggressive play

            for idx in hinted_indices:
                if idx not in self.index_hinted:
                    self.index_hinted.append(idx)

        elif action.type in [PLAY, DISCARD]:

            self._update_index(action.cnr, partner=True)

            if action.type == DISCARD and new_state.get_num_hints() > 1:
                for i in range(self.card_nr):
                    if not (
                        (i in self.index_play)
                        or (i in self.index_play_candidate)
                        or (i in self.index_protect)
                    ):
                        self.index_discard_candidate.append(i)

        else:
            assert False

    def _interpret(self, hinted_indices, knowledge, board, is_five=False):

        play = []
        play_candidate = []
        discard = []
        protect = []

        flag = False
        for idx in hinted_indices:
            card = knowledge[idx]
            if slot_playable_pct(card, board) > 0.8:
                play.append(idx)
                flag = True
            elif slot_discardable_pct(card, board) > 0.98:
                discard.append(idx)
            elif is_five:
                protect.append(idx)

        if not flag:
            newest = hinted_indices[-1]
            card = knowledge[newest]
            if slot_playable_pct(card, board) > 0:
                play.append(newest)
            else:
                protect.append(newest)
            for idx in hinted_indices[:-1]:
                card = knowledge[idx]
                if slot_playable_pct(card, board) > 0:
                    play_candidate.append(idx)

        return play, play_candidate, discard, protect

    def get_action(self, state, model):

        self.turn += 1
        self._update_state(state, model)

        # Always hint the smallest number in the first turn
        if self.turn == 1 and self.last_state is None:
            partner = state.get_hands()[self.partner_nr]
            min_num = min([x[1] for x in partner])
            return Action(HINT_NUMBER, self.partner_nr, num=min_num)

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
        if chosen_action.type in [PLAY, DISCARD]:
            self._update_index(chosen_action.cnr)

        return chosen_action

    def _execute(self, force=False):

        board = self.last_state.get_board()
        knowledge = self.knowledge

        self.index_play.sort()
        while self.index_play != []:
            idx = self.index_play.pop()
            card = knowledge[idx]
            if slot_playable_pct(card, board) > 0:
                return Action(PLAY, cnr=idx)
            else:
                self.index_discard.append(idx)

        # play at risk
        risk_threshold = 0
        for turn, thresh in self.risk_play.items():
            if turn >= self.turn:
                risk_threshold = thresh
                break
        idx = None
        max_pct = 0
        for i in self.index_play_candidate:
            card = knowledge[i]
            if slot_playable_pct(card, board) > max_pct:
                idx = i
                max_pct = slot_playable_pct(card, board)
        if idx and max_pct > risk_threshold:
            return Action(PLAY, cnr=idx)

        # TODO add more conditions for more aggressive play

        if force:
            self.index_play_candidate.sort()
            while self.index_play_candidate != []:
                idx = self.index_play_candidate.pop()
                card = knowledge[idx]
                if slot_playable(card, board) > 0.02:
                    return Action(PLAY, cnr=idx)
                else:
                    self.index_discard.append(idx)
            return Action(PLAY, cnr=self.card_nr - 1)
        else:
            if self.last_state.get_num_hints() > 1:
                return self._hint(force=True)
            else:
                return self._discard(force=True)

    def _discard(self, force=False):

        if self.index_discard != []:
            return Action(DISCARD, cnr=min(self.index_discard))

        if self.index_discard_candidate != []:
            return Action(DISCARD, cnr=min(self.index_discard_candidate))

        for i in range(self.card_nr):
            if i not in self.index_protect:
                return Action(DISCARD, cnr=i)

        return Action(DISCARD, cnr=0)

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

        score = 0
        for i in predicted_play:
            if card_playable(hands[i], board):
                score += 3
            else:
                score -= 2
        if predicted_play != [] and (
            not card_playable(hands[max(predicted_play)], board)
        ):
            score -= 5

        for i in predicted_play_candidate:
            if card_playable(hands[i], board):
                score += 1
            else:
                score -= 0.8

        for i in predicted_discard:
            if card_discardable(hands[i], board):
                score += 1

        return score

    def _hint(self, force=False):

        partner_hand = self.last_state.get_hands()[self.partner_nr]
        partner_knowledge = self.last_state.get_all_knowledge()[self.partner_nr]
        board = self.last_state.get_board()

        max_score = -100
        best_action = None
        for action in self.last_state.get_valid_actions():

            if action.type not in [HINT_NUMBER, HINT_COLOR]:
                continue
            pred_play = self.partner_play.copy()
            pred_play_candidate = self.partner_play_candidate.copy()
            pred_discard = self.partner_discard.copy()
            pred_protect = self.partner_protect.copy()

            hinted = []
            for i in range(self.card_nr):
                if (
                    action.type == HINT_NUMBER and partner_hand[i][1] == action.num
                ) or (action.type == HINT_COLOR and partner_hand[i][0]):
                    hinted.append(i)

            try:
                (
                    new_play,
                    new_play_candidate,
                    new_discard,
                    new_protect,
                ) = self._interpret(
                    hinted,
                    partner_knowledge,
                    board,
                    is_five=(action.type == HINT_NUMBER and action.num == 5),
                )
            except:
                import pdb

                pdb.set_trace()

            for i in new_play:
                if i not in pred_play:
                    pred_play.append(i)
            for i in new_play_candidate:
                if i not in pred_play_candidate:
                    pred_play_candidate.append(i)
            for i in new_discard:
                if i not in pred_discard:
                    pred_discard.append(i)
            for i in new_protect:
                if i not in pred_protect:
                    pred_protect.append(i)

            score = self._evaluate_partner(
                sorted(partner_hand),
                sorted(pred_play),
                sorted(pred_play_candidate),
                sorted(pred_discard),
            )
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
            self.knowledge.append([[3, 2, 2, 2, 1] for _ in range(5)])
        else:
            prefix = "partner_"

        for attr in dir(self):
            if attr.startswith(prefix):
                L = getattr(self, attr)
                if not isinstance(L, list):
                    continue
                setattr(self, attr, [x if x < idx else x - 1 for x in L if x != idx])

    def _update_state(self, new_state, new_model):

        self.last_model = new_model
        self.last_state = new_state
        new_knowledge = new_model.get_knowledge()
        self.knowledge = new_knowledge
        # if self.knowledge:
        #     for i in range(self.card_nr):
        #         for j in range(5):
        #             for k in range(5):
        #                 self.knowledge[i][j][k] = min(
        #                     [
        #                         self.knowledge[i][j][k],
        #                         new_knowledge[i][j][k],
        #                     ]
        #                 )
        # else:
        #     self.knowledge = new_model.get_knowledge()
