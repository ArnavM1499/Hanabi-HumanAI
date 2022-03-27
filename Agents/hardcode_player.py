from contextlib import contextmanager
from copy import deepcopy
from itertools import permutations, product
import pickle
from pprint import pprint
from random import random, sample
from time import time
import common_game_functions as cgf
import Agents.common_player_functions as cpf
from Agents.player import Player, Action


@contextmanager
def timer(name, debug=True):
    start = time()
    yield
    if debug:
        diff = time() - start
        print("{} costs {:.4f} ms".format(name, diff * 1000))


class HardcodePlayer2(Player):
    def __init__(self, name, pnr, **kwargs):

        # Basic Info
        self.name = name
        self.pnr = pnr
        self.turn = 0
        self.debug = False
        self.timer = False
        self.return_value = True
        self.value_wrap = True
        self.action_classes = ["_execute", "_hint", "_discard"]
        self.discard_internal = False

        # TOFIX Hardcoded for 2 players
        self.partner_nr = 1 - self.pnr
        self.card_nr = 5

        # Records
        self.last_state = None
        self.knowledge = [
            [cgf.COUNTS.copy() for _ in range(5)] for _ in range(self.card_nr)
        ]

        self.index_play = []
        self.index_play_candidate = []
        self.index_discard = []
        self.index_discard_candidate = []
        self.index_protect = []

        self.partner_play = []
        self.partner_play_candidate = []
        self.partner_discard = []
        self.partner_protect = []

        # Parameters

        # Decision pattern matches
        # [(func : Player -> bool, action : Action Type)]
        self.decision_protocol = [
            (lambda p, s, m: p.index_play != [] and p.partner_play != [], "_execute"),
            (lambda p, s, m: p.index_play != [] and s.get_num_hints() > 1, "_execute"),
            (lambda p, s, m: p.partner_play == [] and s.get_num_hints() > 0, "_hint"),
            (lambda p, s, m: p.index_play == [], "_hint"),
            (lambda p, s, m: p.index_discard != [], "_discard"),
            (
                lambda p, s, m: p.index_play or p.index_play_candidate,
                "_execute_force",
            ),
            (lambda p, s, m: True, "_execute"),
        ]
        self.decision_permutation = 0
        self.risk_play = {0: 0.5, 5: 0, 12: 0.4, 30: 1}
        self.hint_to_protect = False  # not used
        self.self_card_count = False
        self.self_play_order = "newest"
        self.self_discard_order = "oldest"
        self.self_hint_order = "newest"
        self.partner_card_count = False
        self.partner_play_order = "newest"
        self.partner_samples = 10

        self.settings_score_badplay = -2
        self.settings_score_verybadplay = -2
        self.settings_score_playable = 0.8
        self.settings_score_badplaycandidate = -0.1

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.debug:
            pprint(self.__dict__)

        self._convert_parameter()

    def _convert_parameter(self):

        with timer("coverting parameters", self.timer):
            order = range(len(self.decision_protocol))
            D = permutations(order)
            for i in range(self.decision_permutation):
                try:
                    order = next(D)
                except StopIteration:
                    pass
            self.decision_protocol = [self.decision_protocol[i] for i in order]
            self.risk_play = {int(k): v for k, v in self.risk_play.items()}
            if self.debug:
                print("pattern match order: ", order)
            orders = {
                "newest": lambda x: x,
                "oldest": lambda x: -x,
                "random": lambda x: random(),
            }
            for attr in dir(self):
                if attr.endswith("order"):
                    value = getattr(self, attr)
                    if isinstance(value, str):
                        setattr(self, attr, orders[getattr(self, attr)])

    def inform(self, action, player, new_state, new_model):

        self._update_state(new_state, new_model)

        if self.turn == 0:
            if action.type == cgf.HINT_COLOR:
                hinted_indices = new_state.get_hinted_indices()
                if hinted_indices != []:
                    self.index_play.append(hinted_indices[0])
                    self.index_discard.extend(hinted_indices[1:])
            elif action.type == cgf.HINT_NUMBER:
                if action.num == 1:
                    self.index_play.extend(new_state.get_hinted_indices())
            if self.debug:
                pprint(self.__dict__)
                print("\n\n\n")
            return

        if self.self_card_count and action.type in [cgf.DISCARD, cgf.PLAY]:
            partner_hand = new_state.get_hands()[self.partner_nr]
            if len(partner_hand) == 5:
                col, num = partner_hand[-1]
                for k in self.knowledge:
                    k[col][num - 1] = max(0, k[col][num - 1] - 1)

        if action.pnr != self.pnr:
            return

        board = new_state.get_board()
        trash = new_state.get_trash()
        knowledge = self.knowledge

        if action.type in [cgf.HINT_COLOR, cgf.HINT_NUMBER]:

            with timer("interpret hints", self.timer):

                if self.discard_internal:
                    hinted_indices = list(range(self.card_nr))
                    self.index_play = []
                    self.index_play_candidate = []
                    self.index_discard = []
                    self.index_protect = []
                else:
                    hinted_indices = sorted(new_state.get_hinted_indices())

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
                )

                if self.debug:
                    pprint(self.__dict__)
                    print("\n\n\n")

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
    ):

        play = cur_play.copy()
        play_candidate = cur_play_candidate.copy()
        discard = cur_discard.copy()
        protect = cur_protect.copy()
        playable_pct = {
            idx: cpf.slot_playable_pct(card, board)
            for idx, card in enumerate(knowledge)
        }
        discardable_pct = {
            idx: cpf.slot_discardable_pct(card, board, trash)
            for idx, card in enumerate(knowledge)
        }

        with timer("interpret main", self.timer):

            flag = False

            for idx in hinted_indices:
                card = knowledge[idx]
                if playable_pct[idx] > 0.8:
                    play.append(idx)
                    flag = True
            for i, k in enumerate(knowledge):
                need_protect = True
                for col, num in cpf.get_possible(k):
                    if board[col][1] >= num:
                        need_protect = False
                        break
                    else:
                        for n in range(board[col][1] + 1, num):
                            if trash.count((col, n)) == cgf.COUNTS[n - 1]:
                                need_protect = False
                                break
                if need_protect:
                    self.index_protect.append(i)

            if not flag and hinted_indices != []:
                newest = max(hinted_indices, key=self.self_hint_order)
                card = knowledge[newest]
                if playable_pct[idx] > 0.1:
                    play.append(newest)
                    for idx in hinted_indices:
                        if idx == newest:
                            continue
                        card = knowledge[idx]
                        if playable_pct[idx] > 0.1:
                            play_candidate.append(idx)
                elif discardable_pct[idx] > 0.9:
                    discard.append(idx)
                else:
                    protect.append(newest)

            for i, card in enumerate(knowledge):
                if playable_pct[i] > 0.98:
                    self.index_play.append(i)
                elif discardable_pct[i] > 0.98:
                    self.index_discard.append(i)

        with timer("interpret postprocess", self.timer):
            play = [
                i for i in range(len(knowledge)) if i in play and playable_pct[i] > 0.02
            ]
            play_candidate = [
                i
                for i in range(len(knowledge))
                if i in play_candidate and playable_pct[i] > 0.02
            ]
            discard = [
                i
                for i in range(len(knowledge))
                if i in discard and discardable_pct[i] > 0.02
            ]
            protect = [
                i
                for i in range(len(knowledge))
                if i in protect and discardable_pct[i] < 0.5
            ]

        return play, play_candidate, discard, protect

    def get_action(self, state, model):
        def _wrapper(value_dict, best=None):
            if self.return_value and self.value_wrap:
                if not best:
                    best = max(value_dict.keys(), key=lambda k: value_dict[k])
                if self.debug:
                    print("value_wrap enabled:")
                    print("original value dict:")
                    for k, v in sorted(value_dict.items(), key=lambda x: -x[1]):
                        print("    ", str(k), "has value: ", v)
                    print("wrapped to:")
                    print("    ", str(best))
                if value_dict[best] == -1:
                    return Action(cgf.DISCARD, pnr=self.pnr, cnr=0)
                return best
            else:
                return value_dict

        self.turn += 1

        # Always hint the smallest number in the first turn
        if self.turn == 1 and self.last_state is None:
            partner = state.get_hands()[self.partner_nr]
            min_num = min([x[1] for x in partner])

            if self.debug:
                print("first turn")

            action = Action(cgf.HINT_NUMBER, self.partner_nr, num=min_num)
            self._update_state(state, model)
            if self.return_value:
                value_dict = {}
                for i in range(5):
                    value_dict[Action(cgf.PLAY, cnr=i)] = 0
                    value_dict[Action(cgf.DISCARD, cnr=i)] = 0
                    value_dict[Action(cgf.HINT_NUMBER, num=i)] = 0
                    value_dict[Action(cgf.HINT_COLOR, col=i)] = 0
                value_dict[action] = 1
                return _wrapper(value_dict)
            else:
                return action

        self._update_state(state, model)

        # Pattern matcing [self._decide() in version 1]
        chosen_action = None
        chosen_action_name = None
        for i, (func, action) in enumerate(self.decision_protocol):
            if func(self, state, model):
                force = False
                if action.endswith("_force"):
                    force = True
                    action = action[:-6]
                chosen_action = getattr(self, action)(force=force)
                if self.debug:
                    print("executing pattern ", i)
                    print(chosen_action)
                chosen_action_name = action
                break

        # Default action
        if chosen_action is None:
            if self.debug:
                print("Using Default action!")
            chosen_action_name = self.action_classes[0]
            chosen_action = getattr(self, chosen_action_name)

        if self.return_value:
            final_action = max(chosen_action.keys(), key=lambda a: chosen_action[a])
        else:
            final_action = chosen_action

        if self.return_value:
            value_dict = chosen_action
            for cls in self.action_classes:
                if cls != chosen_action_name:
                    for k, v in getattr(self, cls)(force=True).items():
                        value_dict[k] = max(-1, v - 0.1)
            return _wrapper(value_dict, final_action)
        else:
            return chosen_action

    def _execute(self, force=False):
        with timer("execute main", self.timer):
            board = self.last_state.get_board()
            knowledge = self.knowledge

            if self.return_value:
                order = []

            self.index_play.sort(key=self.self_play_order)
            while self.index_play != []:
                idx = self.index_play.pop()
                card = knowledge[idx]
                if cpf.slot_playable_pct(card, board) > 0:
                    action = Action(cgf.PLAY, cnr=idx)
                    if self.return_value:
                        order.append(action)
                    else:
                        return action
                else:
                    self.index_discard.append(idx)

        # play at risk
        with timer("execute at risk", self.timer):
            risk_threshold = 0
            for turn, thresh in sorted(self.risk_play.items()):
                if turn >= self.turn:
                    risk_threshold = 1 - thresh
                    break
            idx = None
            max_pct = 0
            self.index_play_candidate.sort(key=self.self_play_order)
            for i in self.index_play_candidate[::-1]:
                card = knowledge[i]
                if cpf.slot_playable_pct(card, board) > max_pct:
                    idx = i
                    max_pct = cpf.slot_playable_pct(card, board)
            if idx and max_pct > risk_threshold:
                action = Action(cgf.PLAY, cnr=idx)
                if self.return_value:
                    order.append(action)
                else:
                    return action

        # TODO add more conditions for more aggressive play

        with timer("execute postprocess", self.timer):
            if (
                (self.return_value and not order) or not self.return_value
            ) and not force:
                if self.last_state.get_num_hints() > 1:
                    if self.debug:
                        print("redirected from execute to hint")
                    return self._hint()
                else:
                    if self.debug:
                        print("redirected from execute to discard")
                    return self._discard(force=True)

            if force:
                self.index_play_candidate.sort()
                while self.index_play_candidate != []:
                    idx = self.index_play_candidate.pop()
                    card = knowledge[idx]
                    if cpf.slot_playable_pct(card, board) > 0.02:
                        action = Action(cgf.PLAY, cnr=idx)
                        if self.return_value:
                            order.append(action)
                        else:
                            return action
                    else:
                        self.index_discard.append(idx)
                action = Action(
                    cgf.PLAY, cnr=max(range(self.card_nr), key=self.self_play_order)
                )
                if self.return_value:
                    order.append(action)
                else:
                    return action

            # only reachable for self.return_value
            value_dict = {}
            for i, action in enumerate(order):
                value_dict[action] = 1 - 0.1 * i
            for i in range(len(self.knowledge)):
                value_dict.setdefault(Action(cgf.PLAY, cnr=i), -1)
            return value_dict

    def _discard(self, force=False):

        if self.return_value:
            order = []

        if self.index_discard != []:
            action = Action(
                cgf.DISCARD, cnr=max(self.index_discard, key=self.self_discard_order)
            )
            if self.return_value:
                order.append(action)
            else:
                return action

        if self.index_discard_candidate != []:
            action = Action(
                cgf.DISCARD,
                cnr=max(self.index_discard_candidate, key=self.self_discard_order),
            )
            if self.return_value:
                order.append(action)
            else:
                return action

        for i in sorted(range(self.card_nr), key=self.self_discard_order, reverse=True):
            if i not in self.index_protect:
                action = Action(cgf.DISCARD, cnr=i)
                if self.return_value:
                    order.append(action)
                else:
                    return action

        if self.return_value:
            if order:
                value_dict = {}
                for i, action in enumerate(order):
                    value_dict[action] = 1 - 0.1 * i
                for i in range(self.card_nr):
                    A = Action(cgf.DISCARD, cnr=i)
                    if A not in value_dict.keys():
                        value_dict[A] = -1
            else:
                value_dict = {
                    Action(cgf.DISCARD, cnr=i): 0 for i in range(self.card_nr)
                }
                value_dict[
                    Action(
                        cgf.DISCARD,
                        cnr=max(range(self.card_nr), key=self.self_discard_order),
                    )
                ] = 1
            return value_dict

        else:
            return Action(cgf.DISCARD, cnr=0)

    def _evaluate_partner(
        self, hands, predicted_play, predicted_play_candidate, predicted_discard
    ):

        with timer("evaluate partner", self.timer):
            board = self.last_state.get_board()

            score = 0
            for i in predicted_play:
                if cpf.card_playable(hands[i], board):
                    score += 1
                else:
                    score += self.settings_score_badplay
            if predicted_play != [] and (
                not cpf.card_playable(
                    hands[max(predicted_play, key=self.partner_play_order)], board
                )
            ):
                score += self.settings_score_verybadplay

            for i in predicted_play_candidate:
                if cpf.card_playable(hands[i], board):
                    score += self.settings_score_playable
                else:
                    score += self.settings_score_badplaycandidate

        return score

    def _hint(self, force=False):

        if self.last_state.get_num_hints() < 1:
            assert (not force) or self.return_value, "force hinting when no hints left"
            if self.return_value:
                return {
                    Action(t, pnr=self.partner_nr, col=i, num=i): -1
                    for i, t in product(range(5), [cgf.HINT_NUMBER, cgf.HINT_COLOR])
                }
            else:
                self._discard(force=True)

        partner_hand = self.last_state.get_hands()[self.partner_nr]
        partner_knowledge = deepcopy(
            self.last_state.get_all_knowledge()[self.partner_nr]
        )

        if self.partner_card_count:
            with timer("processs partner card count", self.timer):
                possible_self_hands = list(
                    product(*[cpf.get_possible(k) for k in self.knowledge])
                )

                for (col, num) in self.last_state.get_common_visible_cards():
                    for k in partner_knowledge:
                        k[col][num - 1] = max(0, k[col][num - 1] - 1)
                possible_knowledges = [
                    deepcopy(partner_knowledge)
                    for _ in range(min(len(possible_self_hands), self.partner_samples))
                ]
                possible_self_hands = sample(
                    possible_self_hands,
                    min(len(possible_self_hands), self.partner_samples),
                )

                if self.debug:
                    print("partner knowledge before card count")
                    pprint(partner_knowledge)
                    print("sampling {} possible cases".format(len(possible_knowledges)))

                # update knowledge w.r.t possible hands
                for knowledge, self_hands in zip(
                    possible_knowledges, possible_self_hands
                ):
                    for (col, num) in list(self_hands):
                        for k in knowledge:
                            k[col][num - 1] = max(0, k[col][num - 1] - 1)

                if self.debug:
                    print("\n partner knowledge after card count")
                    pprint(possible_knowledges)
                    print("")
        else:
            possible_knowledges = [partner_knowledge]

        board = self.last_state.get_board()
        trash = self.last_state.get_trash()

        baseline = max_score = self._evaluate_partner(
            partner_hand,
            self.partner_play,
            self.partner_play_candidate,
            self.partner_discard,
        )
        best_action = None
        some_score = -100
        some_action = None
        if self.return_value:
            value_dict = {}

        if self.debug:
            print("comparing hints")

        for i, t in product(range(5), [cgf.HINT_NUMBER, cgf.HINT_COLOR]):

            action = Action(t, pnr=self.partner_nr, col=i, num=i)

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

            scores = []
            for knowledge in possible_knowledges:
                try:
                    (
                        pred_play,
                        pred_play_candidate,
                        pred_discard,
                        pred_protect,
                    ) = self._interpret(
                        hinted,
                        knowledge,
                        board,
                        trash,
                        pred_play,
                        pred_play_candidate,
                        pred_discard,
                        pred_protect,
                    )
                    scores.append(
                        self._evaluate_partner(
                            partner_hand,
                            pred_play,
                            pred_play_candidate,
                            pred_discard,
                        )
                    )
                except ZeroDivisionError:
                    pass

            # TOFIX
            if scores != []:
                score = sum(scores) / len(scores)
            else:
                score = baseline

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

            if score > some_score:
                some_score = score
                some_action = action

            if self.return_value:
                value_dict[action] = score

        if best_action is None:
            if force:
                if not self.return_value:
                    return some_action
            else:
                if self.debug:
                    print("redirected from hint to discard")
                return self._discard(force=True)

        if self.return_value:
            if value_dict:
                # normalization
                score_max = max(value_dict.values())
                score_min = min(value_dict.values())
                if score_max - score_min < 0.05:
                    if score_max > baseline:
                        return {k: 1 for k in value_dict.keys()}
                    else:
                        return {k: -1 for k in value_dict.keys()}
                else:
                    if score_max > baseline:
                        scale_pos = score_max - baseline
                    else:
                        scale_pos = 1
                    if score_min < baseline:
                        scale_neg = baseline - score_min
                    else:
                        scale_neg = 1

                    return {
                        k: v / (scale_pos if v > baseline else scale_neg)
                        for k, v in value_dict.items()
                    }
            else:
                return {}

        else:
            return best_action

    def _update_index(self, idx, partner=False):

        with timer("update index", self.timer):
            if not partner:
                prefix = "index_"
                del self.knowledge[idx]
                new_knowledge = [cgf.COUNTS.copy() for _ in range(5)]
                if self.self_card_count:
                    visible_cards = (
                        self.last_state.get_common_visible_cards()
                        + self.last_state.get_hands()[self.partner_nr]  # noqa E503
                    )
                    for col, num in visible_cards:
                        new_knowledge[col][num - 1] -= 1

                self.knowledge.append(new_knowledge)
            else:
                prefix = "partner_"

            for attr in dir(self):
                if attr.startswith(prefix):
                    L = getattr(self, attr)
                    if not isinstance(L, list):
                        continue
                    setattr(
                        self, attr, [x if x < idx else x - 1 for x in L if x != idx]
                    )

    def _update_state(self, new_state, new_model):

        with timer("update state", self.timer):
            self.last_state = new_state
            new_knowledge = new_model.get_knowledge()
            visible_cards = (
                self.last_state.get_common_visible_cards()
                + self.last_state.get_hands()[self.partner_nr]  # noqa W503
            )
            knowledge_mask = [cgf.COUNTS.copy() for _ in range(5)]
            for col, num in visible_cards:
                knowledge_mask[col][num - 1] -= 1
            for i, k in enumerate(new_knowledge):
                for col in range(5):
                    for num in range(5):
                        self.knowledge[i][col][num] = min(
                            k[col][num], knowledge_mask[col][num]
                        )

    def set_from_key(self, key=0):
        print("using key:", key)

        """search for evaluation parameter
        pos_badplay = [-2, -1.5, -1, -0.5]
        pos_verybadplay = [-3, -2, -1]
        pos_discardable = [1, 0.8, 0.5, 0.2, 0]
        pos_playable = [0.8, 0.5, 0.2, 0]
        pos_badplaycandidate = [0, -0.1, -0.3]
        tot = 4 * 3 * 5 * 4 * 3
        key = key % tot
        self.settings_score_badplay = pos_badplay[key % 4]
        key = key // 4
        self.settings_score_verybadplay = pos_verybadplay[key % 3]
        key = key // 3
        self.settings_score_discardable = pos_discardable[key % 5]
        key = key // 5
        self.settings_score_playable = pos_playable[key % 4]
        key = key // 4
        self.settings_score_badplaycandidate = pos_badplaycandidate[key % 3]
        """

        pos_risk = [
            {5: 0, 12: 0.4, 30: 1},
            {5: 0.5, 12: 0, 30: 1},
            {10: 0.6, 30: 0},
            {30: 1},
            {30: 0},
        ]
        pos_order = ["newest", "oldest", "random"]
        tot = 5040 * (3 ** 3) * 2 * 5
        key = key % tot
        self.risk_play = pos_risk[key % 5]
        key = key // 5
        self.self_card_count = self.partner_card_count = bool(key % 2)
        key = key // 2
        self.self_play_order = self.partner_play_order = pos_order[key % 3]
        key = key // 3
        self.self_discard_order = pos_order[key % 3]
        key = key // 3
        self.self_hint_order = pos_order[key % 3]
        key = key // 3
        self.decision_permutation = key
        self._convert_parameter()

    def snapshot(self, data_file=None):
        if data_file:
            pickle.dump(
                {
                    k: v
                    for k, v in self.__dict__.items()
                    if k != "decision_protocol" and not k.endswith("order")
                },
                open(data_file, "wb"),
            )
