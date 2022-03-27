import csv
import os
import pickle
from common_game_functions import *
from Agents.common_player_functions import *
from Agents.player import Action


def format_card(colnum):
    col, num = colnum
    return COLORNAMES[col] + " " + str(num)


def format_hand(hand):
    return ", ".join(map(format_card, hand))


class Game(object):
    def __init__(
        self,
        players,
        data_file,
        pickle_file=None,
        format=0,
        http_player=-1,
        print_game=True,
    ):
        self.players = players
        self.hits = 3
        self.hints = 8
        self.current_player = 0
        self.board = [(c, 0) for c in ALL_COLORS]
        self.played = []
        self.deck = make_deck()
        self.extra_turns = 0
        self.hands = []
        self.knowledge = []
        self.make_hands()
        self.trash = []
        self.turn = 1
        self.format = format
        self.pickle_file = pickle_file
        self.save_snapshots = None
        if data_file.endswith(".csv"):
            self.data_format = "csv"
            self.data_file = open(data_file, "a+")
            self.data_writer = csv.writer(self.data_file, delimiter=",")
        elif data_file.endswith(".pkl"):
            self.data_format = "pkl"
            self.data_file = open(data_file, "ab+")
            pickle.dump([], self.data_file)
        elif os.path.isdir(data_file):
            self.data_format = "pkl"
            self.data_file = open(os.path.join(data_file, "game.pkl"), "ab+")
            pickle.dump([], self.data_file)
            self.save_snapshots = data_file
        else:
            print("Unsupported data file format!")
            raise NotImplementedError
        self.hint_log = dict([(a, []) for a in range(len(players))])
        self.action_log = dict([(a, []) for a in range(len(players))])
        self.http_player = http_player
        self.print_game = print_game

        if self.format:
            self._print(self.deck)

    def _print(self, *args):
        if self.print_game:
            print(*args)

    # returns blank array for player_nr's own hand if not httpui
    # everything is guaranteed to be a copy
    def _make_game_state(self, player_nr, hinted_indices=[], card_changed=None):
        hands = []

        for i, h in enumerate(self.hands):
            if i == player_nr and i != self.http_player:
                hands.append([])
            else:
                hands.append(deepcopy(h))

        return GameState(
            self.current_player,
            hands,
            deepcopy(self.trash),
            deepcopy(self.played),
            deepcopy(self.board),
            self.hits,
            self.valid_actions(),
            self.hints,
            deepcopy(self.knowledge),
            deepcopy(hinted_indices),
            deepcopy(card_changed),
        )

    # everything is guaranteed to be a copy
    def _make_player_model(self, player_nr):
        return BasePlayerModel(
            player_nr,
            deepcopy(self.knowledge[player_nr]),
            deepcopy(self.hint_log[player_nr]),
            deepcopy(self.action_log),
        )

    def make_hands(self):
        handsize = 4

        if len(self.players) < 4:
            handsize = 5

        for i, p in enumerate(self.players):
            self.hands.append([])
            self.knowledge.append([])
            for j in range(handsize):
                self.draw_card(i)

    def draw_card(self, pnr=None):
        if pnr is None:
            pnr = self.current_player

        if not self.deck:
            return

        self.hands[pnr].append(self.deck.pop())
        self.knowledge[pnr].append(initial_knowledge())

    def perform(self, action):
        hint_indices = []
        card_changed = None
        if format:
            self._print(
                "\nMOVE:",
                self.current_player,
                action.type,
                action.cnr,
                action.pnr,
                action.col,
                action.num,
            )
        self.action_log[self.current_player].append(action)

        if action.type == HINT_COLOR:
            self.hints -= 1
            self._print(
                self.players[self.current_player].name,
                "hints",
                self.players[action.pnr].name,
                "about all their",
                COLORNAMES[action.col],
                "cards",
                "hints remaining:",
                self.hints,
            )
            self._print(
                self.players[action.pnr].name,
                "has",
                format_hand(self.hands[action.pnr]),
            )
            self.hint_log[action.pnr].append((self.current_player, action))
            slot_index = 0

            for (col, num), knowledge in zip(
                self.hands[action.pnr], self.knowledge[action.pnr]
            ):
                if col == action.col:
                    hint_indices.append(slot_index)
                    for i, k in enumerate(knowledge):
                        if i != col:
                            for i in range(len(k)):
                                k[i] = 0
                else:
                    for i in range(len(knowledge[action.col])):
                        knowledge[action.col][i] = 0

                slot_index += 1

        elif action.type == HINT_NUMBER:
            self.hints -= 1
            self._print(
                self.players[self.current_player].name,
                "hints",
                self.players[action.pnr].name,
                "about all their",
                action.num,
                "hints remaining:",
                self.hints,
            )
            self._print(
                self.players[action.pnr].name,
                "has",
                format_hand(self.hands[action.pnr]),
            )
            self.hint_log[action.pnr].append((self.current_player, action))
            slot_index = 0

            for (col, num), knowledge in zip(
                self.hands[action.pnr], self.knowledge[action.pnr]
            ):
                if num == action.num:
                    hint_indices.append(slot_index)
                    for k in knowledge:
                        for i in range(len(COUNTS)):
                            if i + 1 != num:
                                k[i] = 0
                else:
                    for k in knowledge:
                        k[action.num - 1] = 0

                slot_index += 1

        elif action.type == PLAY:
            (col, num) = self.hands[self.current_player][action.cnr]
            card_changed = (col, num)
            self._print(
                self.players[self.current_player].name,
                "plays",
                format_card((col, num)),
            )

            if self.board[col][1] == num - 1:
                self.board[col] = (col, num)
                self.played.append((col, num))

                if num == 5:
                    self.hints += 1
                    self.hints = min(self.hints, 8)

                self._print("successfully! Board is now", format_hand(self.board))
            else:
                self.trash.append((col, num))
                self.hits -= 1
                self._print("and fails. Board was", format_hand(self.board))

            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self.draw_card()
            self._print(
                self.players[self.current_player].name,
                "now has",
                format_hand(self.hands[self.current_player]),
            )
        else:
            self.hints += 1
            self.hints = min(self.hints, 8)
            card_changed = self.hands[self.current_player][action.cnr]
            self.trash.append(card_changed)
            self._print(
                self.players[self.current_player].name,
                "discards",
                format_card(self.hands[self.current_player][action.cnr]),
            )
            self._print("trash is now", format_hand(self.trash))
            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self.draw_card()
            self._print(
                self.players[self.current_player].name,
                "now has",
                format_hand(self.hands[self.current_player]),
            )

        return hint_indices, card_changed

    def valid_actions(self):
        valid = []

        for i in range(len(self.hands[self.current_player])):
            valid.append(Action(PLAY, cnr=i))
            valid.append(Action(DISCARD, cnr=i))

        if self.hints > 0:
            for i, p in enumerate(self.players):
                if i != self.current_player:
                    for col in set(map(lambda colnum: colnum[0], self.hands[i])):
                        valid.append(Action(HINT_COLOR, pnr=i, col=col))

                    for num in set(map(lambda colnum: colnum[1], self.hands[i])):
                        valid.append(Action(HINT_NUMBER, pnr=i, num=num))

        return valid

    def run(self, turns=-1):
        self.turn = 1
        while (not self.done()) and (turns < 0 or self.turn < turns):
            self.turn += 1
            self.single_turn()
        self._print("Game done, hits left:", self.hits)
        points = self.score()
        self._print("Points:", points)
        self._print("Board:", self.board)
        self._print("Hands:", self.hands)
        self.data_file.close()
        return points

    def score(self):
        return sum(map(lambda colnum: colnum[1], self.board))

    # everything is guaranteed to be a copy
    def _make_partner_knowledge_model(self, game_state):
        partner_knowledge_model = {}
        for possible_action in game_state.get_valid_actions():
            if possible_action.type in [HINT_COLOR, HINT_NUMBER]:
                partner_knowledge_model[possible_action] = apply_hint_to_knowledge(
                    possible_action, self.hands, self.knowledge
                )
        return partner_knowledge_model

    def single_turn(self):
        game_state = self._make_game_state(self.current_player)
        player_model = self._make_player_model(self.current_player)
        partner_knowledge_model = self._make_partner_knowledge_model(game_state)
        if hasattr(self.players[self.current_player], "is_behavior_clone"):
            action = self.players[self.current_player].get_action(
                game_state, player_model, partner_knowledge_model
            )
        else:
            action = self.players[self.current_player].get_action(
                game_state, player_model
            )
        if isinstance(action, tuple):  # workaround for experimental player
            action = action[0]

        # Data collection
        if self.pickle_file:
            pickle.dump(["Action", game_state, player_model, action], self.pickle_file)

        # Process action
        self.external_turn(action, partner_knowledge_model)

    def external_turn(self, action, partner_knowledge_model=None):
        if partner_knowledge_model is None:
            game_state = self._make_game_state(self.current_player)
            partner_knowledge_model = self._make_partner_knowledge_model(game_state)
        if not self.done():
            if not self.deck:
                self.extra_turns += 1

            if self.data_format == "csv":
                self.data_writer.writerow(
                    [
                        self.current_player,
                        action.type,
                        self.board,
                        self.trash,
                        self.hints,
                        self.knowledge[self.current_player],
                    ]
                )
            elif self.data_format == "pkl":
                trash = [[0] * 5 for _ in range(5)]
                for (col, num) in self.trash:
                    trash[col][num - 1] += 1
                partner_nr = 1 - self.current_player
                try:
                    last_action = self.action_log[partner_nr][-1]
                except IndexError:
                    last_action = None
                extra = []
                for k in self.knowledge[partner_nr]:
                    extra.append(int_slot_playable_pct(k, self.board))
                while len(extra) < 5:
                    extra.append(0)
                for k in self.knowledge[self.current_player]:
                    extra.append(int_slot_playable_pct(k, self.board))
                while len(extra) < 10:
                    extra.append(0)
                for k in self.knowledge[partner_nr]:
                    extra.append(int_slot_discardable_pct(k, self.board, self.trash))
                while len(extra) < 15:
                    extra.append(0)
                for k in self.knowledge[self.current_player]:
                    extra.append(int_slot_discardable_pct(k, self.board, self.trash))
                while len(extra) < 20:
                    extra.append(0)
                pickle.dump(
                    encode_state(  # noqa F405
                        self.hands[partner_nr],
                        self.knowledge[partner_nr],
                        self.knowledge[self.current_player],
                        self.board,
                        self.trash,
                        self.hits,
                        self.hints,
                        last_action,
                        action,
                        partner_knowledge_model,
                        self.current_player,
                        extra,
                    ),
                    self.data_file,
                )

            hint_indices, card_changed = self.perform(action)

            for p in self.players:
                game_state = self._make_game_state(
                    p.get_nr(), hint_indices, card_changed
                )
                player_model = self._make_player_model(p.get_nr())

                p.inform(
                    action,
                    self.current_player,
                    deepcopy(game_state),
                    deepcopy(player_model),
                )

                # Data collection
                if self.pickle_file:
                    pickle.dump(
                        [
                            "Inform",
                            game_state,
                            player_model,
                            action,
                            p.get_nr(),
                            self.current_player,
                        ],
                        self.pickle_file,
                    )

            self.current_player += 1
            self.current_player %= len(self.players)

    def done(self):
        if self.extra_turns == len(self.players) or self.hits == 0:
            return True

        for (col, num) in self.board:
            if num != 5:
                return False

        return True

    def finish(self):
        if self.save_snapshots:
            for i, p in enumerate(self.players):
                p.snapshot(os.path.join(self.save_snapshots, "{}_{}.pkl".format(p.name, i)))

        if self.format:
            self._print("Score", self.score())


class NullStream(object):
    def write(self, *args):
        pass
