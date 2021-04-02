import random
import sys
import copy
import time
import csv
from common_game_functions import *
from Agents.player import Action

# comment this line out when running multithreaded tests
# random.seed(0)  # for reproducing results


def format_card(colnum):
    col, num = colnum
    return COLORNAMES[col] + " " + str(num)


def format_hand(hand):
    return ", ".join(map(format_card, hand))


class BasePlayerModel(object):
    def __init__(self, nr, knowledge, hints, actions):
        self.nr = nr
        self.knowledge = knowledge  # This is the knowledge matrix based only on updates in game engine
        self.hints = hints  # These are the hints that this player has received (Format: List of (P,Hint) if recieved from player P)
        self.actions = actions  # These are the actions taken by all players in the past (Format: Dictionary with player as keys and actions as values)

    def get_hints(self):
        return self.hints

    def get_knowledge(self):
        return self.knowledge

    def get_actions(self):
        return self.actions

    def get_hints_from_player(self, p):
        filtered_hints = []

        for player, hint in self.hints:
            if p == player:
                filtered_hints.append(hint)

        return filtered_hints


class GameState(object):
    def __init__(
        self,
        current_player,
        hands,
        trash,
        played,
        board,
        valid_actions,
        num_hints,
        all_knowledge,
        hinted_indices=[],
        card_changed=None,
    ):
        self.current_player = current_player
        self.hands = hands
        self.trash = trash
        self.played = played
        self.board = board
        self.valid_actions = valid_actions
        self.num_hints = num_hints
        self.all_knowledge = all_knowledge
        self.hinted_indices = hinted_indices
        self.card_changed = card_changed

    def get_current_player(self):
        return self.current_player

    def get_hands(self):
        return self.hands

    def get_trash(self):
        return self.trash

    def get_played(self):
        return self.played

    def get_board(self):
        return self.board

    def get_valid_actions(self):
        return self.valid_actions

    def get_num_hints(self):
        return self.num_hints

    def get_hinted_indices(self):
        return self.hinted_indices

    def get_card_changed(self):
        return self.card_changed

    def get_all_knowledge(self):
        return self.all_knowledge


class Game(object):
    def __init__(self, players, data_file, format=0, http_player=-1):
        self.players = players
        self.hits = 3
        self.hints = 8
        self.current_player = 0
        self.board = list(map(lambda c: (c, 0), ALL_COLORS))
        self.played = []
        self.deck = make_deck()
        self.extra_turns = 0
        self.hands = []
        self.knowledge = []
        self.make_hands()
        self.trash = []
        self.turn = 1
        self.format = format
        self.dopostsurvey = False
        self.study = False
        self.data_file = open(data_file, "a")
        self.data_writer = csv.writer(self.data_file, delimiter=",")
        self.hint_log = dict([(a, []) for a in range(len(players))])
        self.action_log = dict([(a, []) for a in range(len(players))])
        self.http_player = http_player
        if self.format:
            print(self.deck)

    # returns blank array for player_nr's own hand if not httpui
    def _make_game_state(self, player_nr, hinted_indices=[], card_changed=None):
        hands = []
        for i, h in enumerate(self.hands):
            if i == player_nr and i != self.http_player:
                hands.append([])
            else:
                hands.append(h)
        return GameState(
            self.current_player,
            hands,
            self.trash,
            self.played,
            self.board,
            self.valid_actions(),
            self.hints,
            self.knowledge,
            hinted_indices,
            card_changed,
        )

    def _make_player_model(self, player_nr):
        return BasePlayerModel(
            player_nr,
            self.knowledge[player_nr],
            self.hint_log[player_nr],
            self.action_log,
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
        self.hands[pnr].append(self.deck[0])
        self.knowledge[pnr].append(initial_knowledge())
        del self.deck[0]

    def perform(self, action):
        hint_indices = []
        card_changed = None
        if format:
            print(
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
            print(
                self.players[self.current_player].name,
                "hints",
                self.players[action.pnr].name,
                "about all their",
                COLORNAMES[action.col],
                "cards",
                "hints remaining:",
                self.hints,
            )
            print(
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
            print(
                self.players[self.current_player].name,
                "hints",
                self.players[action.pnr].name,
                "about all their",
                action.num,
                "hints remaining:",
                self.hints,
            )
            print(
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
            print(
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
                print("successfully! Board is now", format_hand(self.board))
            else:
                self.trash.append((col, num))
                self.hits -= 1
                print("and fails. Board was", format_hand(self.board))
            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self.draw_card()
            print(
                self.players[self.current_player].name,
                "now has",
                format_hand(self.hands[self.current_player]),
            )
        else:
            self.hints += 1
            self.hints = min(self.hints, 8)
            card_changed = self.hands[self.current_player][action.cnr]
            self.trash.append(card_changed)
            print(
                self.players[self.current_player].name,
                "discards",
                format_card(self.hands[self.current_player][action.cnr]),
            )
            print("trash is now", format_hand(self.trash))
            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self.draw_card()
            print(
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
        while not self.done() and (turns < 0 or self.turn < turns):
            self.turn += 1
            self.single_turn()
        print("Game done, hits left:", self.hits)
        points = self.score()
        print("Points:", points)
        self.data_file.close()
        return points

    def score(self):
        return sum(map(lambda colnum: colnum[1], self.board))

    def single_turn(self):
        self.external_turn(
            self.players[self.current_player].get_action(
                self._make_game_state(self.current_player),
                self._make_player_model(self.current_player),
            )
        )

    def external_turn(self, action):
        if not self.done():
            if not self.deck:
                self.extra_turns += 1
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
            hint_indices, card_changed = self.perform(action)
            for p in self.players:
                p.inform(
                    action,
                    self.current_player,
                    self._make_game_state(p.get_nr(), hint_indices, card_changed),
                    self._make_player_model(p.get_nr()),
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
        if self.format:
            print("Score", self.score())


class NullStream(object):
    def write(self, *args):
        pass
