import random
import sys
import copy
import time
import csv
import pickle
from common_game_functions import *
from Agents.player import Action

# comment this line out when running multithreaded tests
# random.seed(0)  # for reproducing results

def format_card(colnum):
    col, num = colnum
    return COLORNAMES[col] + " " + str(num)


def format_hand(hand):
    return ", ".join(map(format_card, hand))


class Game(object):
    def __init__(self, players, data_file, pickle_file=None, format=0, http_player=-1):
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
        self.pickle_file = pickle_file
        self.data_writer = csv.writer(self.data_file, delimiter=",")
        self.hint_log = dict([(a, []) for a in range(len(players))])
        self.action_log = dict([(a, []) for a in range(len(players))])
        self.http_player = http_player

        if self.format:
            print(self.deck)

        if http_player != -1:
            for i, player in enumerate(self.players):
                if i != http_player and hasattr(player, "debug"):
                    player.debug = True

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
        game_state = self._make_game_state(self.current_player)
        player_model = self._make_player_model(self.current_player)
        action = self.players[self.current_player].get_action(game_state, player_model)
        
        # Process action
        self.external_turn(action)

        # Data collection
        if self.pickle_file != None:
            pickle.dump(["Action", game_state, player_model, action], self.pickle_file)

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
                game_state = self._make_game_state(p.get_nr(), hint_indices, card_changed)
                player_model = self._make_player_model(p.get_nr())     

                p.inform(
                    action,
                    self.current_player,
                    game_state,
                    player_model,
                )

                # Data collection
                if self.pickle_file != None:
                    pickle.dump(["Inform", action, self.current_player, game_state, player_model], self.pickle_file)   

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
