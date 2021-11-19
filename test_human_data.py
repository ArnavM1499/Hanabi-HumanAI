import json
import sys
import os
from tqdm import tqdm
from collections import defaultdict
from Agents.player import Player, Action
from common_game_functions import *
from hanabi import Game

ANALYZE_NUMBER = 20000


num_games_dict = defaultdict(lambda: 0)


class HumanPlayer(Player):
    def __init__(self, name, pnr, action_index, move_list):
        super().__init__(name, pnr)
        self.move_list = move_list
        self.next_action = action_index
        self.partner_nr = 1 - self.pnr

    def get_action(self, game_state, base_player_model):
        if self.next_action >= len(self.move_list):
            return None
        action_json = self.move_list[self.next_action]
        # increment by number of players
        self.next_action += 2
        # PLAY
        if action_json["type"] == 0:
            for i in range(len(game_state.hand_idxs[self.pnr])):
                if game_state.hand_idxs[self.pnr][i] == action_json["target"]:
                    return Action(PLAY, cnr=i)
        # DISCARD
        elif action_json["type"] == 1:
            for i in range(len(game_state.hand_idxs[self.pnr])):
                if game_state.hand_idxs[self.pnr][i] == action_json["target"]:
                    return Action(DISCARD, cnr=i)
        elif action_json["type"] == 2:
            assert(action_json["target"] == self.partner_nr)
            return Action(HINT_COLOR, self.partner_nr, col=action_json["value"])
        elif action_json["type"] == 3:
            assert (action_json["target"] == self.partner_nr)
            return Action(HINT_NUMBER, self.partner_nr, num=action_json["value"])
        else:
            # "end game" action -- run out of time, etc.
            return None


scores = [0] * 26


def load_games(directory):
    i = 0
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in tqdm(files):
            with open(os.path.join(root, name), 'r', encoding='utf-8') as file_object:
                a = json.load(file_object)
                # NOT IN DOCUMENTATION -- if player 1 is the first player, there will be a label in options
                try:
                    assert(len(a["players"]) == 2)
                    # H1 = HumanPlayer(a["players"][0], 0, a["actions"])
                    # H2 = HumanPlayer(a["players"][1], 1, a["actions"])
                    if "options" in a.keys() and "startingPlayer" in a["options"].keys():
                        assert(a["options"]["startingPlayer"] == 1)
                        H1 = HumanPlayer(a["players"][0], 0, 1, a["actions"])
                        H2 = HumanPlayer(a["players"][1], 1, 0, a["actions"])
                        G = Game([H1, H2], "blah.pkl", deck=load_deck(a["deck"]), starter=1, print_game=False)
                    else:
                        H1 = HumanPlayer(a["players"][0], 0, 0, a["actions"])
                        H2 = HumanPlayer(a["players"][1], 1, 1, a["actions"])
                        G = Game([H1, H2], "blah.pkl", deck=load_deck(a["deck"]), starter=0, print_game=False)
                    scores[G.run(-1)] += 1
                except AssertionError:
                    print(os.path.join(root, name))
                    file_object.close()
                    continue
                file_object.close()
            i += 1
            if i >= ANALYZE_NUMBER:
                break


if __name__ == "__main__":
    load_games(sys.argv[1])
    # player_list = list(num_games_dict.items())
    # player_list.sort(reverse=True, key=lambda x: x[1])
    # for x in player_list:
    #     if x[1] > 10:
    #         print(x)
    print(scores)
