from copy import deepcopy
from fire import Fire
from itertools import product
import json
from multiprocessing import Pool  # noqa
from pprint import pprint
import random
from tqdm import tqdm

from hanabi import Game
from Agents.ChiefAgent.player_pool import PlayerPool
from game_net.behavior_clone import BehaviorClone

PLAYER_POOL1 = PlayerPool("Alice", 0, "Agents/configs/players.json")
PLAYER_POOL2 = PlayerPool("Bob", 1, "Agents/configs/players.json")


class TestbedGame(Game):
    def __init__(self, behavior_funcs, *args, **kwargs):
        super(TestbedGame, self).__init__(*args, **kwargs)
        self.behavior_funcs = behavior_funcs

        # format: [[(actual, predicted), ...], ...]
        self.behavior_results = [[] for _ in self.behavior_funcs]

    def single_turn(self):
        game_state = self._make_game_state(self.current_player)
        player_model = self._make_player_model(self.current_player)
        action = self.players[self.current_player].get_action(game_state, player_model)
        pred = self.behavior_funcs[self.current_player](game_state, player_model)
        self.behavior_results[self.current_player].append((action, pred))
        self.external_turn(action)
        if self.hints < 0:
            print(self.players[self.current_player].name, "makes an illegal hint")
            import pdb

            pdb.set_trace()


def test_single(player1_id, player2_id):
    TBG = TestbedGame(
        [
            lambda s, m: BehaviorClone.predict(player1_id, s, m),
            lambda s, m: BehaviorClone.predict(player2_id, s, m),
        ],
        [
            deepcopy(PLAYER_POOL1.player_dict[player1_id]),
            deepcopy(PLAYER_POOL2.player_dict[player2_id]),
        ],
        "dummy.csv",
    )
    TBG.run(100)
    counts = {k: [] for k in {player1_id, player2_id}}
    counts[player1_id].extend(TBG.behavior_results[0])
    counts[player2_id].extend(TBG.behavior_results[1])
    return counts


def test_all(output_json="", iters=20, seed=0):
    random.seed(seed)
    profile = {}
    for ids in tqdm(
        product(PLAYER_POOL1.player_dict.keys(), PLAYER_POOL2.player_dict.keys())
    ):
        try:
            for i in range(iters):
                for agent_id, result in test_single(*ids).items():
                    if agent_id not in profile.keys():
                        profile[agent_id] = []
                    profile[agent_id].extend(
                        [(actual.encode(), pred.encode()) for (actual, pred) in result]
                    )
        except FileNotFoundError:
            continue
    statistics = {k: {} for k in profile.keys()}
    for k, values in profile.items():
        for actual, pred in values:
            if actual not in statistics[k].keys():
                statistics[k][actual] = {"TP": 0, "FP": 0, "FN": 0}
            if pred not in statistics[k].keys():
                statistics[k][pred] = {"TP": 0, "FP": 0, "FN": 0}
            if actual == pred:
                statistics[k][actual]["TP"] += 1
            else:
                statistics[k][actual]["FN"] += 1
                statistics[k][pred]["FP"] += 1
        for action_key in statistics[k].keys():
            statistics[k][action_key]["precision"] = statistics[k][action_key]["TP"] / (
                statistics[k][action_key]["TP"] + statistics[k][action_key]["FP"]
            )
            statistics[k][action_key]["recall"] = statistics[k][action_key]["TP"] / (
                statistics[k][action_key]["TP"] + statistics[k][action_key]["FN"]
            )
        statistics[k]["accuracy"] = sum(x["TP"] for x in statistics[k].values()) / len(
            values
        )
    pprint(statistics)
    if output_json != "":
        json.dump(statistics, open(output_json, "w"))


if __name__ == "__main__":
    Fire()
