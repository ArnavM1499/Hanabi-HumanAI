import pyximport; pyximport.install()
from copy import deepcopy
from fire import Fire
from itertools import product
import json
from multiprocessing import Pool
import os
from pprint import pprint
import random
import sys
from tqdm import tqdm

from hanabi import Game
from Agents.ChiefAgent.player_pool import PlayerPool
from Agents.behavior_clone_player import BehaviorPlayer

PLAYER_POOL1 = PlayerPool("Alice", 0, "Agents/configs/players.json")
PLAYER_POOL2 = PlayerPool("Bob", 1, "Agents/configs/players.json")


class TestbedGame(Game):
    def __init__(self, behavior_players, *args, **kwargs):
        super(TestbedGame, self).__init__(*args, **kwargs)
        self.behavior_players = behavior_players

        # format: [[(actual, predicted), ...], ...]
        self.behavior_results = [[] for _ in self.behavior_players]

    def single_turn(self):
        game_state = self._make_game_state(self.current_player)
        player_model = self._make_player_model(self.current_player)
        partner_knowledge_model = self._make_partner_knowledge_model(game_state)
        action = self.players[self.current_player].get_action(game_state, player_model)
        pred = self.behavior_players[self.current_player].get_action(
            game_state, player_model, partner_knowledge_model
        )
        self.behavior_results[self.current_player].append((action, pred))
        self.external_turn(action)


def test_single(player1_id, player2_id):
    try:
        TBG = TestbedGame(
            [
                BehaviorPlayer(0, "Alice", agent_id=player1_id),
                BehaviorPlayer(1, "Bob", agent_id=player1_id),
            ],
            [
                deepcopy(PLAYER_POOL1.player_dict[str(player1_id)]),
                deepcopy(PLAYER_POOL2.player_dict[str(player2_id)]),
            ],
            "dummy.csv",
        )
        TBG.run(100)
        counts = {k: [] for k in {player1_id, player2_id}}
        counts[player1_id].extend(TBG.behavior_results[0])
        counts[player2_id].extend(TBG.behavior_results[1])
        return counts
    except FileNotFoundError:
        return {}


def test_all(*agent_ids, output_json="", iters=20, seed=0, threads=16):
    random.seed(seed)
    profile = {}
    if len(agent_ids) == 0:
        agent_ids = [x for x in sorted(PLAYER_POOL1.player_dict.keys()) if x[1] != "9"]
    else:
        agent_ids = [str(x) for x in agent_ids]
    if len(agent_ids) == 0:
        print("no cloned agents found, nothing to do")
        return
    results = []
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    if threads == 1:
        for ids in product(agent_ids, agent_ids):
            for i in range(iters):
                results.append(test_single(*ids))
    else:
        p = Pool(threads)
        for ids in product(agent_ids, agent_ids):
            for i in range(iters):
                results.append(p.apply_async(test_single, ids))
        p.close()
        p.join()
        results = [x.get for x in results]
    for res in tqdm(results):
        for agent_id, result in res.items():
            if agent_id not in profile.keys():
                profile[agent_id] = []
            profile[agent_id].extend(
                [(actual.encode(), pred.encode()) for (actual, pred) in result]
            )
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
            try:
                precision = round(
                    (
                        statistics[k][action_key]["TP"]
                        / (
                            statistics[k][action_key]["TP"]
                            + statistics[k][action_key]["FP"]
                        )
                    ),
                    3,
                )
            except ZeroDivisionError:
                precision = "NA"
            statistics[k][action_key]["precision"] = precision
            try:
                recall = round(
                    (
                        statistics[k][action_key]["TP"]
                        / (
                            statistics[k][action_key]["TP"]
                            + statistics[k][action_key]["FN"]
                        )
                    ),
                    3,
                )
            except ZeroDivisionError:
                recall = "NA"
            statistics[k][action_key]["recall"] = recall
        statistics[k]["accuracy"] = sum(x["TP"] for x in statistics[k].values()) / len(
            values
        )
    sys.stdout = stdout
    pprint(statistics)
    if output_json != "":
        json.dump(statistics, open(output_json, "w"))


if __name__ == "__main__":
    Fire()
