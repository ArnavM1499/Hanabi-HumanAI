import csv
import fire
import json
from multiprocessing import Pool
import os
import random
import time
import copy
from hanabi import Game
from Agents.player import Player
from Agents.value_player import ValuePlayer

player_pool = json.load(open("Agents/configs/players.json"))


def run_single(
    file_name,
    player="00001",
    player2=None,
    clean=False,
):

    # print("running hanabi game on ", player, " and ", player2 if player2 else "itself")
    # print(player_pool)
    if not player2:
        player2 = player
    P1 = Player.from_dict("Alice", 0, player_pool[player])
    P2 = Player.from_dict("Bob", 1, player_pool[player2])
    G = Game([P1, P2], file_name)
    score = G.run(100)
    hints = G.hints
    hits = G.hits
    turns = G.turn
    #if clean:
    #    os.remove(file_name)
    return (score, hints, hits, turns)


def from_param_dict(file_name, dict):
    dict["pnr"] = 0
    dict["name"] = "Alice"
    p1 = ValuePlayer(**dict)
    dict["pnr"] = 1
    dict["name"] = "Bob"
    p2 = ValuePlayer(**dict)
    G = Game([p1, p2], file_name)
    score = G.run(100)
    # os.remove(file_name)
    return score


# starting configuration for hill climbing search
params = {
    "hint_weight": 1000.0,
    "play_preference": "left",
    "discard_preference": "left",
    "card_count": True,
    "card_count_partner": True,
    "discard_base_value": 0.2,
    "protect_importance": 0.4,
    "play_low_multiplier": 0.1,
    "discard_low_multiplier": 0.5,
    "play_threshold": 0.95,
    "discard_threshold": 0.5,
    "play_bias": 1.0,
    "disc_bias": 0.7,
    "hint_bias": 0.9,
    "dynamic_bias": True,
    "hint_biases": [0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.0, 1.0],
    "play_biases": [0, 0.7, 0.9, 1.0]
}


def test_param_config(config, iters=400):
    p = Pool(16)
    res = p.starmap_async(
        from_param_dict,
        [("sink_{}.csv".format(i), config) for i in range(iters)],
    )
    p.close()
    ls = res.get()
    return sum(ls) / iters


def hc1():
    print(test_param_config(params))


# list of parameters to do search on
settings = ["discard_base_value", "protect_importance", "play_low_multiplier", "discard_low_multiplier",
            "play_threshold", "discard_threshold", "play_bias", "disc_bias", "hint_bias",
            ["hint_biases"], ["play_biases"]]
# settings = ["play_bias", "disc_bias", "hint_bias"]


# Hill climbing algorithm. Does a random mutation each iteration
def hill_climb(start_mutation_size, end_mutation_size, start_params, list_params):
    current_mutation_size = start_mutation_size
    best_score = test_param_config(start_params)
    best_params = start_params
    num_tests = 0
    global_num_tests = 0
    while current_mutation_size > end_mutation_size:
        test_params = copy.deepcopy(best_params)
        for param in list_params:
            # small hack to be able to process list variables
            if isinstance(param, list):
                for i in range(len(test_params[param[0]])):
                    mutation_pct = random.uniform(1.0 - current_mutation_size, 1.0 + current_mutation_size)
                    test_params[param[0]][i] *= mutation_pct
            else:
                mutation_pct = random.uniform(1.0 - current_mutation_size, 1.0 + current_mutation_size)
                test_params[param] *= mutation_pct
        new_score = test_param_config(test_params)
        num_tests += 1
        global_num_tests += 1
        if new_score > best_score:
            best_params = test_params
            best_score = new_score
            num_tests = 0
            filename = "result_" + str(global_num_tests) + ".txt"
            with open(filename, "w") as outfile:
                json.dump(best_params, outfile)
                outfile.write(str(best_score))
                outfile.close()
        if global_num_tests % 25 == 0:
            current_mutation_size *= 0.5
    with open("result.json", "w") as outfile:
        print(best_score)
        json.dump(best_params, outfile)


def start_hc():
    hill_climb(0.2, 0.01, params, settings)


def start_hc_from_json(player="00008"):
    hill_climb(0.2, 0.05, player_pool[player], settings)


def record_game(
    player="00001",
    file_name="hanabi_data.csv",
    mode="w",
    iters=1,
    player2=None,
):
    with open(file_name, mode) as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "Player",
                "Action Type",
                "Board",
                "Discards",
                "Hints available",
                "Knowledge from hints",
            ]
        )

    for i in range(iters):
        (score, hints, hits, turns) = run_single(player, file_name, player2=player2)
        print(
            "score: {}, hints left: {}, hits left: {}, turns: {}".format(
                score, hints, hits, turns
            )
        )


def test_player(player="00008", player2=None, iters=1000):
    p = Pool(16)
    res = p.starmap_async(
        run_single,
        [("sink_{}.csv".format(i), player, player2, True) for i in range(iters)],
    )
    p.close()
    results = [list(x) for x in zip(*res.get())]  # [[scores], [hints], [hits], [turns]
    results[0].sort()
    time.sleep(5)  # wait for async file writes
    print(
        "{} games: avg: {}, min: {}, max: {}, median: {}, mode: {}".format(
            iters,
            sum(results[0]) / iters,
            results[0][0],
            results[0][-1],
            results[0][iters // 2],
            max(set(results[0]), key=results[0].count),
        )
    )
    print(
        "average: hints left: {}, hits left: {}, turns: {}".format(
            sum(results[1]) / iters, sum(results[2]) / iters, sum(results[3]) / iters
        )
    )
    return sum(results[0]) / iters




def sequential_test(player, player2=None, iters=1):
    random.seed(0)
    for i in range(iters):
        run_single("sink_{}.csv".format(i), player, player, True)


if __name__ == "__main__":
    fire.Fire()
