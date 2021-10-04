import csv
import fire
import json
from multiprocessing import Pool
import os
from pprint import pprint
import random
import sys
import time
from hanabi import Game
from Agents.player import Player
from Agents.ChiefAgent.player_pool import PlayerPool
from Agents.value_player import ValuePlayer

player_pool = json.load(open("Agents/configs/players.json"))
dummy_pool = PlayerPool("dummy", 0, "Agents/configs/players.json")


def run_single(
    file_name,
    player="00001",
    player2=None,
    key=None,
    key2=None,
    clean=False,
):

    if not player2:
        player2 = player
        key2 = key
    if isinstance(player, Player):
        P1 = player
    else:
        P1 = dummy_pool.from_dict("Alice", 0, player_pool[str(player)])
    if isinstance(player2, Player):
        P2 = player2
    else:
        P2 = dummy_pool.from_dict("Bob", 1, player_pool[str(player2)])
    if (key is not None) and hasattr(P1, "set_from_key"):
        P1.set_from_key(key)
    else:
        print("player1 key not set")
    if (key2 is not None) and hasattr(P2, "set_from_key"):
        P2.set_from_key(key2)
    else:
        print("player2 key not set")
    G = Game([P1, P2], file_name)
    score = G.run(100)
    hints = G.hints
    hits = G.hits
    turns = G.turn
    if clean:
        os.remove(file_name)
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
    os.remove(file_name)
    return score


params = {
    "name": "Alice",
    "hint_weight": 1000.0,
    "discard_type": "first",
    "default_hint": "high",
    "card_count": True,
    "card_count_partner": True,
    "get_action_values": False,
    "play_threshold": 0.95,
    "discard_threshold": 0.5,
    "play_bias": 1.0,
    "disc_bias": 0.7,
    "hint_bias": 0.9,
    "hint_biases": [0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.0, 1.0],
    "play_biases": [0, 0.7, 0.9, 1.0],
}


def test_param_config(config, iters=400):
    p = Pool(16)
    res = p.starmap_async(
        from_param_dict,
        [("sink_{}.csv".format(i), config) for i in range(iters)],
    )
    p.close()
    ls = res.get()
    print(sum(ls) / iters)
    print(ls)


def hc2():
    test_param_config(params)


def hc():
    for i in range(1):
        res = from_param_dict("xd", params)
        if res < 3:
            return


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


def test_player(
    player="00001",
    player2=None,
    iters=5000,
    print_details=False,
    key=None,
    key2=None,
    single_thread=False,
):
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    if single_thread:
        res = [
            list(run_single("sink_{}.csv".format(i), player, player2, key, key2, True))
            for i in range(iters)
        ]
        results = [list(x) for x in zip(*res)]
    else:
        p = Pool(min(16, iters))
        res = p.starmap_async(
            run_single,
            [
                ("sink_{}.csv".format(i), player, player2, key, key2, True)
                for i in range(iters)
            ],
        )
        p.close()
        results = [
            list(x) for x in zip(*res.get())
        ]  # [[scores], [hints], [hits], [turns]
    results[0].sort()
    time.sleep(5)  # wait for async file writes
    sys.stdout = stdout
    avg = sum(results[0]) / iters
    smin = results[0][0]
    smax = results[0][-1]
    smid = results[0][iters // 2]
    smod = max(set(results[0]), key=results[0].count)
    hints = sum(results[1]) / iters
    hits = sum(results[2]) / iters
    turns = sum(results[3]) / iters
    print(
        "{} games: avg: {}, min: {}, max: {}, median: {}, mode: {}".format(
            iters,
            avg,
            smin,
            smax,
            smid,
            smod,
        )
    )
    print(
        "average: hints left: {}, hits left: {}, turns: {}".format(hints, hits, turns)
    )
    return sum(results[0]) / iters

    if print_details:
        pprint(list(zip(*results)))

    return iters, avg, smin, smax, smid, smod, hints, hits, turns


def sequential_test(player, player2=None, iters=5000, seed=0, save_pkl_dir=None):
    random.seed(seed)
    iters = int(iters)
    if isinstance(save_pkl_dir, str):
        print("saving into ", os.path.abspath(save_pkl_dir))
        if not os.path.isdir(save_pkl_dir):
            os.makedirs(save_pkl_dir)
        save_file = os.path.join(
            save_pkl_dir, "{}_{}_{}.pkl".format(player, player2, seed)
        )
        if os.path.isfile(save_file):
            # remove previous data
            f = open(save_file, "w")
            f.close()
        for i in range(iters):
            run_single(
                save_file,
                player,
                player2,
                clean=False,
            )
    else:
        for i in range(iters):
            run_single("sink_{}.csv".format(i), player, player2, clean=True)


def parameter_search(
    player, max_key, iters=1000, result_file="log/search_result.json", pop_size=12
):
    if os.path.isfile(result_file):
        table = json.load(open(result_file, "r"))
        print("loaded {} entries from cached table".format(len(table)))
    else:
        table = {}
    mask_size = 1
    while 2 ** mask_size < max_key:
        mask_size += 1
    keys = random.sample(list(range(max_key)), pop_size)
    for i in range(iters):
        scores = []
        for key in keys:
            if key not in table.keys():
                table[key] = test_player(player, key=key, iters=150)[1]
            scores.append(table[key])
        with open(result_file, "w") as fout:
            json.dump(table, fout)
        tot = sum(scores)
        scores = [x / tot for x in scores]
        new_keys = []
        while len(new_keys) < pop_size:
            (key, key2) = tuple(random.choices(population=keys, weights=scores, k=2))
            hi = random.randint(0, mask_size)
            lo = random.randint(0, hi)
            mask = ((1 << (hi - lo)) - 1) << lo
            mask_neg = (1 << mask_size) - 1 - mask
            tmp = ((key & mask) + (key2 & mask_neg)) % max_key
            if random.random() < 0.1:
                tmp = tmp ^ random.randint(0, max_key - 1)
            new_keys.append(tmp)
        keys = new_keys


if __name__ == "__main__":
    fire.Fire()
