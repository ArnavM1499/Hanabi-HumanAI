import csv
import fire
import json
from multiprocessing import Pool
import os
from pprint import pprint
import random
import time
from hanabi import Game
from Agents.player import Player
from Agents.ChiefAgent.player_pool import PlayerPool

player_pool = json.load(open("Agents/configs/players.json"))


def run_single(
    file_name,
    player="00001",
    player2=None,
    key=0,
    key2=0,
    clean=False,
):

    print("running hanabi game on ", player, " and ", player2 if player2 else "itself")
    if not player2:
        player2 = player
        key2 = key
    if isinstance(player, str):
        P1 = PlayerPool.from_dict("Alice", 0, player_pool[player])
    elif isinstance(player, Player):
        P1 = player
    else:
        assert False
    if isinstance(player2, str):
        P2 = PlayerPool.from_dict("Bob", 1, player_pool[player2])
    elif isinstance(player2, Player):
        P2 = player2
    else:
        assert False
    if hasattr(P1, "set_from_key"):
        P1.set_from_key(key)
    else:
        print("player does support set from key")
    if hasattr(P2, "set_from_key"):
        P2.set_from_key(key2)
    else:
        print("player2 does support set from key")
    G = Game([P1, P2], file_name)
    score = G.run(100)
    hints = G.hints
    hits = G.hits
    turns = G.turn
    if clean:
        os.remove(file_name)
    return (score, hints, hits, turns)


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
    player="00001", player2=None, iters=5000, print_details=False, key=0, key2=0
):
    p = Pool(min(16, iters))
    res = p.starmap_async(
        run_single,
        [
            ("sink_{}.csv".format(i), player, player2, key, key2, True)
            for i in range(iters)
        ],
    )
    p.close()
    results = [list(x) for x in zip(*res.get())]  # [[scores], [hints], [hits], [turns]
    results[0].sort()
    time.sleep(5)  # wait for async file writes
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

    if print_details:
        pprint(list(zip(*results)))

    return iters, avg, smin, smax, smid, smod, hints, hits, turns


def sequential_test(player, player2=None, iters=5000):
    random.seed(0)
    for i in range(iters):
        run_single("sink_{}.csv".format(i), player, player, True)


def parameter_search(
    player, max_key, iters=1000, result_file="search_result.json", pop_size=12
):
    mask_size = 1
    while 2 ** mask_size < max_key:
        mask_size += 1
    table = {}
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
