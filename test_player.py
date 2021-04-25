import csv
import fire
import json
from multiprocessing import Pool
import os
import random
import time
from hanabi import Game
from Agents.player import Player

player_pool = json.load(open("Agents/configs/players.json"))


def run_single(
    file_name,
    player="00001",
    player2=None,
    clean=False,
):

    print("running hanabi game on ", player, " and ", player2 if player2 else "itself")
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


def test_player(player="00001", player2=None, iters=1):
    p = Pool(1)
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


def sequential_test(player, player2=None, iters=5000):
    random.seed(0)
    for i in range(iters):
        run_single("sink_{}.csv".format(i), player, player, True)


if __name__ == "__main__":
    fire.Fire()
