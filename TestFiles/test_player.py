import pyximport; pyximport.install()
import csv
import fire
import json
from multiprocessing import Pool
import os
import pickle
from pprint import pprint
import random
from subprocess import Popen, DEVNULL
from signal import SIGTERM
import sys
import time
import threading
from tqdm import tqdm

from hanabi import Game
from Agents.player import Player
from Agents.ChiefAgent.player_pool import PlayerPool
from Agents.value_player import ValuePlayer

base_path = os.path.abspath(__file__).replace("TestFiles/test_player.py", "")
pool_path = os.path.abspath(__file__).replace(
    "TestFiles/test_player.py", "Agents/configs/players.json"
)
player_pool = json.load(open(pool_path))
# pool_ids = ["00001","00002","00003","00004","00005","10001","10002","10003","10004","10005"]
pool_ids = ["10001"]
dummy_pool = PlayerPool("dummy", 0, pool_path, pool_ids)


def try_pickle(file):
    try:
        return pickle.load(file)
    except:  # noqa
        return None


def relabel_run(original_filename, relabeler, game_log_filename, output_filename):
    new_rows = []
    relabeled_actions = []

    # simulate relabeler on game to get new actions
    with open(game_log_filename, "rb") as f:
        row = try_pickle(f)
        cnt = [0, 0, 0]

        while row is not None:
            if row[0] == "Action" and row[1].get_current_player() == relabeler.pnr:
                relabeled_actions.append(relabeler.get_action(row[1], row[2]))
                cnt[0] += 1

            if row[0] == "Inform" and row[4] == relabeler.pnr:
                relabeler.inform(row[3], row[5], row[1], row[2])
                if row[5] == relabeler.pnr:
                    cnt[1] += 1
                else:
                    cnt[2] += 1

            row = try_pickle(f)

    # create new rows for file with relabeled actions
    with open(original_filename, "rb") as f:
        row = try_pickle(f)
        idx = 0

        while row is not None:
            if row != [] and row[-1] == relabeler.pnr:
                # replacing encoded action
                new_rows.append(row[:-2] + [relabeled_actions[idx].encode()] + row[-1:])
                idx += 1
            else:
                new_rows.append(row)
            row = try_pickle(f)

    # update file so that the top-level training scripts get the relabeled version
    with open(output_filename, "ab+") as f:
        for r in new_rows:
            pickle.dump(r, f)


def run_single(
    file_name,
    player="00001",
    player2=None,
    key=None,
    key2=None,
    clean=False,
    print_game=True,
    print_data=False,
    relabeler=None,
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
    if relabeler is None:
        RELABEL = None
    elif isinstance(relabeler, Player):
        RELABEL = relabeler
    else:
        RELABEL = dummy_pool.from_dict(
            "Relabeler", int(str(relabeler)[-1]), player_pool[str(relabeler)[:-1]]
        )
    if (key is not None) and hasattr(P1, "set_from_key"):
        P1.set_from_key(key)
    elif print_data:
        print("player1 key not set")
    if (key2 is not None) and hasattr(P2, "set_from_key"):
        P2.set_from_key(key2)
    elif print_data:
        print("player2 key not set")

    if RELABEL is None:
        gamelog_file = None
        original_output_file = file_name
    else:
        gamelog_file_name = file_name[:-4] + "LOG.pkl"
        gamelog_file = open(gamelog_file_name, "wb")
        original_output_file = file_name[:-4] + "ORG.pkl"

    G = Game(
        [P1, P2], original_output_file, print_game=print_game, pickle_file=gamelog_file
    )

    score = G.run(100)
    hints = G.hints
    hits = G.hits
    turns = G.turn
    print(turns)
    if RELABEL is not None:
        gamelog_file.close()
        relabel_run(original_output_file, RELABEL, gamelog_file_name, file_name)
        os.remove(gamelog_file_name)
        os.remove(original_output_file)

    if clean:
        try:
            os.remove(file_name)
        except FileNotFoundError:
            pass
    return score, hints, hits, turns


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
        p = Pool(min(24, iters))
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
    full = results[0].count(25) / iters
    print(
        "{} games: avg: {}, min: {}, max: {}, median: {}, mode: {}, full: {}".format(
            iters, avg, smin, smax, smid, smod, full
        )
    )
    print(
        "average: hints left: {}, hits left: {}, turns: {}".format(hints, hits, turns)
    )

    if print_details:
        pprint(list(zip(*results)))

    return iters, avg, smin, smax, smid, smod, hints, hits, turns


def sequential_test(
    player, player2=None, iters=5000, seed=0, save_pkl_dir=None, tid=0, relabeler=None
):
    random.seed(seed)
    iters = int(iters)
    if isinstance(save_pkl_dir, str):
        if not os.path.isdir(save_pkl_dir):
            os.makedirs(save_pkl_dir)
        save_file = os.path.join(
            save_pkl_dir, "{}_{}_{}.pkl".format(player, player2, str(seed).zfill(4))
        )
        print("saving into ", os.path.abspath(save_file))
        if os.path.isfile(save_file):
            # remove previous data
            f = open(save_file, "w")
            f.close()
        if tid == 0:
            for i in tqdm(range(iters)):
                run_single(
                    save_file,
                    player,
                    player2,
                    clean=False,
                    print_game=False,
                    relabeler=relabeler,
                )
        else:
            for i in range(iters):
                run_single(
                    save_file,
                    player,
                    player2,
                    clean=False,
                    print_game=False,
                    relabeler=relabeler,
                )
    else:
        print(save_pkl_dir, "is not a str")
        for i in tqdm(range(iters)):
            run_single(
                "sink_{}.csv".format(i),
                player,
                player2,
                clean=True,
                print_game=False,
                relabeler=relabeler,
            )


########################################################
# Generating data for LSTMs:
# player and player2 may be passed in as id strings
# relabeler will be passed in as an id string with the
#           pnr appended to the id string
########################################################
def generate_data(
    player,
    save_pkl_dir,
    player2=None,
    iters=20000,
    threads=16,
    method="thread",
    seed=0,
    relabeler=None,
):
    if player2 is None:
        player2 = player
    if method == "thread":
        tds = []
        print("using {} threads".format(threads))
        for i in range(threads):
            thread = threading.Thread(
                target=sequential_test,
                args=(
                    player,
                    player2,
                    iters / threads,
                    seed + i,
                    save_pkl_dir,
                    1,
                    relabeler,
                ),
            )
            tds.append(thread)

        for thr in tds:
            thr.start()

        for thr in tds:
            thr.join()
    elif method == "process":
        P = Pool(threads)
        for i in range(threads):
            P.apply_async(
                sequential_test,
                (
                    player,
                    player2,
                    iters // threads,
                    seed + i,
                    save_pkl_dir,
                    1,
                    relabeler,
                ),
            )
        P.close()
        P.join()
    elif method == "single":
        sequential_test(player, player2, iters, 0, save_pkl_dir, 0, relabeler)
    elif method == "subprocess":
        processes = []
        for i in range(threads):
            processes.append(
                Popen(
                    " ".join(
                        [
                            "cd {}; ".format(base_path),
                            "python3 -m TestFiles.test_player sequential_test",
                            "--player=" + str(player),
                            "--player2=" + str(player2),
                            "--iters=" + str(iters),
                            "--seed=" + str(i + seed),
                            "--save_pkl_dir=" + save_pkl_dir,
                            "--tid=1",
                            "--relabeler=" + str(relabeler),
                        ]
                    ),
                    shell=True,
                    stdout=DEVNULL,
                    preexec_fn=os.setsid,
                )
            )
        try:
            for p in processes:
                p.wait()
                try:
                    os.killpg(os.getpgid(p.pid), SIGTERM)
                except ProcessLookupError:
                    pass
        except KeyboardInterrupt:
            for p in processes:
                try:
                    os.killpg(os.getpgid(p.pid), SIGTERM)
                except ProcessLookupError:
                    pass
    else:
        print("method {} not found".format(method))
        raise NotImplementedError


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
