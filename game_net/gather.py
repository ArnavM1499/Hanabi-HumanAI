from fire import Fire
import numpy as np
import os
import pickle
from tqdm import tqdm
from common_game_functions import decode_state


def pkl_to_txt(dataset_dir, *paths):
    name2id = {}
    cnt = 0
    for path in tqdm(paths):
        if not os.path.isfile(path):
            print(path, "Not Found!")
            continue
        (p1, p2, _) = tuple(os.path.basename(path).split("_"))
        if p1 not in name2id.keys():
            name2id[p1] = cnt
            cnt += 1
        if p2 not in name2id.keys():
            name2id[p2] = cnt
            cnt += 1
        player = [p1, p2]
        with open(path, "rb") as f:
            while True:
                try:
                    p, a, state = decode_state(pickle.load(f))
                except EOFError:
                    break
                if len(state) == 558:
                    encodings = [0] * 18
                    for i, s in enumerate(state):
                        encodings[i // 31] *= 3
                        encodings[i // 31] += s
                    with open(
                        os.path.join(dataset_dir, "{}.txt".format(player[p])), "a+"
                    ) as fout:
                        fout.write(str(name2id[player[p]]) + " ")
                        fout.write(str(a) + " ")
                        for encoding in encodings:
                            fout.write(str(encoding) + " ")
                        fout.write("\n")


def pkl_to_np(dataset_dir, *paths):
    for path in tqdm(paths):
        if not os.path.isfile(path):
            print(path, "Not Found!")
            continue
        (p1, p2, _) = tuple(os.path.basename(path).split("_"))
        player = [p1, p2]
        with open(path, "rb") as f:
            while True:
                try:
                    p, a, s = decode_state(pickle.load(f))
                except EOFError:
                    break
                subdir = os.path.join(dataset_dir, player[p])
                if not os.path.isdir(subdir):
                    os.makedirs(subdir)
                np.save(
                    open(os.path.join(subdir, "{}.npy".format(str(a).zfill(2))), "ab+"),
                    np.array(s, dtype=np.int8),
                )


if __name__ == "__main__":
    Fire()
