from fire import Fire
from glob import glob
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


def pkl_to_np(dataset_dir, *paths, rename=False):
    name2id = {}
    cnt = 0
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if len(paths) == 1 and os.path.isdir(paths[0]):
        paths = glob(os.path.join(paths[0], "*.pkl"))
    for path in tqdm(paths):
        if not os.path.isfile(path):
            print(path, "Not Found!")
            continue
        (p1, p2, _) = tuple(os.path.basename(path).split("_"))
        if p1 not in name2id.keys():
            if rename:
                name2id[p1] = str(cnt).zfill(5)
                cnt += 1
            else:
                name2id[p1] = p1
        if p2 not in name2id.keys():
            if rename:
                name2id[p2] = str(cnt).zfill(5)
                cnt += 1
            else:
                name2id[p2] = p2
        player = [p1, p2]
        with open(path, "rb") as f:
            while True:
                try:
                    p, a, s = decode_state(pickle.load(f))
                except EOFError:
                    break
                output_path = os.path.join(
                    dataset_dir, "{}.npy".format(name2id[player[p]])
                )
                with open(output_path, "ab+") as fout:
                    np.save(
                        fout,
                        np.array(s, dtype=np.int8),
                    )
                    np.save(
                        fout,
                        np.array([a], dtype=np.int8),
                    )


if __name__ == "__main__":
    Fire()
