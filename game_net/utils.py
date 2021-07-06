from fire import Fire
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from random import shuffle
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
                subdir = os.path.join(dataset_dir, name2id[player[p]])
                if not os.path.isdir(subdir):
                    os.makedirs(subdir)
                output_path = os.path.join(subdir, "{}.npy".format(str(a).zfill(2)))
                with open(output_path, "ab+") as fout:
                    np.save(
                        fout,
                        np.array(s, dtype=np.int8),
                    )


def merge_np(dataset_root, agent_name, train_split=0.9):
    agent_name = str(agent_name)
    all_data = []
    print("loading all data into memory")
    for i in tqdm(range(20)):
        all_data.append([])
        with open(
            os.path.join(dataset_root, agent_name, "{}.npy".format(str(i).zfill(2))),
            "rb",
        ) as fin:
            while True:
                try:
                    all_data[-1].append(np.load(fin))
                except ValueError:
                    break

    print("split train & val")
    trains = []
    vals = []
    for i in range(20):
        shuffle(all_data[i])
        split = int(len(all_data[i]) * train_split)
        vals.append(all_data[i][split:])
        trains.append(all_data[i][:split])
    num_train = sum([len(x) for x in trains])
    weights = [20 * len(x) / num_train for x in trains]
    all_train = []
    for i, action in enumerate(trains):
        for state in action:
            all_train.append((i, state))
    shuffle(all_train)

    train_np = open(os.path.join(dataset_root, "{}_train.npy".format(agent_name)), "wb")
    print("writing into train")
    np.save(train_np, np.array(weights, dtype=np.float32))
    for i, state in all_train:
        np.save(train_np, state)
        np.save(train_np, i)
    train_np.close()
    val_np = open(os.path.join(dataset_root, "{}_val.npy".format(agent_name)), "wb")
    print("writing into val")
    np.save(val_np, np.ones(20, dtype=np.float32))
    for i in range(20):
        for j in range(len(vals[i])):
            np.save(val_np, vals[i][j])
            np.save(val_np, i)
    val_np.close()


def draw_transfer_curve(np_file, save_image, title=""):
    points = []
    with open(np_file, "rb") as fin:
        while True:
            try:
                points.append(np.load(fin))
            except ValueError:
                break
    points = np.array(points)
    plt.title(title)
    plt.xlabel("samples used")
    plt.ylabel("accuracy")
    plt.plot(points[:, 0], points[:, 1], linewidth=1)
    plt.savefig(save_image, dpi=1000)


if __name__ == "__main__":
    Fire()
