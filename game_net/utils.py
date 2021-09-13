from fire import Fire
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from random import shuffle, random
from shutil import copyfile
from tqdm import tqdm
from common_game_functions import decode_state, StartOfGame


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
                except StartOfGame:
                    continue
                subdir = os.path.join(dataset_dir, name2id[player[p]])
                if not os.path.isdir(subdir):
                    os.makedirs(subdir)
                output_path = os.path.join(subdir, "{}.npy".format(str(a).zfill(2)))
                with open(output_path, "ab+") as fout:
                    np.save(
                        fout,
                        np.array(s, dtype=np.int8),
                    )


def pkl_to_lstm_np(dataset_dir, *paths, rename=False, train_split=0.7):
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
        game_states = [[], []]
        game_actions = [[], []]
        with open(path, "rb") as f:
            if pickle.load(f) != []:
                print(path, ": bad format")
            else:
                while True:
                    try:
                        p, a, s = decode_state(pickle.load(f))
                    except EOFError:
                        break
                    except StartOfGame:
                        p1_output_path_all = os.path.join(
                            dataset_dir, name2id[p1] + "_all.npy"
                        )
                        p2_output_path_all = os.path.join(
                            dataset_dir, name2id[p2] + "_all.npy"
                        )
                        p1_output_path_train = os.path.join(
                            dataset_dir, name2id[p2] + "_train.npy"
                        )
                        p2_output_path_train = os.path.join(
                            dataset_dir, name2id[p2] + "_train.npy"
                        )
                        p1_output_path_val = os.path.join(
                            dataset_dir, name2id[p2] + "_val.npy"
                        )
                        p2_output_path_val = os.path.join(
                            dataset_dir, name2id[p2] + "_val.npy"
                        )
                        game_states = [np.array(x, dtype=np.int8) for x in game_states]
                        game_actions = [
                            np.array(x, dtype=np.int8) for x in game_actions
                        ]
                        with open(p1_output_path_all, "ab+") as fout:
                            np.save(fout, game_states[0])
                            np.save(fout, game_actions[0])
                        with open(p2_output_path_all, "ab+") as fout:
                            np.save(fout, game_states[1])
                            np.save(fout, game_actions[1])
                        if random() < train_split:
                            with open(p1_output_path_train, "ab+") as fout:
                                np.save(fout, game_states[0])
                                np.save(fout, game_actions[0])
                            with open(p2_output_path_train, "ab+") as fout:
                                np.save(fout, game_states[1])
                                np.save(fout, game_actions[1])
                        else:
                            with open(p1_output_path_val, "ab+") as fout:
                                np.save(fout, game_states[0])
                                np.save(fout, game_actions[0])
                            with open(p2_output_path_val, "ab+") as fout:
                                np.save(fout, game_states[1])
                                np.save(fout, game_actions[1])
                        game_states = [[], []]
                        game_actions = [[], []]

                    game_states[p].append(s)
                    game_actions[p].append(a)


def merge_np(dataset_root, agent_name, train_split=0.7):
    agent_name = str(agent_name)
    all_data = []
    print("loading all data into memory")
    for i in tqdm(range(20)):
        all_data.append([])
        try:
            with open(
                os.path.join(
                    dataset_root, agent_name, "{}.npy".format(str(i).zfill(2))
                ),
                "rb",
            ) as fin:
                while True:
                    try:
                        all_data[-1].append(np.load(fin))
                    except ValueError:
                        break
        except FileNotFoundError:
            print("WARNING: action {} not found!".format(i))
            continue

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
    print("writing {} entries into train".format(len(all_train)))
    np.save(train_np, np.array(weights, dtype=np.float32))
    for i, state in all_train:
        np.save(train_np, state)
        np.save(train_np, i)
    train_np.close()
    val_np = open(os.path.join(dataset_root, "{}_val.npy".format(agent_name)), "wb")
    print("writing {} entries into val".format(sum([len(x) for x in vals])))
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


def publish_models(model_dir, output_dir, val_dir):
    def _copy(file_name, new_name=None):
        if not new_name:
            new_name = file_name
        for ext in [".index", ".data-00000-of-00001"]:
            copyfile(
                os.path.join(model_dir, file_name + ext),
                os.path.join(output_dir, new_name + ext),
            )

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.isfile(os.path.join(model_dir, "model.index")):
        print("no models found!")
        raise FileNotFoundError
    else:
        _copy("model")
    for i, name in enumerate(sorted(glob(os.path.join(val_dir, "*.npy")))):
        _copy(
            "model_head_" + str(i).zfill(3), "model_head_" + os.path.basename(name)[:5]
        )


def draw_confusion(matrix_np, output_image):
    label_names = (
        ["HC-" + str(i) for i in range(1, 6)]
        + ["HN-" + str(i) for i in range(1, 6)]
        + ["P-" + str(i) for i in range(1, 6)]
        + ["D-" + str(i) for i in range(1, 6)]
    )
    matrix = np.load(matrix_np)
    (n, _) = matrix.shape
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(matrix, cmap=plt.cm.Greens)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    fig.tight_layout()
    plt.savefig(output_image)


if __name__ == "__main__":
    Fire()
