from glob import glob
import numpy as np
import os
from random import shuffle
import tensorflow as tf

GAME_STATE_LENGTH = 558
CACHE_QUOTA = 24 << 30  # about 24G


def np2tf_generator(file_path, num_samples=-1, loop=True):
    cnt = 0
    f = open(file_path, "rb")
    while num_samples < 0 or cnt < num_samples:
        try:
            state = np.load(f)
            cnt += 1
            if len(state) == GAME_STATE_LENGTH:
                yield tf.constant(state * 0.333, dtype=tf.float32)
        except ValueError:
            f.close()
            if loop:
                f = open(file_path, "rb")
            else:
                break
    f.close()


def merged_np2tf_generator(file_path, idx=None):
    f = open(file_path, "rb")
    weights = np.load(f).tolist()
    while True:
        try:
            state = np.load(f)
            action = np.load(f)
            if len(state) == GAME_STATE_LENGTH:
                if idx is not None:
                    yield (
                        tf.constant(state * 0.333, dtype=tf.float32),
                        idx,
                        action,
                        weights[action],
                    )
                else:
                    yield (
                        tf.constant(state * 0.333, dtype=tf.float32),
                        action,
                        weights[action],
                    )
        except ValueError:
            break
    f.close()


def np2tf_generator_randomized(file_path, num_samples=-1, cache_size=200):
    G = np2tf_generator(file_path)
    cnt = 0
    cache = []
    while num_samples < 0 or cnt < num_samples:
        for _ in range(cache_size):
            cache.append(next(G))
        shuffle(cache)
        for _ in range(cache_size):
            yield cache.pop()


class DatasetGenerator:
    def __new__(cls, dataset_root, num_samples):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(GAME_STATE_LENGTH,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),  # Agent id
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Action id
            ),
            args=(dataset_root, num_samples),
        )

    def _generator(dataset_root, num_samples):

        dataset_root = str(dataset_root)[2:-1]
        file_generators = [
            [
                np2tf_generator(
                    "{}/{}/{}.npy".format(dataset_root, pid, str(i).zfill(2)),
                    num_samples,
                )
                for i in range(20)
            ]
            for pid in os.listdir(dataset_root)
        ]
        file_generators = [x for x in file_generators if x != []]
        num_agents = len(file_generators)

        for idx in range(num_samples // 20):
            pos = idx % num_agents
            for i in range(20):
                # game state, agent_id, action_id
                yield next(file_generators[pos][i]), pos, i


class DatasetGenerator2:
    def __new__(cls, dataset_root, filter_size=0):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(GAME_STATE_LENGTH,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),  # Agent id
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Action id
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Weight
            ),
            args=(dataset_root, filter_size),
        )

    def _generator(dataset_root, filter_size=0):

        dataset_root = str(dataset_root)[2:-1]
        all_agents = glob(os.path.join(dataset_root, "*.npy"))
        if filter_size > 1000:
            all_agents = [x for x in all_agents if os.path.getsize(x) > filter_size]
        all_agents.sort()
        file_generators = [merged_np2tf_generator(pid) for pid in all_agents]
        num_agents = len(file_generators)
        pos = 0
        visited = set()
        while True:
            try:
                # game state, agent_id, action_id, weight
                state, action, weight = next(file_generators[pos])
                yield state, pos, action, weight
                pos = (pos + 1) % num_agents
            except StopIteration:
                file_generators[pos] = merged_np2tf_generator(all_agents[pos])
                if pos not in visited:
                    visited.add(pos)
                    if len(visited) == num_agents:
                        break


class merged_tf2np_wrapper:
    def __new__(cls, dataset_file, idx=0):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(GAME_STATE_LENGTH,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),  # Agent id
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Action id
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Weight
            ),
            args=(dataset_file, idx),
        )

    def _generator(dataset_file, idx):
        return merged_np2tf_generator(dataset_file, idx)


def DatasetGenerator_cached(dataset_root, num_samples, max_cache=-1):
    all_datasets = []
    for i, np_file in enumerate(sorted(glob(os.path.join(dataset_root, "*.npy")))):
        all_datasets.append(
            (
                merged_tf2np_wrapper(np_file, i),
                os.path.getsize(np_file) * 4,  # estimated size of all data
            )
        )
    num_dataset = len(all_datasets)
    num_samples_individual = num_samples // num_dataset + 1
    if max_cache < 0:
        max_cache = CACHE_QUOTA
    for i in sorted(range(num_dataset), key=lambda j: all_datasets[j][1]):
        if all_datasets[i][1] < max_cache:
            max_cache -= all_datasets[i][1]
            all_datasets[i] = all_datasets[i][0].cache()
            #  initialize cache
            for _ in all_datasets[i]:
                pass
        else:
            all_datasets[i] = all_datasets[i][0]
    all_datasets = [d.repeat() for d in all_datasets]
    choice_dataset = tf.data.Dataset.range(num_dataset).repeat(num_samples_individual)
    return tf.data.experimental.choose_from_datasets(all_datasets, choice_dataset)


def DatasetGenerator_concat(dataset_root):
    all_datasets = []
    for i, np_file in enumerate(sorted(glob(os.path.join(dataset_root, "*.npy")))):
        all_datasets.append(merged_tf2np_wrapper(np_file, i))
    assert all_datasets != []
    for i in range(1, len(all_datasets)):
        all_datasets[0] = all_datasets[0].concatenate(all_datasets[i])
    return all_datasets[0]
