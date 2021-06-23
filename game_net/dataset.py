from glob import glob
import numpy as np
import os
from random import shuffle
import tensorflow as tf


def np2tf_generator(file_path, num_samples=-1):
    cnt = 0
    f = open(file_path, "rb")
    while num_samples < 0 or cnt < num_samples:
        try:
            state = np.load(f)
            cnt += 1
            yield tf.constant(state, dtype=tf.float32)
        except ValueError:
            f.close()
            f = open(file_path, "rb")
    f.close()


def np2tf_generator_randomized(file_path, num_samples=-1, cache_size=20):
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
    def __new__(cls, dataset_root, num_samples, batch_size=60):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(batch_size,), dtype=tf.int8),
                tf.TensorSpec(shape=(batch_size, 287), dtype=tf.float32),
            ),
            args=(dataset_root, num_samples, batch_size),
        )

    def _generator(dataset_root, num_samples, batch_size):

        dataset_root = str(dataset_root)[2:-1]
        file_generators = [
            [
                np2tf_generator_randomized(f, num_samples * 2)
                for f in glob("{}/{}/*.npy".format(dataset_root, pid))
            ]
            for pid in os.listdir(dataset_root)
        ]
        file_generators = [x for x in file_generators if x != []]
        num_agents = len(file_generators)
        num_generators = [len(g) for g in file_generators]

        for idx in range(num_samples):
            pos = idx % num_agents
            labels = []
            data = []
            for i in range(batch_size):
                label = i % num_generators[pos]
                labels.append(tf.constant(label, dtype=tf.int8))
                data.append(next(file_generators[pos][label]))
            yield tf.stack(labels), tf.stack(data)
