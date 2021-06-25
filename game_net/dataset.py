from glob import glob
import numpy as np
import os
from random import shuffle
import tensorflow as tf


def np2tf_generator(file_path, num_samples=-1, loop=True):
    cnt = 0
    f = open(file_path, "rb")
    while num_samples < 0 or cnt < num_samples:
        try:
            state = np.load(f)
            cnt += 1
            yield tf.constant(state, dtype=tf.float32)
        except ValueError:
            f.close()
            if loop:
                f = open(file_path, "rb")
            else:
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
            output_signature=tf.TensorSpec(shape=(20, 287), dtype=tf.float32),
            args=(dataset_root, num_samples),
        )

    def _generator(dataset_root, num_samples):

        dataset_root = str(dataset_root)[2:-1]
        file_generators = [
            [
                np2tf_generator_randomized(f, num_samples)
                for f in glob("{}/{}/*.npy".format(dataset_root, pid))
            ]
            for pid in os.listdir(dataset_root)
        ]
        file_generators = [x for x in file_generators if x != []]
        num_agents = len(file_generators)

        for idx in range(num_samples):
            pos = idx % num_agents
            yield tf.stack(next(zip(*file_generators[pos])))
