from glob import glob
import numpy as np
import os
from random import shuffle
import tensorflow as tf

GAME_STATE_LENGTH = 558


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
                tf.TensorSpec(shape=(), dtype=tf.int8),  # Agent id
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
                yield next(file_generators[pos][i]), pos, i


def get_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000),
    )
