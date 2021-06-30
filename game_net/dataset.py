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


def merged_np2tf_generator(file_path):
    f = open(file_path, "rb")
    while True:
        try:
            state = np.load(f)
            action = np.load(f)
            if len(state) == GAME_STATE_LENGTH:
                yield tf.constant(state * 0.333, dtype=tf.float32), action[0]
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
    def __new__(cls, dataset_root, dummy=0):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(GAME_STATE_LENGTH,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),  # Agent id
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Action id
            ),
            args=(dataset_root, dummy),
        )

    def _generator(dataset_root, dummy=0):

        dataset_root = str(dataset_root)[2:-1]
        all_agents = [
            os.path.join(dataset_root, pid) for pid in os.listdir(dataset_root)
        ]
        file_generators = [merged_np2tf_generator(pid) for pid in all_agents]
        num_agents = len(file_generators)
        pos = 0
        visited = set()
        while True:
            try:
                # game state, agent_id, action_id
                state, action = next(file_generators[pos])
                yield state, pos, action
                pos = (pos + 1) % num_agents
            except StopIteration:
                file_generators[pos] = merged_np2tf_generator(all_agents[pos])
                if pos not in visited:
                    visited.add(pos)
                    if len(visited) == num_agents:
                        break


def _3decode(encoding):
    state = []
    for i in range(31):
        state.append(encoding % 3)
        encoding = encoding // 3
    return tf.stack(state)


def _parse_3encode(encoding):
    encoding = tf.map_fn(
        lambda x: tf.strings.to_number(x, out_type=tf.int64),
        tf.strings.split(encoding),
        fn_output_signature=tf.int64,
    )
    action = float(encoding[1])
    agent = encoding[0]
    states = tf.concat(tf.map_fn(_3decode, encoding[2:]), axis=1)
    states = tf.map_fn(
        lambda x: 0.333 * float(x), states, fn_output_signature=tf.float32
    )
    return states, agent, action


def txt_to_dataset(text_files):
    return tf.data.TextLineDataset(text_files).map(_parse_3encode)


def serialize_example(state, agent_id, action_id):
    feature = {
        "state": tf.train.Feature(float_list=tf.train.FloatList(value=state)),
        "agent_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[agent_id])),
        "action_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[action_id])),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()


def tf_serialize_example(state, agent_id, action_id):
    tf_string = tf.py_function(
        serialize_example, (state, agent_id, action_id), tf.string
    )
    return tf.reshape(tf_string, ())


def np_to_tfrecord(dataset_root, output_record, num_samples):
    # num_samples used to manually balence sample weights
    original = DatasetGenerator(dataset_root, num_samples)
    serialized = original.map(tf_serialize_example)
    writer = tf.data.experimental.TFRecordWriter(output_record)
    writer.write(serialized)


game_state_discription = {
    "state": tf.io.FixedLenFeature([], tf.float32),
    "agent_id": tf.io.FixedLenFeature([], tf.int64),
    "action_id": tf.io.FixedLenFeature([], tf.int64),
}


def parse_state_func(proto):
    return tf.io.parse_single_example(proto, game_state_discription)


def get_dataset_from_tfrecord(tfrecord):
    raw = tf.data.TFRecordDataset([tfrecord])
    for data in raw:
        import pdb

        pdb.set_trace()


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
