from fire import Fire
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_addons as tfa
from time import time
from tqdm import tqdm

from dataset import GAME_STATE_LENGTH
from dataset import DatasetGenerator_cached, DatasetGenerator2
from dataset import np2tf_generator, merged_np2tf_generator
from naiveFC import NaiveFC

GPU = True

if GPU:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf.config.set_visible_devices([], "GPU")

# model = NaiveFC(20, num_units=[600, 400, 200], activation="relu", dropout=0)
model = NaiveFC(20, num_units=[800, 800, 800, 800], activation="relu", dropout=0)
heads = [
    NaiveFC(20, num_units=[20], activation="relu", last="softmax") for _ in range(30)
]
with tf.device("/GPU:0"):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.995, beta_2=0.9999
    )


@tf.function
def embedding_to_onehot(features, agent_ids, labels, weights, num_heads):
    # features has shape [batch_size, embedding_size]
    # agent_ids has shape [batch_size]
    # lables has shape [batch_size]
    # weights has shape [batch_size]
    pred = []
    new_labels = []
    new_weights = []
    for i, head in enumerate(heads[:num_heads]):
        mask = tf.math.equal(agent_ids, i)
        output = head(features)
        pred.append(tf.boolean_mask(output, mask))
        new_labels.append(tf.boolean_mask(labels, mask))
        new_weights.append(tf.boolean_mask(weights, mask))
    pred = tf.concat(pred, axis=0)
    new_labels = tf.concat(new_labels, axis=0)
    new_weights = tf.concat(new_weights, axis=0)
    # pred has shape [batch_size, 20]
    return pred, new_labels, new_weights


@tf.function
def calculate_loss(features):
    # features has shape [batch_size, 20, feature_dim]
    positive_variance = tf.math.reduce_euclidean_norm(
        tf.math.reduce_std(features, axis=0), axis=1
    )
    negative_distances = tf.reshape(
        tf.boolean_mask(
            tfa.losses.metric_learning.pairwise_distance(
                tf.math.reduce_mean(features, axis=0), squared=False
            ),
            tf.equal(tf.eye(20), 0),
        ),
        (20, 19),
    )
    negative_distances_mean = tf.reduce_mean(tf.minimum(0.5, negative_distances))
    loss = tf.math.reduce_sum(
        tf.math.maximum(0.0, 0.5 + 3.0 * positive_variance - negative_distances_mean)
    )
    return loss, positive_variance, negative_distances_mean


def train(
    dataset_root,
    save_checkpoint_dir,
    epoch=100,
    samples_per_batch=14000000,  # default for 5 + 3 train
    batch_size=10000,
    shuffle=5,
    use_val=True,
    start_from_scratch=False,
):
    save_checkpoint = os.path.join(save_checkpoint_dir, "model")
    if not start_from_scratch:
        try:
            model.load_weights(save_checkpoint)
            for i in range(20):
                heads[i].load_weights(save_checkpoint + "_head_" + str(i).zfill(3))
        except:  # noqa E722
            print("cannot load existing model")
    num_heads = len(glob(os.path.join(dataset_root, "train", "*.npy")))
    # trainset = (
    #     DatasetGenerator2(os.path.join(dataset_root, "train"), 0)
    #     .shuffle(1 + int(shuffle * batch_size))
    #     .batch(batch_size)
    #     .prefetch(1)
    # )
    trainset = (
        DatasetGenerator_cached(os.path.join(dataset_root, "train"), samples_per_batch)
        .shuffle(1 + int(shuffle * batch_size))
        .batch(batch_size)
        .prefetch(2)
    )
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    valset = (
        DatasetGenerator2(os.path.join(dataset_root, "val"), 0)
        .cache()
        .batch(batch_size)
        .prefetch(2)
    )
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
    for e in range(epoch):
        t = time()
        for i, (state, agent_id, label, weights) in enumerate(trainset):
            with tf.device("/GPU:0"):
                with tf.GradientTape() as tape:
                    feature = model(state, training=True)
                    pred, new_label, new_weights = embedding_to_onehot(
                        feature, agent_id, label, weights, num_heads
                    )
                    loss = loss_object(new_label, pred, sample_weight=new_weights)
                trainable_variables = (
                    sum([h.trainable_variables for h in heads[:num_heads]], [])
                    + model.trainable_variables  # noqa W503
                )
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))
                train_loss(loss)
                train_accuracy(new_label, pred)
                if i % 50 == 0:
                    if use_val:
                        for state_t, agent_id_t, label_t, weights_t in valset:
                            feature = model(state_t, training=False)
                            pred, new_label, new_weights = embedding_to_onehot(
                                feature, agent_id_t, label_t, weights_t, num_heads
                            )
                            loss = loss_object(
                                new_label, pred, sample_weight=new_weights
                            )
                            val_loss(loss)
                            val_accuracy(new_label, pred)
                        for j, h in enumerate(heads[:num_heads]):
                            h.save_weights(save_checkpoint + "_head_" + str(j).zfill(3))
                    model.save_weights(save_checkpoint)
                    print("=" * 40)
                    print("epoch: ", e, "iter: ", i)
                    print(
                        "  train: ",
                        "loss:",
                        round(float(train_loss.result()), 5),
                        "accuracy:",
                        round(float(train_accuracy.result()), 5),
                        "sec passed:",
                        round(time() - t, 5),
                    )
                    print(
                        "  val: ",
                        "loss:",
                        round(float(val_loss.result()), 5),
                        "accuracy:",
                        round(float(val_accuracy.result()), 5),
                    )
                    t = time()
                    train_loss.reset_states()
                    train_accuracy.reset_states()
                    val_loss.reset_states()
                    val_accuracy.reset_states()


def vis(test_dir, load_checkpoint, save_image, num_samples=-1, use_pca=True):
    # deprecated
    colors = [
        ["#d61a1a", "#d67b1a", "#d6b41a", "#d0d61a", "#9ed61a"],
        ["#1ad620", "#1ad6b4", "#1ac6d6", "#1a9ed6", "#1a6bd6"],
        ["#1a20d6", "#591ad6", "#8e1ad6", "#c31ad6", "#d61aad"],
        ["#d61a68", "#8a7e6d", "#6d8a71", "#6d7f8a", "#886d8a"],
    ]
    model.load_weights(load_checkpoint)
    samples = []
    for i in range(20):
        samples.append([])
        data = np2tf_generator(
            os.path.join(test_dir, "{}.npy".format(str(i).zfill(2))),
            int(num_samples),
            loop=False,
        )
        batch = []
        cnt = 0
        with tf.device("/GPU:0"):
            for state in tqdm(data):
                batch.append(state)
                cnt += 1
                if cnt > 500:
                    samples[-1].extend(model(tf.stack(batch), training=False).numpy())
                    cnt = 0
                    batch = []
            samples[-1].extend(model(tf.stack(batch), training=False).numpy())
    if use_pca:
        pca = PCA(n_components=2)
        pca.fit(np.concatenate(samples))
        transform = lambda x: pca.transform(x)  # noqa E731
    else:
        transform = lambda x: np.array(x)  # noqa E731
    for i, sample in enumerate(samples):
        points = transform(sample)
        print("#{}: {}".format(i, np.array(tf.math.reduce_mean(sample, axis=0))))
        plt.scatter(points[:, 0], points[:, 1], s=1, c=colors[i % 4][i // 4])
    plt.savefig(save_image, dpi=1000)


def transfer(
    train_file,
    val_file,
    load_checkpoint_dir,
    save_result,
    load_heads=True,
    save_head="",
    step=10,
    learning_rate=0.001,
    epoch=5,
    max_samples=4000,
):
    load_checkpoint = os.path.join(load_checkpoint_dir, "model")
    if load_heads:
        heads_checkpoints = [x[:-6] for x in glob(load_checkpoint + "_head_*")]
    else:
        heads_checkpoints = []
    print(len(heads_checkpoints), " heads found")
    trainset = (
        tf.data.Dataset.from_generator(
            lambda: merged_np2tf_generator(train_file),
            output_signature=(
                tf.TensorSpec(shape=(GAME_STATE_LENGTH,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Action id
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Weight
            ),
        )
        .take(max_samples)
        .cache()
        .batch(step)
        .prefetch(2)
    )
    # dummmy traverse to enable cache
    for _ in trainset:
        pass
    valset = (
        tf.data.Dataset.from_generator(
            lambda: merged_np2tf_generator(val_file),
            output_signature=(
                tf.TensorSpec(shape=(GAME_STATE_LENGTH,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Action id
                tf.TensorSpec(shape=(), dtype=tf.float32),  # Weight
            ),
        )
        .cache()
        .batch(10000)
        .prefetch(2)
    )
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def eval(idx):
        val_accuracy.reset_states()
        for state, label, weight in valset:
            agent_id = tf.broadcast_to(idx, label.shape)
            with tf.device("/GPU:0"):
                feature = model(state, training=False)
                pred, new_label, new_weights = embedding_to_onehot(
                    feature, agent_id, label, weight, idx + 1
                )
                val_accuracy(new_label, pred)
        return float(val_accuracy.result())

    model.load_weights(load_checkpoint)
    model.trainable = False
    num_heads = len(heads_checkpoints)
    if num_heads > 0:
        for i, h in enumerate(heads_checkpoints):
            heads[i].load_weights(h)
        # pick best exisiting head
        accuracy = []
        for i in tqdm(list(range(num_heads))):
            accuracy.append(eval(num_heads - 1))
        best_id = max(range(num_heads), key=lambda x: accuracy[x])
        heads[0] = heads[best_id]
    with open(save_result, "wb") as fout:
        for i, _ in enumerate(tqdm(trainset)):
            for state, label, weight in trainset.take(i).repeat(epoch):
                with tf.device("/GPU:0"):
                    with tf.GradientTape() as tape:
                        feature = model(state, training=False)
                        pred, new_label, new_weights = embedding_to_onehot(
                            feature, tf.zeros(label.shape), label, weight, 1
                        )
                        loss = loss_object(new_label, pred)
                    gradients = tape.gradient(loss, heads[0].trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, heads[0].trainable_variables)
                    )
            np.save(fout, np.array([i * step, eval(0)], dtype=np.float32))
            if i % 20 == 0 and save_head != "":
                heads[0].save_weights(save_head)


if __name__ == "__main__":
    Fire()
