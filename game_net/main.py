from copy import deepcopy  # noqa F401
from fire import Fire
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_addons as tfa
from time import time
from tqdm import tqdm

from dataset import GAME_STATE_LENGTH
from dataset import DatasetGenerator_cached, DatasetGenerator_concat
from dataset import np2tf_generator, merged_np2tf_generator
from settings import model_config, classification_head_config
from naiveFC import NaiveFC

model = NaiveFC(**model_config)
heads = []
dummy_head = NaiveFC(**classification_head_config)
# initialize weights
_ = dummy_head.layers[0].get_weights()
for i in range(20):
    new_head = tf.keras.models.clone_model(dummy_head)
    heads.append(new_head)
with tf.device("/GPU:0"):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.008, beta_1=0.95, beta_2=0.9999
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
    # pred = tf.concat(pred, axis=0)
    # new_labels = tf.concat(new_labels, axis=0)
    # new_weights = tf.concat(new_weights, axis=0)
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
    epoch=400,
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
            for i in range(len(heads)):
                heads[i].load_weights(save_checkpoint + "_head_" + str(i).zfill(3))
        except:  # noqa E722
            print("cannot load existing model")
    num_heads = len(glob(os.path.join(dataset_root, "train", "*.npy")))
    trainset = (
        DatasetGenerator_cached(os.path.join(dataset_root, "train"), samples_per_batch)
        .shuffle(1 + int(shuffle * batch_size))
        .batch(batch_size)
        .prefetch(5)
    )
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    valset = (
        DatasetGenerator_concat(os.path.join(dataset_root, "val"))
        .cache()
        .batch(batch_size)
        .prefetch(5)
    )
    val_loss = [tf.keras.metrics.Mean(name="val_loss") for i in range(num_heads)]
    val_accuracy = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
        for i in range(num_heads)
    ]
    for e in range(epoch):
        t = time()
        for i, (state, agent_id, label, weights) in enumerate(trainset):
            with tf.GradientTape() as tape:
                feature = model(state, training=True)
                pred, new_label, new_weights = embedding_to_onehot(
                    feature, agent_id, label, weights, num_heads
                )
                new_label = tf.concat(new_label, axis=0)
                pred = tf.concat(pred, axis=0)
                new_weights = tf.concat(new_weights, axis=0)
                loss = loss_object(new_label, pred, sample_weight=new_weights)
                loss = loss_object(new_label, pred, sample_weight=new_weights)
            trainable_variables = (
                sum([h.trainable_variables for h in heads[:num_heads]], [])
                + model.trainable_variables  # noqa W503
            )
            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            train_loss(loss)
            train_accuracy(new_label, pred)
            if i % 500 == 0:
                if use_val:
                    for k, (state_t, agent_id_t, label_t, weights_t) in enumerate(
                        valset
                    ):
                        feature = model(state_t, training=False)
                        pred, new_label, new_weights = embedding_to_onehot(
                            feature, agent_id_t, label_t, weights_t, num_heads
                        )
                        for j in range(num_heads):
                            loss = loss_object(new_label[j], pred[j])
                            val_loss[j](loss)
                            val_accuracy[j](new_label[j], pred[j])
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
                for j in range(num_heads):
                    print(
                        "  val{}: ".format(str(j).zfill(2)),
                        "loss:",
                        round(float(val_loss[j].result()), 5),
                        "accuracy:",
                        round(float(val_accuracy[j].result()), 5),
                    )
                    val_loss[j].reset_states()
                    val_accuracy[j].reset_states()
                train_loss.reset_states()
                train_accuracy.reset_states()
                t = time()


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
        heads_checkpoints = [x[:-6] for x in glob(load_checkpoint + "_head_*.index")]
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
        .prefetch(5)
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
        .prefetch(5)
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
                    feature, agent_id, label, weight, num_heads + 1
                )
                val_accuracy(new_label[idx], pred[idx])
        return float(val_accuracy.result())

    model.load_weights(load_checkpoint)
    model.trainable = False
    num_heads = len(heads_checkpoints)
    if num_heads > 0:
        for i, h in enumerate(sorted(heads_checkpoints)):
            heads[i].load_weights(h)
        # pick best exisiting head
        accuracy = []
        for i in tqdm(list(range(num_heads))):
            accuracy.append(eval(i))
        best_id = max(range(num_heads), key=lambda x: accuracy[x])
        print(
            "using existing head {} with {} accuracy".format(best_id, accuracy[best_id])
        )
        pprint(accuracy)
        heads[0] = heads[best_id]
    else:
        best_id = 0
    with open(save_result, "wb") as fout:
        for i, _ in enumerate(tqdm(trainset)):
            for state, label, weight in trainset.take(i).repeat(epoch):
                with tf.device("/GPU:0"):
                    with tf.GradientTape() as tape:
                        feature = model(state, training=False)
                        pred, new_label, new_weights = embedding_to_onehot(
                            feature, tf.zeros(label.shape), label, weight, 1
                        )
                        loss = loss_object(new_label[0], pred[0])
                    gradients = tape.gradient(loss, heads[0].trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, heads[0].trainable_variables)
                    )
            np.save(fout, np.array([i * step, eval(0)], dtype=np.float32))
            if i % 20 == 0 and save_head != "":
                heads[0].save_weights(save_head)


if __name__ == "__main__":
    Fire()
