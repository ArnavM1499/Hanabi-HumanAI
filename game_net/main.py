from fire import Fire
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
from sklearn.decomposition import PCA
from time import time
from tqdm import tqdm

from dataset import DatasetGenerator2, np2tf_generator
from naiveFC import NaiveFC

# model = NaiveFC(8, num_layers=4, activation="relu")
model = NaiveFC(10, num_layers=4, num_units=1000, activation="relu")
heads = [NaiveFC(20, num_layers=0, activation="relu", last="softmax") for _ in range(8)]
with tf.device("/GPU:0"):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


@tf.function
def embedding_to_onehot(features, agent_ids, labels):
    # features has shape [batch_size, embedding_size]
    # agent_ids has shape [batch_size]
    # lables has shape [batch_size]
    pred = []
    new_labels = []
    for i, head in enumerate(heads):
        mask = tf.math.equal(agent_ids, i)
        output = head(features)
        pred.append(tf.boolean_mask(output, mask))
        new_labels.append(tf.boolean_mask(labels, mask))
    pred = tf.concat(pred, axis=0)
    new_labels = tf.concat(new_labels, axis=0)
    # pred has shape [batch_size, 20]
    return pred, new_labels


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
    train_root,
    save_checkpoint,
    epoch=100,
    batch_size=6000,
    shuffle=3,
):
    trainset = tf.data.Dataset.range(1).interleave(
        lambda _: DatasetGenerator2(train_root, 0)
        .shuffle(1 + int(shuffle * batch_size))
        .batch(batch_size),
        num_parallel_calls=4,
        deterministic=False,
    )
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    feature_std = tf.keras.metrics.Mean(name="feature_std")
    F = tf.keras.layers.Flatten()
    for e in range(epoch):
        t = time()
        for i, (state, agent_id, label) in enumerate(trainset):
            with tf.device("/GPU:0"):
                with tf.GradientTape() as tape:
                    feature = model(F(state), training=True)
                    pred, new_label = embedding_to_onehot(feature, agent_id, label)
                    loss = loss_object(new_label, pred)
                trainable_variables = (
                    sum([h.trainable_variables for h in heads], [])
                    + model.trainable_variables
                )
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))
                train_loss(loss)
                feature_std(tf.math.reduce_mean(tf.math.reduce_std(feature, axis=0)))
                train_accuracy(new_label, pred)
                if i % 10 == 0:
                    model.save_weights(save_checkpoint)
                    print("=" * 20)
                    print("epoch: ", e, "iter: ", i)
                    print(
                        "  train: ",
                        "loss:",
                        round(float(train_loss.result()), 5),
                        "accuracy:",
                        round(float(train_accuracy.result()), 5),
                        "feature std:",
                        round(float(feature_std.result()), 5),
                        "sec passed:",
                        round(time() - t, 5),
                    )
                    t = time()
                    train_loss.reset_states()
                    train_accuracy.reset_states()


def vis(test_dir, load_checkpoint, save_image, num_samples=-1, use_pca=True):
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
        transform = lambda x: pca.transform(x)
    else:
        transform = lambda x: np.array(x)
    for i, sample in enumerate(samples):
        points = transform(sample)
        print("#{}: {}".format(i, np.array(tf.math.reduce_mean(sample, axis=0))))
        plt.scatter(points[:, 0], points[:, 1], s=1, c=colors[i % 4][i // 4])
    plt.savefig(save_image, dpi=1000)


if __name__ == "__main__":
    Fire()
