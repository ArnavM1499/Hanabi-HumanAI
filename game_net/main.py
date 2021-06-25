from fire import Fire
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
from sklearn.decomposition import PCA
from tqdm import tqdm

from dataset import DatasetGenerator, np2tf_generator
from naiveFC import NaiveFC, ResFC

# model = NaiveFC(8, num_layers=4, activation="relu")
model = ResFC(8, num_layers=8, activation="relu")
with tf.device("/GPU:0"):
    loss_func = tfa.losses.TripletSemiHardLoss()
    optimizer = tf.keras.optimizers.Adam()


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
    validation_root,
    save_checkpoint,
    sample_per_epoch=4000,
    epoch=10,
    batch_size=100,
):
    trainset = tf.data.Dataset.range(4).interleave(
        lambda _: DatasetGenerator(train_root, sample_per_epoch)
        .shuffle(1)
        .batch(batch_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    valset = tf.data.Dataset.range(4).interleave(
        lambda _: DatasetGenerator(validation_root, sample_per_epoch).batch(batch_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    distance_positive = tf.keras.metrics.Mean(name="distance_positive")
    distance_negative = tf.keras.metrics.Mean(name="distance_negative")
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_positive = tf.keras.metrics.Mean(name="val_positive")
    val_negative = tf.keras.metrics.Mean(name="val_negative")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    for e in range(epoch):
        for i, (state, state_t) in enumerate(zip(trainset, valset)):
            with tf.device("/GPU:0"):
                with tf.GradientTape() as tape:
                    feature = model(state, training=True)
                    loss, distance_positives, distance_negatives = calculate_loss(
                        feature
                    )
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(loss)
                distance_positive(distance_positives)
                distance_negative(distance_negatives)
                feature = model(state_t, training=False)
                loss, distance_positives, distance_negatives = calculate_loss(feature)
                val_loss(loss)
                val_positive(tf.math.reduce_mean(distance_positives))
                val_negative(tf.math.reduce_mean(distance_negatives))
                if i % 10 == 0:
                    model.save_weights(save_checkpoint)
                    print("=" * 20)
                    print("epoch: ", e, "iter: ", i)
                    print(
                        "  train: ",
                        "loss:",
                        round(float(train_loss.result()), 5),
                        "positive:",
                        round(float(distance_positive.result()), 5),
                        "negative:",
                        round(float(distance_negative.result()), 5),
                    )
                    print(
                        "  val: ",
                        "loss:",
                        round(float(val_loss.result()), 5),
                        "positive:",
                        round(float(val_positive.result()), 5),
                        "negative:",
                        round(float(val_negative.result()), 5),
                    )
                    train_loss.reset_states()
                    distance_positive.reset_states()
                    distance_negative.reset_states()
                    val_loss.reset_states()
                    val_positive.reset_states()
                    val_negative.reset_states()


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
