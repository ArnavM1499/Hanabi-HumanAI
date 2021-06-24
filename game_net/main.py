from fire import Fire
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import os
from tqdm import tqdm

from dataset import DatasetGenerator, np2tf_generator
from naiveFC import NaiveFC

model = NaiveFC(2, num_layers=3)
with tf.device("/GPU:0"):
    loss_func = tfa.losses.TripletHardLoss(soft=True)
    optimizer = tf.keras.optimizers.Adam()


@tf.function
def calculate_distance(label, feature, batch_size):
    label = tf.reshape(label, [batch_size, 1])
    distance_matrix = tfa.losses.metric_learning.pairwise_distance(
        feature, squared=False
    )
    adjacency = tf.math.equal(label, tf.transpose(label))
    adjacency_not = tf.math.logical_not(adjacency)
    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )
    mask_negatives = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    distance_positives = tf.math.reduce_mean(
        tf.math.multiply(distance_matrix, mask_positives)
    )
    distance_negatives = tf.math.reduce_mean(
        tf.math.multiply(distance_matrix, mask_negatives)
    )
    return distance_positives, distance_negatives


def train(
    train_root,
    validation_root,
    save_checkpoint,
    sample_per_epoch=400,
    epoch=10,
    batch_size=1000,
):
    trainset = tf.data.Dataset.range(4).interleave(
        lambda _: DatasetGenerator(train_root, sample_per_epoch, batch_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    valset = tf.data.Dataset.range(4).interleave(
        lambda _: DatasetGenerator(validation_root, sample_per_epoch, batch_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    distance_positive = tf.keras.metrics.Mean(name="distance_positive")
    distance_negative = tf.keras.metrics.Mean(name="distance_negative")
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_positive = tf.keras.metrics.Mean(name="val_positive")
    val_negative = tf.keras.metrics.Mean(name="val_negative")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    for e in range(epoch):
        for i, ((label, state), (label_t, state_t)) in enumerate(zip(trainset, valset)):
            with tf.device("/GPU:0"):
                with tf.GradientTape() as tape:
                    feature = model(state, training=True)
                    loss = loss_func(label, feature)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                distance_positives, distance_negatives = calculate_distance(
                    label, feature, batch_size
                )
                train_loss(loss)
                distance_positive(distance_positives)
                distance_negative(distance_negatives)
                feature = model(state_t, training=False)
                loss = loss_func(label_t, feature)
                distance_positives, distance_negatives = calculate_distance(
                    label_t, feature, batch_size
                )
                val_loss(loss)
                val_positive(distance_positives)
                val_negative(distance_negatives)
                if i % 10 == 0:
                    model.save(save_checkpoint)
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


def vis(test_dir, load_checkpoint, save_image, num_samples=-1):
    model = tf.keras.models.load_model(load_checkpoint)
    markers = ["o", "X", "^", "P"]
    colors = ["r", "g", "b", "c", "m"]
    for i in [2]:
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
                    point = model(tf.stack(batch), training=False).numpy()
                    plt.scatter(
                        point[:, 0],
                        point[:, 1],
                        s=3,
                        marker=markers[i % 4],
                        c=colors[i // 4],
                    )
                    cnt = 0
                    batch = []
            point = model(tf.stack(batch), training=False).numpy()
            plt.scatter(
                point[:, 0],
                point[:, 1],
                s=3,
                marker=markers[i % 4],
                c=colors[i // 4],
            )
    plt.savefig(save_image)


if __name__ == "__main__":
    Fire()
