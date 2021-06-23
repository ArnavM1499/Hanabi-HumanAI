from fire import Fire
import tensorflow as tf
import tensorflow_addons as tfa

from dataset import DatasetGenerator
from naiveFC import NaiveFC

model = NaiveFC(num_layers=2)
with tf.device("/GPU:0"):
    loss_func = tfa.losses.TripletSemiHardLoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def train(
    dataset_root,
    save_model_to,
    sample_per_epoch=4000,
    epoch=1,
    batch_size=500,
    model_type="NaiveFC",
):
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    distance_positive = tf.keras.metrics.Mean(name="distance_positive")
    distance_negative = tf.keras.metrics.Mean(name="distance_negative")
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    for e in range(epoch):
        DataLoader = tf.data.Dataset.range(4).interleave(
            lambda _: DatasetGenerator(dataset_root, sample_per_epoch, batch_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        for i, (label, state) in enumerate(DataLoader):
            with tf.GradientTape() as tape:
                feature = model(state.gpu())
                loss = loss_func(label, feature.gpu())
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # calculate distance between samples
            label = tf.reshape(label, [batch_size, 1])
            distance_matrix = tfa.losses.metric_learning.pairwise_distance(
                feature, squared=False
            )
            adjacency = tf.math.equal(label, tf.transpose(label))
            adjacency_not = tf.math.logical_not(adjacency)
            mask_positives = tf.cast(
                adjacency, dtype=tf.dtypes.float32
            ) - tf.linalg.diag(tf.ones([batch_size]))
            mask_negatives = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
            distance_positives = tf.math.reduce_mean(
                tf.math.multiply(distance_matrix, mask_positives)
            )
            distance_negatives = tf.math.reduce_mean(
                tf.math.multiply(distance_matrix, mask_negatives)
            )
            train_loss(loss)
            distance_positive(distance_positives)
            distance_negative(distance_negatives)
            if (i + 1) % 10 == 0:
                print(
                    "epoch:",
                    e,
                    "iter:",
                    i + 1,
                    "loss",
                    float(train_loss.result()),
                    "dis_pos",
                    float(distance_positive.result()),
                    "dis_neg",
                    float(distance_negative.result()),
                )
                train_loss.reset_states()
                distance_positive.reset_states()
                distance_negative.reset_states()


if __name__ == "__main__":
    Fire()
