from fire import Fire
import tensorflow as tf
import tensorflow_addons as tfa

from dataset import DatasetGenerator
from naiveFC import NaiveFC

model = NaiveFC(2, num_layers=1)
with tf.device("/GPU:0"):
    loss_func = tfa.losses.TripletSemiHardLoss()
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
    save_model_to=None,
    sample_per_epoch=400,
    epoch=10,
    batch_size=500,
    model_type="NaiveFC",
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


if __name__ == "__main__":
    Fire()
