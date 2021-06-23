from fire import Fire
import tensorflow as tf
import tensorflow_addons as tfa

from dataset import DatasetGenerator
from naiveFC import NaiveFC

model = NaiveFC()
with tf.device("/GPU:0"):
    loss_func = tfa.losses.TripletHardLoss()
    optimizer = tf.keras.optimizers.Adam()


def train(
    dataset_root,
    save_model_to,
    sample_per_epoch=10000,
    epoch=3,
    batch_size=1000,
    model_type="NaiveFC",
):
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
            train_loss(loss)
            if i % 50 == 1:
                print("epoch:", e, "iter:", i, "loss", train_loss.result())
        train_loss.reset()


if __name__ == "__main__":
    Fire()
