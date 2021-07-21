from fire import Fire
from glob import glob
import os
import tensorflow as tf
import tensorflow_addons as tfa
from time import time
from tqdm import tqdm

from dataset import GAME_STATE_LENGTH
from dataset import DatasetGenerator_cached, DatasetGenerator2
from naiveFC import NaiveFC

GPU = True
NUM_AGENTS = 8

if GPU:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf.config.set_visible_divices([], "GPU")

model = NaiveFC(20, num_units=[600, 400, 200], activation="relu")
head = NaiveFC(20, num_units=[20, 20], activation="relu")
transfer_head = NaiveFC(20, num_units=[20, 20], activation="relu", last="softmax")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.995, beta_2=0.9999)


@tf.function
def agentwise_loss(features, agent_ids, labels):
    losses = []
    for i in range(NUM_AGENTS):
        mask = tf.math.equal(agent_ids, i)
        agent_labels = tf.boolean_mask(labels, mask)
        agent_features = tf.boolean_mask(features, mask)
        agent_features = tf.math.l2_normalize(agent_features, 1)
        feature_similarity = tf.matmul(agent_features, agent_features, transpose_b=True)
        losses.append(tfa.losses.npair_loss(agent_labels, feature_similarity))
    return sum(losses)


@tf.function
def train(
    dataset_root,
    transfer_root,
    save_checkpoint_dir,
    epoch=500,
    samples_per_batch=14000000,  # default for 5 + 3 train
    batch_size=10000,
    transfer_epoch=5,
    transfer_samples=3000,
    shuffle=5,
    start_from_scratch=False,
):
    save_checkpoint = os.path.join(save_checkpoint_dir, "model")
    if not start_from_scratch:
        try:
            model.load_weights(save_checkpoint)
        except:  # noqa E722
            print("cannot load existing model")

    trainset = (
        DatasetGenerator_cached(os.path.join(dataset_root, "train"), samples_per_batch)
        .shuffle(1 + int(shuffle * batch_size))
        .batch(batch_size)
        .prefetch(2)
    )
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    transfer_trainset = (
        DatasetGenerator2(os.path.join(transfer_root, "train"), 0)
        .take(transfer_samples)
        .cache()
        .batch(min(transfer_samples, batch_size))
        .prefetch(2)
    )
    transfer_valset = (
        DatasetGenerator2(os.path.join(transfer_root, "val"), 0)
        .cache()
        .batch(batch_size)
        .prefetch(2)
    )
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    transfer_train_loss = tf.keras.metrics.Mean(name="trainsfer_train_loss")
    transfer_val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="transfer_val_accuracy"
    )
    transfer_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    def eval_transfer():
        transfer_val_accuracy.reset_states()
        for state, label, weight in transfer_valset:
            feature = model(state, training=False)
            pred = transfer_head(feature, training=False)
            transfer_val_accuracy(label, pred)
        return float(transfer_val_accuracy.result())

    for e in range(epoch):
        t = time()
        for i, (state, agent_id, label, weights) in enumerate(trainset):
            with tf.GradientTape() as tape:
                feature = model(state, training=True)
                feature = head(feature, training=True)
                loss = agentwise_loss(feature, agent_id, label)
            trainable_variables = model.trainable_variables + head.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            train_loss(loss)
            if i % 50 == 0:
                transfer_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                for f in range(transfer_epoch):
                    for j, (state_t, agent_id_t, label_t, weight_t) in enumerate(
                        transfer_trainset
                    ):
                        with tf.GradientTape() as tape_t:
                            feature = model(state, training=False)
                            pred = transfer_head(feature, training=True)
                            loss = transfer_loss_object(label, pred)
                        transfer_train_loss(loss)
                        gradients_t = tape_t.gradient(
                            loss, transfer_head.trainable_variables
                        )
                        transfer_optimizer.apply_gradients(
                            zip(gradients_t, transfer_head.trainable_variables)
                        )
                transfer_accuracy = eval_transfer()
                print("=" * 40)
                print("epoch: ", e, "iter: ", i)
                print(
                    "  train_loss:",
                    round(float(train_loss.result()), 2),
                    "  seconds passed:",
                    round(time() - t, 3),
                )
                print(
                    "  transfer_loss:",
                    round(float(transfer_train_loss.result()), 2),
                    "  transfer_accuracy:",
                    round(transfer_accuracy, 2),
                )
                t = time()
                train_loss.reset_states()
                transfer_train_loss.reset_states()