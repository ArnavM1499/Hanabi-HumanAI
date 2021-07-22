from naiveFC import NaiveFC
import tensorflow as tf


GPU = True
if GPU:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf.config.set_visible_divices([], "GPU")

model = NaiveFC(20, num_units=[600, 400, 200], activation="relu")
plain_head = NaiveFC(20, num_units=[20], activation="relu")
classification_head = NaiveFC(20, num_units=[20], activation="relu", last="softmax")
