import tensorflow as tf
from dataset import GAME_STATE_LENGTH


GPU = True
if GPU:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf.config.set_visible_devices([], "GPU")

model_config = {
    "num_output": 800,
    "num_units": [800, 800],
    "num_input": GAME_STATE_LENGTH,
    "activation": "relu",
    "last": "L2",
}
plain_head_config = {
    "num_output": 20,
    "num_units": [20],
    "activation": "relu",
}
classification_head_config = {
    "num_output": 20,
    "num_units": [800, 800],
    "activation": "relu",
    "last": "softmax",
}
