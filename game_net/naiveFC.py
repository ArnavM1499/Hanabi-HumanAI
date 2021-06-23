import tensorflow as tf


def NaiveFC(num_output=16, num_layers=4, num_units=256, activation="sigmoid"):

    with tf.device("/GPU:0"):
        layers = [
            tf.keras.layers.Dense(num_units, activation=activation)
            for _ in range(num_layers)
        ]
        layers.append(tf.keras.layers.Dense(num_output, activation=None))
        layers.append(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
        return tf.keras.Sequential(layers)
