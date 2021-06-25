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


class ResFC(tf.keras.Model):
    def __init__(
        self, num_output=16, num_layers=4, num_units=256, activation="sigmoid"
    ):
        super(ResFC, self).__init__()
        self.num_layers = num_layers
        self.hidden_layers = [
            tf.keras.layers.Dense(num_units, activation=activation)
            for _ in range(num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(num_output, activation=None)
        self.L2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    def call(self, x):
        layers = [x]
        for i, fc in enumerate(self.hidden_layers):
            if i > 1:
                layers.append(fc(layers[-1]) + layers[-2])
            else:
                layers.append(fc(layers[-1]))
        if self.num_layers > 0:
            output = self.output_layer(layers[-1] + layers[-2])
        else:
            output = self.output_layer(layers[0])
        return self.L2(output)
