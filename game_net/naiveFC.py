import tensorflow as tf


def NaiveFC(
    num_output=16, num_layers=4, num_units=256, activation="sigmoid", last="", dropout=0
):

    if isinstance(num_units, int):
        num_units = [num_units for _ in range(num_layers)]
    assert isinstance(num_units, list)

    with tf.device("/GPU:0"):
        layers = [tf.keras.layers.Dense(u, activation=activation) for u in num_units]
        # if dropout > :
        #     layers = sum([[L, tf.keras.layers.Dropout(dropout)] for L in layers], [])
        layers.append(tf.keras.layers.Dense(num_output, activation=None))
        if last == "L2":
            layers.append(
                tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
            )
        elif last == "softmax":
            layers.append(tf.keras.layers.Softmax())
        return tf.keras.Sequential(layers)


class ResFC(tf.keras.Model):
    def __init__(
        self, num_output=16, num_layers=4, num_units=256, activation="sigmoid", last=""
    ):
        super(ResFC, self).__init__()
        self.num_layers = num_layers
        self.hidden_layers = [
            tf.keras.layers.Dense(num_units, activation=activation)
            for _ in range(num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(num_output)
        if last == "L2":
            self.last = tf.keras.layers.Lambda(
                lambda x: tf.math.l2_normalize(x, axis=1)
            )
        elif last == "softmax":
            self.last = tf.keras.layers.Softmax()

    def call(self, x):
        layers = [x]
        for i, fc in enumerate(self.hidden_layers):
            if i > 0:
                layers.append(fc(layers[-1]) + layers[-1])
            else:
                layers.append(fc(layers[-1]))
        output = self.output_layer(layers[-1])
        return self.last(output) if self.last else output
