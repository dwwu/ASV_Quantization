import tensorflow as tf

tf.compat.v1.enable_eager_execution()

class StatPooling(tf.keras.layers.Layer):

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def get_config(self, ):
        base_config = super().get_config()
        return base_config

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis=1)
        std = tf.math.reduce_std(inputs, axis=1)
        stat = tf.concat([mean, std], axis=-1)
        return tf.squeeze(stat, axis=1)


def make_tdnn_model(n_labels, n_frames=None):
    """
        dimension = (N, W, H, C) = (batch_size, T, 1, feat) for channel_last
        :return:
    """

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(512, kernel_size=(5, 1), strides=1, dilation_rate=1,
                                     input_shape=(n_frames, 1, 65),
                                     activation='linear', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 1), strides=1, dilation_rate=(3, 1),
                                     activation='linear', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 1), strides=1, dilation_rate=(4, 1),
                                     activation='linear', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, dilation_rate=1,
                                     activation='linear', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(1500, kernel_size=1, strides=1, dilation_rate=1,
                                     activation='linear', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(StatPooling())

    model.add(tf.keras.layers.Dense(1024, activation='linear', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(1024, activation='linear', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(n_labels, activation='softmax'))

    return model



