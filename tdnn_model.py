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

def tdnn_config(model_size):
    if model_size == 'S':
        conv_dim = 256
        stat_dim = 512
        fc_dim = 256
    elif model_size == 'M':
        conv_dim = 512
        stat_dim = 1500
        fc_dim = 512
    elif model_size == 'L':
        conv_dim = 512
        stat_dim = 1500
        fc_dim = 512
    else:
        raise NotImplementedError

    config = dict(conv_dim=conv_dim, stat_dimm=stat_dim, fc_dim=fc_dim)
    return config

def make_tdnn_model(config, n_labels, n_frames=None):
    """
        dimension = (N, W, H, C) = (batch_size, T, 1, feat) for channel_last
        :return:
    """

    conv_dim = config.get('conv_dim', 512)
    stat_dim = config.get('stat_dim', 1500)
    fc_dim = config.get('fc_dim', 1024)

    l2_decay = 0.000001
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(conv_dim, kernel_size=(5, 1), strides=1, dilation_rate=1,
                                     input_shape=(n_frames, 1, 65),
                                     activation='linear', use_bias=True,
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
                                     )
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(conv_dim, kernel_size=(3, 1), strides=1, dilation_rate=(3, 1),
                                     activation='linear', use_bias=True,
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
                                     )
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(conv_dim, kernel_size=(3, 1), strides=1, dilation_rate=(4, 1),
                                     activation='linear', use_bias=True,
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
                                     )
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(conv_dim, kernel_size=1, strides=1, dilation_rate=1,
                                     activation='linear', use_bias=True,
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
                                     )
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(stat_dim, kernel_size=1, strides=1, dilation_rate=1,
                                     activation='linear', use_bias=True,
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_decay))
                                     )
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(StatPooling())

    model.add(tf.keras.layers.Dense(fc_dim, activation='linear', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(fc_dim, activation='linear', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(n_labels, activation='softmax'))

    return model



