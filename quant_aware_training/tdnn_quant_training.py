#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append("../")
from tdnn_model import make_tdnn_model

# train parameters
n_epochs = 60
batch_size = 64
steps_per_epoch = 134000 // batch_size
config = {'conv_dim':512, 'stat_dim':1500, 'fc_dim':512}

# datasets
voxc1_train_dir = "../sv_set/voxc1/fbank64/dev/train/"
voxc1_val_dir = "../sv_set/voxc1/fbank64/dev/val/"
voxc1_test_dir = "../sv_set/voxc1/fbank64/dev/test/"

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_0/tdnn-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=3)

def scheduler(epoch):
    if epoch < 35:
        return 0.001
    elif epoch < 50:
        return 0.0005
    else:
        return 0.0005/2

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

train_x = np.expand_dims(np.load("train_500_1.npy"), 2)
train_y = np.load("train_500_1_label.npy")
val_x = np.expand_dims(np.load("val_500.npy"), 2)
val_y = np.load("val_500_label.npy")

def train_generator():
    for x, y in zip(train_x, train_y):
        yield x, y

def val_generator():
    for x, y in zip(val_x, val_y):
        yield x, y

def scheduler(epoch):
    if epoch < 35:
        return 0.001
    elif epoch < 50:
        return 0.0005
    else:
        return 0.0005/2
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

tf.keras.backend.set_session(train_sess)
with train_graph.as_default():
    model = make_tdnn_model(config, n_labels=1211)

    train_ds = tf.data.Dataset.from_generator(train_generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=((500, 1, 65), ()))
    train_ds = train_ds.shuffle(buffer_size=134000)
    train_ds = train_ds.repeat()
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)

    val_ds = tf.data.Dataset.from_generator(val_generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=((500, 1, 65), ()))
    val_ds = val_ds.batch(batch_size)

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    # model.fit(train_ds, epochs=n_epochs, steps_per_epoch=steps_per_epoch,
            # callbacks = [lr_callback],
            # validation_data=val_ds, validation_steps=100,
            # verbose=1)

    # save graph and checkpoints
    saver = tf.train.Saver()
    saver.save(train_sess, 'checkpoints')
