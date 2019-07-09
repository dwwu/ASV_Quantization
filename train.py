import os
import argparse
import tensorflow as tf

from tdnn_model import make_quant_tdnn_model, tdnn_config
from data.dataset import Voxceleb1

AUTOTUNE = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser("train speaker extractor")
parser.add_argument("-n_epochs", type=int, default=40)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-ckpt_dir", type=str, required=True)
parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
args = parser.parse_args()


#####################################################
# train parameters
#####################################################

n_epochs = args.n_epochs
batch_size = args.batch_size
model_size = args.model_size
ckpt_dir =  args.ckpt_dir

#####################################################
# datasets
#####################################################

dataset = Voxceleb1("/tmp/sv_set/voxc1/fbank64")
train_x, train_y = dataset.get_norm("dev/train", scale=24)
val_x, val_y = dataset.get_norm("dev/val", scale=24)

n_train_samples = len(train_x)
steps_per_epoch = n_train_samples // batch_size
input_shape = (train_x.shape[1], train_x.shape[2], train_x.shape[3])


def train_generator():
    for x, y in zip(train_x, train_y):
        yield x, y

def val_generator():
    for x, y in zip(val_x, val_y):
        yield x, y

train_ds = tf.data.Dataset.from_generator(train_generator,
                                          output_types=(tf.float32, tf.int32),
                                          output_shapes=(input_shape, ()))
train_ds = train_ds.shuffle(buffer_size=n_train_samples)
train_ds = train_ds.repeat()
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(val_generator,
                                        output_types=(tf.float32, tf.int32),
                                        output_shapes=(input_shape, ()))
val_ds = val_ds.batch(batch_size)

#####################################################
# callbacks
#####################################################

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = os.path.join(ckpt_dir, args.model_size,
                               "checkpoint-{epoch:04d}.ckpt")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True, save_freq='epoch')

def scheduler(epoch):
    if epoch < 35:
        return 0.05
    elif epoch < 50:
        return 0.05/2
    else:
        return 0.05/2/2

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

#####################################################
# models
#####################################################

config = tdnn_config(args.model_size)
model = make_quant_tdnn_model(config, 1211, input_shape)
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

#####################################################
# fit model
#####################################################

model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_ds, epochs=n_epochs, steps_per_epoch=steps_per_epoch,
          callbacks=[cp_callback, lr_callback],
          validation_data=val_ds, validation_steps=len(val_x)//batch_size,
          verbose=1, initial_epoch=0)
