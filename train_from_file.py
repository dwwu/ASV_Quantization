import os
import numpy as np
import argparse
import tensorflow as tf

from tdnn_model import make_tdnn_model, tdnn_config
from data_loader import generate_voxc1_ds

AUTOTUNE = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser("train speaker extractor")
parser.add_argument("-n_epochs", type=int, default=40)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-ckpt_dir", type=str, required=True)
# parser.add_argument("-frame_range", nargs='+', default=(200, 400))
parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
args = parser.parse_args()


#####################################################
# train parameters
#####################################################

n_train_samples = 134000
n_epochs = args.n_epochs
batch_size = args.batch_size
model_size = args.model_size
ckpt_dir =  args.ckpt_dir

#####################################################
# datasets
#####################################################

steps_per_epoch = n_train_samples // batch_size


train_ds = generate_voxc1_ds("sv_set/voxc1/fbank64/dev/train/",
        frame_range=(200, 400), is_train=True)
train_ds = train_ds.padded_batch(batch_size, padded_shapes=([None, 1, 65], []),
        drop_remainder=False)

val_ds = generate_voxc1_ds("sv_set/voxc1/fbank64/dev/val/",
        frame_range=(300, 300), is_train=False)
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
model = make_tdnn_model(config, n_labels=1211)
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

#####################################################
# fit model
#####################################################

model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_ds, epochs=n_epochs, steps_per_epoch=steps_per_epoch,
          callbacks = [cp_callback, lr_callback],
          validation_data = val_ds, verbose=1,
          initial_epoch=0)

