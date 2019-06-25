import os
import tensorflow as tf

from tdnn_model import make_tdnn_model
from data_loader import generate_voxc1_ds


# train parameters
frame_range = (200, 400)
n_epochs = 60
batch_size = 64
step_per_epoch_ = 134000 // batch_size
config = {'conv_dim':512, 'stat_dim':1500, 'fc_dim':512}

# datasets
voxc1_train_dir = "sv_set/voxc1/fbank64/dev/train/"
voxc1_val_dir = "sv_set/voxc1/fbank64/dev/val/"
voxc1_test_dir = "sv_set/voxc1/fbank64/dev/test/"

train_ds = generate_voxc1_ds(voxc1_train_dir, frame_range, is_train=True)
# train_ds = train_ds.batch(batch_size)
train_ds = train_ds.padded_batch(batch_size, padded_shapes=([None, 1, 65], []),
        drop_remainder=True)
val_ds = generate_voxc1_ds(voxc1_val_dir, (300, 300))
val_ds = val_ds.batch(batch_size)
# val_ds = val_ds.padded_batch(batch_size, padded_shapes=([None, 1, 65], []),
        # drop_remainder=True)

# callbacks
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "tf_models/voxc1/training_tdnn_fc_512/tdnn-{epoch:04d}.ckpt"
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

model = make_tdnn_model(config, n_labels=1211)
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_ds, epochs=n_epochs, steps_per_epoch=step_per_epoch_,
          callbacks = [cp_callback, lr_callback],
          validation_data = val_ds, verbose=1,
          initial_epoch=0)

