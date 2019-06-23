import os
import tensorflow as tf

from tdnn_model import make_tdnn_model, StatPooling
from data_loader import generate_voxc1_ds
from convert_model  import convert_to_tflite, convert_to_quant


voxc1_train_dir = "sv_set/voxc1/fbank64/dev/train/"
voxc1_val_dir = "sv_set/voxc1/fbank64/dev/val/"
voxc1_test_dir = "sv_set/voxc1/fbank64/dev/test/"

n_frames = 300
batch_size = 64
step_per_epoch_ = 130000 // batch_size

train_ds, n_labels = generate_voxc1_ds(voxc1_train_dir, n_frames, is_train=True)
train_ds = train_ds.batch(batch_size)

val_ds, _ = generate_voxc1_ds(voxc1_val_dir, n_frames)
val_ds = val_ds.batch(batch_size)


# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "tf_models/voxc1/tdnn-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=1)

model = make_tdnn_model(n_labels=n_labels)
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_ds, epochs=21, steps_per_epoch=step_per_epoch_,
          callbacks = [cp_callback],
          validation_data = val_ds, verbose=0)

# tflite_file = "./tflite_models/voxc1_tdnn_model.h5"
# convert_to_tflite(tf_file, tflite_file, custom_objects_={'StatPooling':StatPooling})
# convert_to_quant(tf_file, tflite_file, custom_objects_={'StatPooling':StatPooling})



