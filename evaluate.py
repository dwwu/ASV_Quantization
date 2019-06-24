import tensorflow as tf

from tdnn_model import make_tdnn_model
from data_loader import generate_voxc1_ds


model = make_tdnn_model(n_labels=1211, n_frames=300)
<<<<<<< HEAD
checkpoint_dir = "tf_models/voxc1/training_0"
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
=======
checkpoint_dir = "tf_models/voxc1/"
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
# model.load_weights('tf_models/voxc1/tdnn-0008.ckpt')
>>>>>>> 305f80c7220b81bf19998654c3cb5e84fceddc15
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

voxc1_val_dir = "sv_set/voxc1/fbank64/dev/val/"
val_ds, _ = generate_voxc1_ds(voxc1_val_dir, n_frames=300)
val_ds = val_ds.batch(64)

loss, acc = model.evaluate(val_ds)
