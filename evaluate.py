import tensorflow as tf

from tdnn_model import make_tdnn_model
from data_loader import generate_voxc1_ds


config={}
model = make_tdnn_model(config, n_labels=1211)
checkpoint_dir = "tf_models/voxc1/training_0"
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

batch_size = 64
voxc1_val_dir = "sv_set/voxc1/fbank64/dev/test/"
val_ds = generate_voxc1_ds(voxc1_val_dir, frame_range=(300, 300))
val_ds = val_ds.batch(batch_size)

loss, acc = model.evaluate(val_ds)
