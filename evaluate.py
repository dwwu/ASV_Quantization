import tensorflow as tf
import numpy as np
import argparse

from tdnn_model import make_tdnn_model, tdnn_config
from data_loader import generate_voxc1_ds


parser = argparse.ArgumentParser("train speaker extractor")
parser.add_argument("-ckpt_dir", type=str, required=True)
args = parser.parse_args()

ckpt_dir = args.ckpt_dir
model_size = ckpt_dir.split('/')[-1]

config = tdnn_config(model_size)
model = make_tdnn_model(config, n_labels=1211)
latest = tf.train.latest_checkpoint(ckpt_dir)
model.load_weights(latest)
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

batch_size = 64
# val_x = np.expand_dims(np.load("val_500.npy"), 2)
# val_y = np.load("val_500_label.npy")
val_x = np.random.random((1000, 500, 1, 65))
val_y = np.random.randint(0, 1211, (1000,))
val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_ds = val_ds.batch(batch_size)
loss, acc = model.evaluate(val_ds)

print("eval loss:{:.3f}, acc:{:.3f}".format(loss, acc))
