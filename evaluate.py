import tensorflow as tf
import numpy as np
import argparse

from tdnn_model import make_tdnn_model, make_quant_tdnn_model, tdnn_config


parser = argparse.ArgumentParser("train speaker extractor")
parser.add_argument("-ckpt_dir", type=str, required=True)
parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
args = parser.parse_args()

ckpt_dir = args.ckpt_dir
model_size = args.model_size

config = tdnn_config(model_size)
model = make_quant_tdnn_model(config, n_labels=1211)
latest = tf.train.latest_checkpoint(ckpt_dir)
model.load_weights(latest)
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

batch_size = 64
test_x = np.load("sv_set/voxc1/fbank64/dev/merged/train_500_1_quant.npy")
test_y = np.load("sv_set/voxc1/fbank64/dev/merged/train_500_1_label.npy")
# test_x = np.load("sv_set/voxc1/fbank64/dev/merged/val_500_quant.npy")
# test_y = np.load("sv_set/voxc1/fbank64/dev/merged/val_500_label.npy")
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds = test_ds.batch(batch_size)
loss, acc = model.evaluate(test_ds)

print("eval loss:{:.3f}, acc:{:.3f}".format(loss, acc))
