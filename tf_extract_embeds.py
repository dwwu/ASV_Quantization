import os
import numpy as np
import argparse

import tensorflow as tf

from tdnn_model import make_tdnn_model, tdnn_config

parser = argparse.ArgumentParser("train speaker extractor")
parser.add_argument("-ckpt_dir", type=str, required=True)
# parser.add_argument("-embed_dir", type=str, required=True)
# parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
args = parser.parse_args()

n_labels = 1211
batch_size = 64
ckpt_dir = args.ckpt_dir
model_size = ckpt_dir.split('/')[-1]
embed_dir = os.path.join(ckpt_dir, 'embeds', 'tf')

config = tdnn_config(model_size)
model = make_tdnn_model(config, n_labels=n_labels)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

latest = tf.train.latest_checkpoint(ckpt_dir)
model.load_weights(latest)

# datasets

# train_x = np.expand_dims(np.load("train_500_1.npy"), 2)
# train_y = np.load("train_500_1_label.npy")
# val_x = np.expand_dims(np.load("val_500.npy"), 2)
# val_y = np.load("val_500_label.npy")
dev_x = np.random.random((1000, 500, 1, 65))
dev_y = np.random.randint(0, 1211, (1000,))
test_x = np.random.random((1000, 500, 1, 65))
test_y = np.random.randint(0, 1211, (1000,))

dev_ds = tf.data.Dataset.from_tensor_slices((dev_x, dev_x))
dev_ds = dev_ds.batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds = test_ds.batch(batch_size)

# layer_name = "dense" # xvector
# intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
#                                  outputs=model.get_layer(layer_name).output)
# test_embeds = intermediate_layer_model.predict(test_ds, verbose=1)

if not os.path.isfile(embed_dir):
    os.makedirs(embed_dir, exist_ok=True)

dev_embeds = model.predict(dev_ds, verbose=1)
np.save(os.path.join(embed_dir, "si_embeds.npy"), dev_embeds)
np.save(os.path.join(embed_dir, "si_labels.npy"), dev_y)

test_embeds = model.predict(test_ds, verbose=1)
np.save(os.path.join(embed_dir, "sv_embeds.npy"), test_embeds)
np.save(os.path.join(embed_dir, "sv_labels.npy"), test_y)

print("embeds are saved to {}".format(embed_dir))




