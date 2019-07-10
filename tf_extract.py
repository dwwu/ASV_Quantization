import os
import numpy as np
import argparse

import tensorflow as tf
from tdnn_model import make_quant_tdnn_model, tdnn_config
from data.dataset import Voxceleb1

parser = argparse.ArgumentParser("train speaker extractor")
parser.add_argument("-ckpt_dir", type=str, required=True)
parser.add_argument("-set_n", choices=['dev', 'test'], type=str, default='test')
parser.add_argument("-embed_file", type=str, required=True)
parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
parser.add_argument("-output_layer", type=str, default='xvector')
args = parser.parse_args()

#####################################################
# parameters
#####################################################

n_labels = 1211
batch_size = 64
ckpt_dir = args.ckpt_dir
model_size = args.model_size
embed_file = args.embed_file
embed_dir = os.path.dirname(embed_file)
set_name = args.set_n
layer_name = args.output_layer


#####################################################
# model
#####################################################

config = tdnn_config(model_size)
model = make_quant_tdnn_model(config, n_labels, (500, 1, 65))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

latest = tf.train.latest_checkpoint(ckpt_dir)
model.load_weights(latest)


#####################################################
# datasets
#####################################################
dataset = Voxceleb1("/tmp/sv_set/voxc1/fbank64")
if set_name == 'dev':
    x, _ = dataset.get_norm("dev/train", scale=24)
else:
    x, _ = dataset.get_norm("test/test", scale=24)

ds = tf.data.Dataset.from_tensor_slices(x)
ds = ds.batch(batch_size)

layer_name = "dense" # xvector
intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
embeds = intermediate_layer_model.predict(ds, verbose=1)

if not os.path.isfile(embed_dir):
    os.makedirs(embed_dir, exist_ok=True)
np.save(embed_file, embeds)
print("embeds are saved to {}".format(embed_file))
