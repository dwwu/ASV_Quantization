import tensorflow as tf
import numpy as np
import pickle

from tdnn_model import make_tdnn_model
from data_loader import generate_voxc1_ds

n_labels = 1211
model = make_tdnn_model(config={}, n_labels=n_labels)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
checkpoint_dir = "tf_models/voxc1/training_4"
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)


voxc1_dev_dir = "sv_set/voxc1/fbank64/dev/train"
voxc1_ds, label_list = generate_voxc1_ds(voxc1_dev_dir, frame_range=(700, 700),
        return_labels=True)
voxc1_ds = voxc1_ds.batch(64)
layer_name = "dense" # xvector
intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
xvector_embeds = intermediate_layer_model.predict(voxc1_ds, verbose=1)
np.save("xvector_embeds/voxc1_si_embeds.npy", xvector_embeds)
np.save("xvector_embeds/voxc1_si_labels.npy", label_list)

voxc1_test_dir = "sv_set/voxc1/fbank64/test"
voxc1_ds, label_list = generate_voxc1_ds(voxc1_test_dir, frame_range=(700, 700),
        return_labels=True)
voxc1_ds = voxc1_ds.batch(64)
layer_name = "dense" # xvector
intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
xvector_embeds = intermediate_layer_model.predict(voxc1_ds, verbose=1)
np.save("xvector_embeds/voxc1_sv_embeds.npy", xvector_embeds)
np.save("xvector_embeds/voxc1_sv_labels.npy", label_list)




