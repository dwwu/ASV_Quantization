# import os
import tensorflow as tf

from tdnn_model import make_tdnn_model, StatPooling
from convert_model import convert_to_tflite, convert_to_quant
from data_loader import generate_voxc1_ds

batch_size = 32
n_frames = 300
checkpoint_dir = "tf_models/voxc1/training_4"

model = make_tdnn_model(config={}, n_labels=1211, n_frames=n_frames)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

for _ in range(6):
    model.pop()

model.summary()

tf_file = "tf_models/voxc1/training_4/latest.h5"
model.save(tf_file)

# tflite_file = "tf_models/voxc1/training_0/latest.tflite"
# convert_to_tflite(tf_file, tflite_file,
        # custom_objects={'StatPooling':StatPooling}
        # )

# # tflite_file = "tf_models/voxc1/training_0/latest_32.tflite"
# # convert_to_tflite(tf_file, tflite_file,
        # # custom_objects={'StatPooling':StatPooling},
        # # input_shapes={"conv2d_input":[batch_size, 300, 1, 65]}
        # # )


# tflite_quant_file = "tf_models/voxc1/training_0/latest_quant.tflite"
# convert_to_quant(tf_file, tflite_quant_file,
        # custom_objects={'StatPooling':StatPooling}
        # )

tflite_quant_file = "tf_models/voxc1/training_0/latest_quant" + \
        "_b{}_l{}.tflite".format(batch_size, n_frames)

convert_to_quant(tf_file, tflite_quant_file,
        custom_objects={'StatPooling':StatPooling},
        input_shapes={"conv2d_input":[batch_size, n_frames, 1, 65]}
        )


# voxc1_val_dir = "sv_set/voxc1/fbank64/dev/val/"
# val_ds, _ = generate_voxc1_ds(voxc1_val_dir, n_frames=300)
# val_ds = val_ds.batch(batch_size)

# tflite_quant_file = "tf_models/voxc1/training_0/latest_quant_int_32.tflite"
# convert_to_quant(tf_file, tflite_quant_file,
        # custom_objects={'StatPooling':StatPooling},
        # input_shapes={"conv2d_input":[batch_size, 300, 1, 65]},
        # act_quant=True, repr_ds=val_ds
        # )

