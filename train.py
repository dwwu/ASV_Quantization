import tensorflow as tf

from tdnn_model import make_tdnn_model, StatPooling
from data_loader import generate_voxc1_ds
from convert_model  import convert_to_tflite, convert_to_quant

model = make_tdnn_model()
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# voxc1_dir = "./voxceleb1/feat/fbank64/"
# n_frames = 300
# batch_size = 32
# voxc1_ds = generate_voxc1_ds(voxc1_dir, n_frames, batch_size)
# model.fit(voxc1_ds, epochs=10, steps_per_epoch=10)

tf_file = "./tf_models/voxc1_tdnn_model.h5"
tflite_file = "./tflite_models/voxc1_tdnn_model.h5"

tf.keras.models.save_model(model, tf_file)

convert_to_tflite(tf_file, tflite_file, custom_objects_={'StatPooling':StatPooling})
convert_to_quant(tf_file, tflite_file, custom_objects_={'StatPooling':StatPooling})



