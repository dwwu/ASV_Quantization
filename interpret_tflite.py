import tensorflow as tf

from data_loader import generate_voxc1_ds

voxc1_dir = "./voxceleb1/feat/fbank64/"
n_frames = 300
batch_size = 32  # requires the model to be converted with input_shapes
voxc1_ds = generate_voxc1_ds(voxc1_dir, n_frames, batch_size)

tflite_file = "./tflite_models/voxc1_tdnn_model.h5"

interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

tf.logging.set_verbosity(tf.logging.DEBUG)

for feat, _ in voxc1_ds:
    interpreter.set_tensor(input_index, feat)
    interpreter.invoke()
    embed = interpreter.get_tensor(output_index)
    print(embed.shape)




