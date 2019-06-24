import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_loader import generate_voxc1_ds

tf.enable_eager_execution()

voxc1_test_dir = "sv_set/voxc1/fbank64/test"
n_frames = 300
batch_size = 32  # requires the model to be converted with input_shapes
voxc1_ds, _= generate_voxc1_ds(voxc1_test_dir, n_frames, is_train=False)
voxc1_ds = voxc1_ds.batch(batch_size)

tflite_file = "./tf_models/voxc1/training_0/latest_quant32.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# tf.logging.set_verbosity(tf.logging.DEBUG)

embed_list = []
for i, (feat, label) in enumerate(tqdm(voxc1_ds.take(3), total=4874//batch_size)):
    interpreter.set_tensor(input_index, feat)
    interpreter.invoke()
    embed = interpreter.get_tensor(output_index)
    embed_list.append(embed)


# for feat, label in tqdm(voxc1_ds, total=100):
    # interpreter.set_tensor(input_index, feat)
    # interpreter.invoke()
    # embed = interpreter.get_tensor(output_index)
    # embed_list.append(embed)

embed_array = np.array(embed_list).squeeze()
print(embed_array.shape)
np.save("xvector_embeds/voxc1_xvector_embeds_quant.npy", embed_array)




