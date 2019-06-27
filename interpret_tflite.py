import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_loader import generate_voxc1_ds

tf.enable_eager_execution()

voxc1_test_dir = "sv_set/voxc1/fbank64/test"
batch_size = 32  # requires the model to be converted with input_shapes
voxc1_ds = generate_voxc1_ds(voxc1_test_dir, frame_range=(300, 300))
# voxc1_ds = tf.data.Dataset.from_tensor_slices((np.random.random((10, 300, 1,
    # 65)).astype(np.float32), np.random.random((10))))
voxc1_ds = voxc1_ds.batch(batch_size)

tflite_file = "./tf_models/voxc1/training_0/latest_quant_b32_l300.tflite"

interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


embed_list = []
for i, (feat, label) in enumerate(tqdm(voxc1_ds, total=4874//batch_size)):
    if feat.shape[0] != batch_size:
        batch_residual = batch_size - feat.shape[0].value
        paddings = tf.constant([[0, batch_residual], [0, 0], [0, 0], [0, 0]])
        feat = tf.pad(feat, paddings, 'CONSTANT')
    interpreter.set_tensor(input_index, feat)
    interpreter.invoke()
    embed = interpreter.get_tensor(output_index)
    embed_list.append(embed)

embed_array = np.array(embed_list).reshape(-1, 1024)
print(embed_array.shape)
np.save("xvector_embeds/voxc1_xvector_embeds_quant.npy", embed_array)




