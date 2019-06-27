import os
import numpy as np
import argparse

from tqdm import tqdm
import tensorflow as tf


parser = argparse.ArgumentParser("train speaker extractor")
parser.add_argument("-tflite_file", type=str, required=True)
args = parser.parse_args()

n_labels = 1211
tflite_file = args.tflite_file
batch_size = int(os.path.basename(tflite_file).split('_')[-1].rstrip('.h5'))
ckpt_dir = '/'.join(tflite_file.split('/')[:-2])
embed_dir = os.path.join(ckpt_dir, 'embeds', 'tflite')


# datasets

# train_x = np.expand_dims(np.load("train_500_1.npy"), 2)
# train_y = np.load("train_500_1_label.npy")
# val_x = np.expand_dims(np.load("val_500.npy"), 2)
# val_y = np.load("val_500_label.npy")
# dev_x = np.random.random((1000, 500, 1, 65))
# dev_y = np.random.randint(0, 1211, (1000,))
test_x = np.random.random((1000, 500, 1, 65)).astype(np.float32)
test_y = np.random.randint(0, 1211, (1000,))

# dev_ds = tf.data.Dataset.from_tensor_slices((dev_x, dev_x))
# dev_ds = dev_ds.batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds = test_ds.batch(batch_size)


interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


embed_list = []
for i, (feat, label) in enumerate(tqdm(test_ds)):
    # handling last batch
    if feat.shape[0] != batch_size:
        batch_residual = batch_size - feat.shape[0].value
        paddings = tf.constant([[0, batch_residual], [0, 0], [0, 0], [0, 0]])
        feat = tf.pad(feat, paddings, 'CONSTANT')
    interpreter.set_tensor(input_index, feat)
    interpreter.invoke()
    embed = interpreter.get_tensor(output_index)
    embed_list.append(embed)

if not os.path.isfile(embed_dir):
    os.makedirs(embed_dir, exist_ok=True)

embed_array = np.array(embed_list)
np.save(os.path.join(embed_dir, "{}_sv_embeds.npy"), embed_array)
np.save(os.path.join(embed_dir, "{}_sv_labels.npy"), test_y)
