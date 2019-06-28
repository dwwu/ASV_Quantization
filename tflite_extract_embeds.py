import os
import numpy as np
import argparse

from tqdm import tqdm
import tensorflow as tf
from multiprocessing import Process, Queue, Pool

tf.enable_eager_execution()

parser = argparse.ArgumentParser("train speaker extractor")
parser.add_argument("-tflite_file", type=str, required=True)
parser.add_argument("-batch_size", type=int, required=True)
parser.add_argument("-n_workers", type=int, default=4)
args = parser.parse_args()


#####################################################
# parameters
#####################################################
n_labels = 1211
tflite_file = args.tflite_file
batch_size = args.batch_size
ckpt_dir = '/'.join(tflite_file.split('/')[:-2])
embed_dir = os.path.join(ckpt_dir, 'embeds', 'tflite')


#####################################################
# datasets
#####################################################

test_x = np.expand_dims(np.load("sv_set/voxc1/fbank64/npy/test_500.npy"), 2)
test_y = np.expand_dims(np.load("sv_set/voxc1/fbank64/npy/test_500_label.npy"), 2)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds = test_ds.batch(batch_size)

#####################################################
# interpreters
#####################################################

def f_interpreter(data):
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    data = tf.convert_to_tensor(np.expand_dims(data, 0))
    interpreter.set_tensor(input_index, data)
    interpreter.invoke()
    embed = interpreter.get_tensor(output_index)

    return embed

num_workers = args.n_workers
with Pool(processes=num_workers) as pool:
    embed_list = pool.map(f_interpreter, test_x, len(test_x)//num_workers+1)
    print(len(embed_list))


# batch_residual = batch_size - feat.shape[0].value
# paddings = tf.constant([[0, batch_residual], [0, 0], [0, 0], [0, 0]])
# feat = tf.pad(feat, paddings, 'CONSTANT')

#####################################################
# save outputs
#####################################################
if not os.path.isfile(embed_dir):
    os.makedirs(embed_dir, exist_ok=True)

embed_array = np.array(embed_list)
np.save(os.path.join(embed_dir, "{}_sv_embeds.npy"), embed_array)
np.save(os.path.join(embed_dir, "{}_sv_labels.npy"), test_y)
