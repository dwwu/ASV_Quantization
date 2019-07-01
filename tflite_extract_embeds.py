import os
import numpy as np
import argparse

import tensorflow as tf
from multiprocessing import Process, Queue, Pool

tf.enable_eager_execution()

parser = argparse.ArgumentParser("Extract embed from tflite model")
parser.add_argument("-tflite_file", type=str, required=True)
parser.add_argument("-embed_file", type=str, required=True)
parser.add_argument("-batch_size", type=int, required=True)
parser.add_argument("-dataset", choices=['dev', 'test'], type=str, default='test', required=True)
parser.add_argument("-n_workers", type=int, default=4)
args = parser.parse_args()


#####################################################
# parameters
#####################################################
n_labels = 1211
tflite_file = args.tflite_file
batch_size = args.batch_size
ckpt_dir = '/'.join(tflite_file.split('/')[:-2])
embed_file = args.embed_file
dataset = args.dataset


#####################################################
# datasets
#####################################################

if dataset == 'dev':
	x = np.expand_dims(np.load("voxc1_dev/test_500.npy"), 2)
	y = np.load("voxc1_dev/test_500_label.npy")
else:
	x = np.expand_dims(np.load("voxc1_test/sv_test_500.npy"), 2)
	y = np.load("voxc1_test/sv_test_500_label.npy")

print("dataset loading complete")

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
    embed_list = pool.map(f_interpreter, x, len(x)//num_workers+1)
    print(len(embed_list))


# batch_residual = batch_size - feat.shape[0].value
# paddings = tf.constant([[0, batch_residual], [0, 0], [0, 0], [0, 0]])
# feat = tf.pad(feat, paddings, 'CONSTANT')

#####################################################
# save outputs
#####################################################
embed_dir = os.path.dirname(embed_file)
if not os.path.isfile(embed_dir):
    os.makedirs(embed_dir, exist_ok=True)
embed_array = np.array(embed_list)
np.save(embed_file, embed_array)
np.save(embed_dir+'/{}_label'.format(dataset), y)
