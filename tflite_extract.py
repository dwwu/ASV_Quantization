import os
import numpy as np
import argparse

import tensorflow as tf
from multiprocessing import Pool
from data.dataset import Voxceleb1

tf.enable_eager_execution()

parser = argparse.ArgumentParser("Extract embed from tflite model")
parser.add_argument("-tflite_file", type=str, required=True)
parser.add_argument("-embed_file", type=str, required=True)
parser.add_argument("-output_node", type=str, required=True)
parser.add_argument("-batch_size", type=int, required=True)
parser.add_argument("-set_n", choices=['dev', 'test'], type=str, default='test')
parser.add_argument("-n_worker", type=int, default=4)
parser.add_argument("-quantize", action='store_true')
args = parser.parse_args()

def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    a, b = detail['quantization']

    return (data/a + b).astype(dtype).reshape(shape)

def dequantize(detail, data):
    a, b = detail['quantization']

    return (data - b)*a

#####################################################
# parameters
#####################################################

n_labels = 1211
tflite_file = args.tflite_file
batch_size = args.batch_size
ckpt_dir = '/'.join(tflite_file.split('/')[:-2])
embed_file = args.embed_file
set_name = args.set_n
output_node = args.output_node


#####################################################
# datasets
#####################################################

dataset = Voxceleb1("/tmp/sv_set/voxc1/fbank64")
if set_name == 'dev':
    x, _ = dataset.get_norm("dev/train", scale=24)
else:
    x, _ = dataset.get_norm("test/test", scale=24)


#####################################################
# interpreters
#####################################################

def get_output_tensor(interpreter, name):
    tensor_info = interpreter.get_tensor_details()
    tensor_info = sorted(tensor_info, key=lambda x: x["index"])
    tensor_name_index = {info['name']:info['index'] for info in tensor_info}
    try:
        tensor_index = tensor_name_index[name]
    except KeyError as e:
        print(tensor_info)
    tensor_detail = tensor_info[tensor_index]

    return tensor_index, tensor_detail

def interpreter_fn(data):
    data = np.expand_dims(data, 0)
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    input_details = interpreter.get_input_details()
    output_index, output_details = get_output_tensor(interpreter, output_node)
    if args.quantize:
        data = quantize(input_details[0], data)
    data = tf.convert_to_tensor(data)
    interpreter.set_tensor(input_index, data)
    interpreter.invoke()
    embed = interpreter.get_tensor(output_index)
    if args.quantize:
        embed = dequantize(output_details, embed)

    return embed

num_workers = args.n_worker
with Pool(processes=num_workers) as pool:
    embed_list = pool.map(interpreter_fn, x, len(x)//num_workers+1)

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
print("embed shape:{}".format(embed_array.shape))
np.save(embed_file, embed_array)
