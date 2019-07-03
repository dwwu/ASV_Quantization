import numpy as np
import argparse

import tensorflow as tf


parser = argparse.ArgumentParser("evaluate tflite model")
parser.add_argument("-tflite_file", type=str, required=True)
parser.add_argument("-quantize", action='store_true')
args = parser.parse_args()

tflite_file = args.tflite_file

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
test_x = np.expand_dims(test_x / 255.0, -1).astype(np.float32)


def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    a, b = detail['quantization']
    return (data/a + b).astype(dtype).reshape(shape)

def dequantize(detail, data):
    a, b = detail['quantization']
    return (data - b)*a

#####################################################
# TFLite Interpreter
#####################################################

# load TFLite file
interpreter = tf.lite.Interpreter(model_path=tflite_file)
# Allocate memory.
interpreter.allocate_tensors()

# get some informations.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#####################################################
# Inference
#####################################################
if args.quantize:
    for x, y in zip(test_x, test_y):
        x = x.reshape(1, 28, 28, 1)
        x = quantize(input_details[0], x)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()

        # The results are stored on 'index' of output_details
        output_ = interpreter.get_tensor(output_details[0]['index'])
        output_ = dequantize(output_details[0], output_)
        pred = output_.argmax()
        print(pred, y)
else:
    for x, y in zip(test_x, test_y):
        x = x.reshape(1, 28, 28, 1)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()

        # The results are stored on 'index' of output_details
        output_ = interpreter.get_tensor(output_details[0]['index'])
        pred = output_.argmax()
        print(pred, y)
