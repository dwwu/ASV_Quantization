import os
import argparse
import tensorflow as tf

from tdnn_model import make_quant_tdnn_model, tdnn_config
from data.dataset import Voxceleb1


parser = argparse.ArgumentParser("evalute quantized model")
parser.add_argument("-ckpt_file", type=str, required=True)
parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
args = parser.parse_args()

ckpt_file = args.ckpt_file
ckpt_dir = os.path.dirname(ckpt_file)
model_size = args.model_size
batch_size = 64

####################################################
# eval
####################################################

dataset = Voxceleb1("/tmp/sv_set/voxc1/fbank64")
test_x, test_y = dataset.get_norm("dev/test", scale=24)
input_shape = (test_x.shape[1], test_x.shape[2], test_x.shape[3])

def test_generator():
    for x, y in zip(test_x, test_y):
        yield x, y

eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)
tf.keras.backend.set_session(eval_sess)
with eval_graph.as_default():
    test_ds = tf.data.Dataset.from_generator(test_generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=(input_shape, ()))
    test_ds = test_ds.batch(batch_size)
    test_iterator = test_ds.make_one_shot_iterator()
    test_feat, test_label = test_iterator.get_next()
    test_feat = tf.quantization.fake_quant_with_min_max_args(test_feat,
            min=-1, max=1)

    tf.keras.backend.set_learning_phase(0)
    config = tdnn_config(model_size)
    eval_model = make_quant_tdnn_model(config, 1211, input_shape)
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, ckpt_file)

    eval_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    loss, acc = eval_model.evaluate(test_feat, test_label,
            steps=len(test_x)//batch_size)

    print("loss: {:4.f}, acc: {:3.f}".format(loss, acc))
