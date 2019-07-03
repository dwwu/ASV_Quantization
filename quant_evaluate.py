import tensorflow as tf
import numpy as np
import os
import argparse

from tdnn_model import make_quant_tdnn_model, tdnn_config


parser = argparse.ArgumentParser("evalute quantized model")
parser.add_argument("-ckpt_file", type=str, required=True)
parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
args = parser.parse_args()

ckpt_file = args.ckpt_file
ckpt_dir = os.path.dirname(ckpt_file)
model_size = args.model_size
batch_size = 64

# latest = tf.train.latest_checkpoint(ckpt_dir)

####################################################
# eval
####################################################

test_x = np.load("sv_set/voxc1/fbank64/dev/merged/val_500.npy")
test_y = np.load("sv_set/voxc1/fbank64/dev/merged/val_500_label.npy")
test_x = np.expand_dims(test_x, 2)

def test_generator():
    for x, y in zip(test_x, test_y):
        yield x, y

eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)
tf.keras.backend.set_session(eval_sess)
with eval_graph.as_default():
    test_ds = tf.data.Dataset.from_generator(test_generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=((500, 1, 65), ()))
    test_ds = test_ds.batch(batch_size)
    test_iterator = test_ds.make_one_shot_iterator()
    test_feat, test_label = test_iterator.get_next()
    test_feat = tf.quantization.fake_quant_with_min_max_args(test_feat,
            min=-24, max=16)

    tf.keras.backend.set_learning_phase(0)
    config = tdnn_config(model_size)
    eval_model = make_quant_tdnn_model(config, n_labels=1211, n_frames=500)
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, ckpt_file)

    eval_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    loss, acc = eval_model.evaluate(test_feat, test_label,
            steps=len(test_x)//batch_size)

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
    )

    with open(os.path.join(ckpt_dir, 'frozen_model.pb'), 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
