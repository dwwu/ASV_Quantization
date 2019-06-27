import os
import numpy as np
import argparse

import tensorflow as tf
# from tdnn_model import make_tdnn_model, tdnn_config


# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

tf.keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    tf.keras.backend.set_learning_phase(0)
    tf.contrib.quantizue.create_eval_graph(input_graph=eval_graph)
    eval_model = make_tdnn_model()
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, 'checkpoints')

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
    )

    with open('frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())