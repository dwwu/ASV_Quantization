import tensorflow as tf
import argparse
from tdnn_model import tdnn_config, make_quant_tdnn_model

parser = argparse.ArgumentParser("sess_ckpt to frozen model")
parser.add_argument("-sess_file", type=str, required=True)
parser.add_argument("-frozen_dir", type=str, required=True)
parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
args = parser.parse_args()

save_file = args.sess_file
frozen_dir = args.frozen_dir
model_size = args.model_size
input_shape = (500, 1, 65)

eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)
tf.keras.backend.set_session(eval_sess)
with eval_graph.as_default():
    tf.keras.backend.set_learning_phase(0)
    config = tdnn_config(model_size)
    eval_model = make_quant_tdnn_model(config, 1211, input_shape)
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, save_file)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
    )
    tf.train.write_graph(
            frozen_graph_def,
            frozen_dir,
            'frozen_model.pb',
            as_text=False)
