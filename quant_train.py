import os
import argparse
import tensorflow as tf

from tdnn_model import make_quant_tdnn_model, tdnn_config
from data.dataset import Voxceleb1

AUTOTUNE = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser("train speaker extractor in quantized model")
parser.add_argument("-n_epochs", type=int, default=40)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-root_dir", type=str, default='/tmp/sv_set/voxc1/fbank64')
parser.add_argument("-ckpt_dir", type=str, required=True)
parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
parser.add_argument("-model_file", type=str, default=None)
args = parser.parse_args()

####################################################
# train parameters
####################################################

n_epochs = args.n_epochs
batch_size = args.batch_size
model_size = args.model_size
ckpt_dir = args.ckpt_dir
model_file = args.model_file

####################################################
# datasets
####################################################

dataset = Voxceleb1(args.root_dir)
train_x, train_y = dataset.get_norm("dev/train", scale=24)
val_x, val_y = dataset.get_norm("dev/val", scale=24)
input_shape = (train_x.shape[1], train_x.shape[2], train_x.shape[3])

def train_generator():
    for x, y in zip(train_x, train_y):
        yield x, y

def val_generator():
    for x, y in zip(val_x, val_y):
        yield x, y

####################################################
# callbacks
####################################################

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = os.path.join(ckpt_dir, "checkpoint-{epoch:04d}.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True, save_freq='epoch')

def scheduler(epoch):
    if epoch < 35:
        return 0.001
    elif epoch < 50:
        return 0.0005
    else:
        return 0.0005/2
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

####################################################
# train & models
####################################################

train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

tf.keras.backend.set_session(train_sess)
with train_graph.as_default():
    config = tdnn_config(model_size)
    train_model = make_quant_tdnn_model(config, 1211, input_shape)
    if model_file is not None:
        train_model.load_weights(model_file)

    train_ds = tf.data.Dataset.from_generator(train_generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=(input_shape, ()))
    train_ds = train_ds.shuffle(buffer_size=len(train_x))
    train_ds = train_ds.repeat()
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_iterator = train_ds.make_one_shot_iterator()
    train_feat, train_label = train_iterator.get_next()
    train_feat = tf.quantization.fake_quant_with_min_max_args(train_feat,
            min=-1, max=1)

    val_ds = tf.data.Dataset.from_generator(val_generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=(input_shape, ()))
    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(batch_size)
    val_iterator = val_ds.make_one_shot_iterator()
    val_feat, val_label = val_iterator.get_next()
    val_feat = tf.quantization.fake_quant_with_min_max_args(val_feat,
            min=-1, max=1)

    tf.contrib.quantize.create_training_graph(input_graph=train_graph,
            quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    train_model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9,
        nesterov=True),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    train_model.fit(train_feat, train_label, epochs=n_epochs,
            steps_per_epoch=len(train_x)//batch_size,
            callbacks = [cp_callback, lr_callback],
            validation_data=(val_feat, val_label),
            validation_steps=len(val_x)//batch_size,
            verbose=1)

    # save graph and checkpoints
    saver = tf.train.Saver()
    save_file = os.path.join(ckpt_dir, 'sess_checkpoints')
    if not os.path.isfile(save_file):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
    saver.save(train_sess, save_file)

####################################################
# eval
####################################################

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
            ckpt_dir,
            'frozen_model.pb',
            as_text=False)
