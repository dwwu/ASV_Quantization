import os
import numpy as np
import argparse
import tensorflow as tf
import subprocess

from tdnn_model import make_tdnn_model, tdnn_config

AUTOTUNE = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser("train speaker extractor")
parser.add_argument("-n_epochs", type=int, default=40)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-ckpt_dir", type=str, required=True)
# parser.add_argument("-frame_range", nargs='+', default=(200, 400))
parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], default='M', required=True)
args = parser.parse_args()

# train parameters

n_epochs = args.n_epochs
batch_size = args.batch_size
steps_per_epoch = 10
model_size = args.model_size
ckpt_dir = os.path.join(args.ckpt_dir, model_size)

# datasets

# train_x = np.expand_dims(np.load("train_500_1.npy"), 2)
# train_y = np.load("train_500_1_label.npy")
# val_x = np.expand_dims(np.load("val_500.npy"), 2)
# val_y = np.load("val_500_label.npy")

train_x = np.random.random((1000, 500, 1, 65))
train_y = np.random.randint(0, 1211, (1000,))
val_x = np.random.random((1000, 500, 1, 65))
val_y = np.random.randint(0, 1211, (1000,))

def train_generator():
    for x, y in zip(train_x, train_y):
        yield x, y

def val_generator():
    for x, y in zip(val_x, val_y):
        yield x, y


# callbacks

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = os.path.join(ckpt_dir, model_size,
                               "checkpoint-{epoch:04d}.ckpt")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True, save_freq=3)

def scheduler(epoch):
    if epoch < 35:
        return 0.001
    elif epoch < 50:
        return 0.0005
    else:
        return 0.0005/2

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


# train & models

train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

tf.keras.backend.set_session(train_sess)
with train_graph.as_default():
    config = tdnn_config(model_size)
    train_model = make_tdnn_model(config, n_labels=1211)

    train_ds = tf.data.Dataset.from_generator(train_generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=((500, 1, 65), ()))
    train_ds = train_ds.shuffle(buffer_size=134000)
    train_ds = train_ds.repeat()
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)

    val_ds = tf.data.Dataset.from_generator(val_generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=((500, 1, 65), ()))
    val_ds = val_ds.batch(batch_size)

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    train_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    train_model.fit(train_ds, epochs=n_epochs, steps_per_epoch=steps_per_epoch,
            callbacks = [cp_callback, lr_callback],
            validation_data=val_ds, validation_steps=10,
            verbose=1)

    # save graph and checkpoints
    saver = tf.train.Saver()
    save_file = os.path.join(ckpt_dir, 'checkpoints')
    if not os.path.isfile(save_file):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
    saver.save(train_sess, os.path.join(ckpt_dir, 'checkpoints'))

# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

tf.keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    tf.keras.backend.set_learning_phase(0)
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_model = make_tdnn_model(config, n_labels=1211)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, save_file)

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
    )

    with open(os.path.join(ckpt_dir, 'frozen_model.pb'), 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

converter = tf.lite.TFLiteConverter.from_frozen_graph(
    os.path.join(ckpt_dir, 'frozen_model.pb'), [eval_model.input.op.name],
    [eval_model.output.op.name], {eval_model.input.op.name: (batch_size, 500, 1, 65)}
)
# conversion quant_aware model to fully quantized model
converter.inference_type = tf.uint8
converter.inference_input_type = tf.int8
converter.quantized_input_stats = {eval_model.input.op.name: (0., 1.)}
tflite_model = converter.convert()

with open(os.path.join(ckpt_dir, 'quant_aware_tflite.h5'), 'wb') as f:
    f.write(tflite_model)

