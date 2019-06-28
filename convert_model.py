import os
import numpy as np
import tensorflow as tf
import argparse

from tdnn_model import make_tdnn_model, tdnn_config, StatPooling


def append_suffix_path(path, suffix):
    file_name = path.rstrip(".tflite")
    suffixed_file_name = file_name + "_" + suffix

    return suffixed_file_name + ".tflite"

def convert_to_tflite(tf_file, tflite_file, custom_objects=None, input_shapes=None):
    """
    transform tf_model to tflite_model
    :param tf_file:
    :param tflite_file:
    :param custom_objects:
    :return:
    """
    # Convert to TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.\
    from_keras_model_file(tf_file,
                          custom_objects=custom_objects,
                          input_shapes=input_shapes)

    # TFLite conversion without quantization
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)

    return converter

def convert_to_quant(tf_file, tflite_quant_file, custom_objects=None,
        input_shapes=None, act_quant=False, repr_data_gen=None):
    """
    transform tf_model to tflite_quant model
    :param tf_file:
    :param tflite_file:
    :param custom_objects: dict of custom layers {"layer_name": layer class}
    :param act_quant: flag for activation quantization (should provide repr_data)
    :param repr_data: representative dataset being used to quantize activations
    :return:
    """
    # Convert to TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.\
        from_keras_model_file(tf_file,
                              custom_objects=custom_objects,
                              input_shapes=input_shapes)

    if act_quant and not repr_data_gen:
        raise NotImplementedError

    if act_quant:
        # post-training integer quantization (quantize weights and activation)

        converter.representative_dataset = repr_data_gen
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_int_model = converter.convert()
        open(tflite_quant_file, "wb").write(tflite_quant_int_model)
    else:
        # post-training quantization (quantize only weights)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        open(tflite_quant_file, "wb").write(tflite_quant_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("convert tf model to tflite model")
    parser.add_argument("-ckpt_dir", type=str, required=True)
    # parser.add_argument("-tflite_dir", type=str, required=True)
    parser.add_argument("-model_size", type=str, choices=['S', 'M', 'L'], required=True)
    parser.add_argument("-post_quant", action='store_true')
    parser.add_argument("-act_quant", action='store_true')
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    tflite_dir = os.path.join(args.ckpt_dir, 'tflite_model')
    model_size = args.model_size

    config = tdnn_config(model_size)
    model = make_tdnn_model(config, 1211)

    latest = tf.train.latest_checkpoint(ckpt_dir)
    model.load_weights(latest)

    for _ in range(6):
        model.pop()
    model.summary()

    tf_file = os.path.join(ckpt_dir, "tflite", "tf_model.h5")
    if not os.path.isfile(tf_file):
        os.makedirs(os.path.dirname(tf_file), exist_ok=True)
    model.save(tf_file)

    if args.post_quant:
        if args.act_quant:
            val_x = np.random.random((1000, 500, 1, 65))
            val_y = np.random.randint(0, 1211, (1000,))
            val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))

            # batch_size = 1
            batch_size = 1
            val_ds_ = val_ds.batch(batch_size)
            def representative_data_gen():
                for input_value, _ in val_ds_.take(200//batch_size):
                    yield [tf.cast(input_value, tf.float32)]

            output_file = os.path.join(ckpt_dir, 'tflite', 'tflite_quant_act_{}.h5'.format(batch_size))
            convert_to_quant(tf_file, output_file,
                             custom_objects={'StatPooling':StatPooling},
                             input_shapes={"conv2d_input":[batch_size, 500, 1, 65]},
                             act_quant=True, repr_data_gen=representative_data_gen
                             )

            # batch_size = 32
            batch_size = 32
            val_ds_ = val_ds.batch(batch_size)
            def representative_data_gen1():
                for input_value, _ in val_ds_.take(200//batch_size):
                    yield [tf.cast(input_value, tf.float32)]

            output_file = os.path.join(ckpt_dir, 'tflite', 'tflite_quant_act_{}.h5'.format(batch_size))
            convert_to_quant(tf_file, output_file,
                             custom_objects={'StatPooling':StatPooling},
                             input_shapes={"conv2d_input":[batch_size, 500, 1, 65]},
                             act_quant=True, repr_data_gen=representative_data_gen
                             )

        else:
            # batch_size = 1
            batch_size = 1
            output_file = os.path.join(ckpt_dir, 'tflite', 'tflite_quant_{}.h5'.format(batch_size))
            convert_to_quant(tf_file, output_file,
                             custom_objects={'StatPooling':StatPooling},
                             input_shapes={"conv2d_input":[batch_size, 500, 1, 65]}
                             )

            # batch_size = 32
            batch_size = 32
            output_file = os.path.join(ckpt_dir, 'tflite', 'tflite_quant_{}.h5'.format(batch_size))
            convert_to_quant(tf_file, output_file,
                             custom_objects={'StatPooling':StatPooling},
                             input_shapes={"conv2d_input":[batch_size, 500, 1, 65]}
                             )

    else:
        # batch_size = 1
        batch_size = 1
        output_file = os.path.join(ckpt_dir, 'tflite', 'tflite_{}.h5'.format(batch_size))
        convert_to_tflite(tf_file, output_file,
                          input_shapes={"conv2d_input":[batch_size, 500, 1, 65]},
                          custom_objects={'StatPooling':StatPooling}
                          )

        # batch_size = 32
        batch_size = 32
        output_file = os.path.join(ckpt_dir, 'tflite', 'tflite_{}.h5'.format(batch_size))
        convert_to_tflite(tf_file, output_file,
                          input_shapes={"conv2d_input":[batch_size, 500, 1, 65]},
                          custom_objects={'StatPooling':StatPooling}
                          )

    print("{} transformed to {}".format(latest, output_file))
