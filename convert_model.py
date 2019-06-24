import tensorflow as tf
import argparse


def append_suffix_path(path, suffix):
    file_name = path.rstrip(".tflite")
    suffixed_file_name = file_name + "_" + suffix

    return suffixed_file_name + ".tflite"

def convert_to_tflite(tf_file, tflite_file, custom_objects=None,
        input_shapes=None):
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
        input_shapes=None,
        act_quant=False, repr_ds=None):
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

    if act_quant and not repr_ds:
        raise NotImplementedError

    if act_quant:
        # post-training integer quantization (quantize weights and activation)
        # repr_data = tf.cast(repr_data, tf.float32)
        # repr_ds = tf.data.Dataset.from_tensor_slices((repr_data)).batch(1)
        def representative_data_gen():
            for input_value, _ in repr_ds.take(10):
                yield [input_value]

        converter.representative_dataset = representative_data_gen
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_int_model = converter.convert()
        open(tflite_quant_file, "wb").write(tflite_quant_int_model)
    else:
        # post-training quantization (quantize only weights)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        open(tflite_quant_file, "wb").write(tflite_quant_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tf_file", type=str)
    parser.add_argument("tflite_file", type=str)
    args = parser.parse_args()




