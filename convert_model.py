import tensorflow as tf

def append_suffix_path(path, suffix):
    file_name = path.rstrip(".tflite")
    suffixed_file_name = file_name + "_" + suffix

    return suffixed_file_name + ".tflite"


def convert_to_tflite(tf_file, tflite_file, custom_objects_=None, input_shapes_=None):
    """
    transform tf_model to tflite_model
    :param tf_file:
    :param tflite_file:
    :param custom_objects_:
    :return:
    """
    # Convert to TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.\
        from_keras_model_file(tf_file,
                              custom_objects=custom_objects_,
                              input_shapes=input_shapes_)

    # TFLite conversion without quantization
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)

    return converter

def convert_to_quant(tf_file, tflite_file, custom_objects_=None,
                     act_quant=False, repr_data=None):
    """
    transform tf_model to tflite_quant model
    :param tf_file:
    :param tflite_file:
    :param custom_objects_: dict of custom layers {"layer_name": layer class}
    :param act_quant: flag for activation quantization (should provide repr_data)
    :param repr_data: representative dataset being used to quantize activations
    :return:
    """

    tflite_quant_file = append_suffix_path(tflite_file, "quant")
    tflite_quant_int_file = append_suffix_path(tflite_file, "quant_int")
    tf.logging.set_verbosity(tf.logging.INFO)
    converter = convert_to_tflite(tf_file, tflite_file, custom_objects_)

    if act_quant and not repr_data:
        raise NotImplementedError

    if act_quant:
        # post-training integer quantization (quantize weights and activation)
        repr_data = tf.cast(repr_data, tf.float32)
        repr_ds = tf.data.Dataset.from_tensor_slices((repr_data)).batch(1)
        def representative_data_gen():
            for input_value in repr_ds.take(100):
                yield [input_value]

        converter.representative_dataset = representative_data_gen
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_int_model = converter.convert()
        open(tflite_quant_int_file, "wb").write(tflite_quant_int_model)
    else:
        # post-training quantization (quantize only weights)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        open(tflite_quant_file, "wb").write(tflite_quant_model)


