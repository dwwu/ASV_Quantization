#!/bin/sh

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./tco_converter graph_file output_file"
    exit
fi

graph_file=$1
output_file=$2

tflite_convert \
    --output_file=$output_file \
    --graph_def_file=$graph_file \
    --inference_type=FLOAT \
    --input_arrays=conv2d_input \
    --input_shape=1,28,28,1 \
    --output_arrays=dense_1/Softmax

    #--input_shape=1,500,1,65 \
