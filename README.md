# Lightweight_ASV

## Prerequisite
dataset: voxceleb1_fbank64

## Post-training quantization
**Workflow**
1. train model
> python train.py -ckpt_dir /tmp/training -model_size {S|M|L}
2. post_quantize and convert tf model to tflite model
> python post_quantize.py -ckpt_dir /tmp/training -model_size {S|M|L}

## Quantization-aware training
**Workflow**
1. train model
> python quant_train.py -ckpt_dir /tmp/training -model_size {S|M|L}
2. convert to tflite model
> ./tco_quant_converter.sh /tmp/frozen_model.pb /tmp/a.tflite

## Extract and Evaluate embeddings
**Workflow**
1. tflite_evaluate: check a tflite model's consistency
> python tflite_evaluate.py -tflite_file /tmp/a.tflite
2. tflite_extract: extract xvectors
> python tflite_extract.py -tflite_file /tmp/a.tflite -embed_file embeds/out.embed -output_node xvector/Relu -batch_size 1
3. score cosine similarity
> python sv_test_voxc1.py -embed_file embeds/out.embed
