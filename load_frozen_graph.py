import tensorflow as tf

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

model_filepath='quant_training0/frozen_model.pb'
with tf.gfile.GFile(model_filepath, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

print('Check out the input placeholders:')
nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
for node in nodes:
    print(node)

print('Check out all nodes:')
nodes = [n.name + ' => ' +  n.op for n in graph_def.node]
for node in nodes:
    print(node)
