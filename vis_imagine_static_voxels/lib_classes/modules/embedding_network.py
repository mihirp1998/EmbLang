import tensorflow as tf
import ipdb
st = ipdb.set_trace
from tensorflow.python.platform import tf_logging
tf_logging.set_verbosity(tf_logging.DEBUG)

import constants as const

class embedding_network(tf.keras.Model):
    def __init__(self):
        super(embedding_network, self).__init__()
        dims = [const.embedding_size]*const.embedding_layers
        ksizes = [4]*4
        paddings = ['SAME']*4

        nets = []
        for i, (dim, ksize, padding) in enumerate(zip(dims, ksizes, paddings)):
            net = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(dim, ksize, activation=tf.nn.relu, padding=padding)])
            nets.append(net)
        self.nets = nets
        self.final_nets = tf.keras.layers.Conv2D(const.embedding_size, 1, padding='SAME')

    @tf.function
    def call(self, inputs):
        with tf.name_scope('embed_net') as scope:
            outputs = []
            for model in self.nets:
                inputs = model(inputs)
            outputs = self.final_nets(inputs)
            return outputs
