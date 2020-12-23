import tensorflow as tf
import ipdb
st = ipdb.set_trace
from tensorflow.python.platform import tf_logging
tf_logging.set_verbosity(tf_logging.DEBUG)

class OutputEncoder_f1(tf.keras.Model):
    def __init__(self):
        super(OutputEncoder_f1, self).__init__()
        dims = [32, 64, 128, 256]
        ksizes = [3, 3, 3, 3]
        nets = []
        for i, (dim, ksize) in enumerate(zip(dims, ksizes)):
            net = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(dim, ksize, 2, activation=tf.nn.relu, padding='SAME'),
                tf.keras.layers.BatchNormalization(axis=3),
                 tf.keras.layers.Conv2D(dim, ksize, 1, activation=tf.nn.relu, padding='SAME'),
                 tf.keras.layers.BatchNormalization(axis=3)])
            nets.append(net)
        self.nets = nets

    @tf.function
    def call(self, inputs):
        with tf.name_scope('out_enc_2d') as scope:
            outputs = []
            for model in self.nets:
                inputs = model(inputs)
                outputs.append(inputs)
            return outputs



if __name__ =="__main__":
    e = OutputEncoder_f1()
    val = e(tf.zeros([6,64,64,3]))
