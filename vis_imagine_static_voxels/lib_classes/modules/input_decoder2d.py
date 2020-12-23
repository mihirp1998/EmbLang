import tensorflow as tf
import ipdb
st = ipdb.set_trace
from tensorflow.python.platform import tf_logging
tf_logging.set_verbosity(tf_logging.DEBUG)

import constants as const

class InputDecoder_f3(tf.keras.Model):
    def __init__(self):
        super(InputDecoder_f3, self).__init__()
        if const.W == const.H == 128:
            K = 8
        elif const.W == const.H == 64:
            K = 4

        if const.W == const.H == 128:
            raise Exception('Find out the correct value for S')
            S = 64
            dim = 4096//S
        elif const.W == const.H == 64:
            S = 32
            dim = 2048//S
      
        self.pool = tf.keras.layers.MaxPool3D(pool_size=(1,1,K), strides=(1,1,K), padding='SAME')
        self.net = tf.keras.layers.Conv2D(dim, 1, 1, 'SAME')

    @tf.function
    def call(self, inputs):
        with tf.name_scope('in_dec_2d') as scope:
            output = self.pool(inputs)
            bs, h, w, d_, c = map(int, output.shape)
            output = tf.reshape(output, (bs, h, w, d_*c))
            output = self.net(output)
            return output



if __name__ =="__main__":
    e = InputDecoder_f3()
    val = e(tf.zeros([1,128,128,128,3]))

