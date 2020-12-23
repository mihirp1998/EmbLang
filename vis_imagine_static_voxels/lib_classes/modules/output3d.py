import tensorflow as tf
import ipdb
st = ipdb.set_trace
from tensorflow.python.platform import tf_logging
tf_logging.set_verbosity(tf_logging.DEBUG)

import constants as const

class Output3D_f2(tf.keras.Model):
    def __init__(self):
        super(Output3D_f2, self).__init__()
        d0 = 64
        if const.H == const.W == 128:
            #64 -> 32 -> 16 -> 8 -> 4 -> 1            
            dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
            ksizes = [4, 4, 4, 4, 4]
            strides = [2, 2, 2, 2, 4]
            paddings = ['SAME'] * 4 + ['VALID']
        elif const.H == const.W == 64:
            #32 -> 16 -> 8 -> 4 -> 1
            dims = [d0, 2*d0, 4*d0, 8*d0]
            ksizes = [4, 4, 4, 4]
            strides = [2, 2, 2, 4]
            paddings = ['SAME'] * 3 + ['VALID']

        nets_1 = []
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = tf.keras.Sequential([tf.keras.layers.Conv3D(dim, ksize, stride, padding)])
            nets_1.append(net)
        self.nets_1 = nets_1


        dims = dims[:-1][::-1] + [d0//2]
        ksizes = ksizes[::-1]
        strides = strides[::-1]
        paddings = paddings[::-1]

        nets_2 = []
        for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
            net = tf.keras.Sequential([tf.keras.layers.Conv3DTranspose(dim, ksize, stride, padding)])
            nets_2.append(net)
        self.nets_2 = nets_2


    @tf.function
    def call(self, inputs):
        with tf.name_scope('out_3d') as scope:
            outputs = []
            inputs = inputs[::-1]
            inputs_updated = inputs.pop()
            skipcons = [inputs_updated]
            for model in self.nets_1:
                inputs_updated = model(inputs_updated)
                if inputs:
                    inputs_updated += inputs.pop()
                skipcons.append(inputs_updated)

            skipcons.pop()

            for model in self.nets_2:
                inputs_updated = model(inputs_updated)
                if skipcons:
                    inputs_updated += skipcons.pop()
                outputs.append(inputs_updated)

            if const.H == const.W == 128:
                outputs.pop(0)

            return outputs



if __name__ =="__main__":
    e = Output3D_f2()
    a = tf.zeros([1,32,32,32,32])
    b = tf.zeros([1,16,16,16,64])
    c = tf.zeros([1,8,8,8,128])
    d = tf.zeros([1,4,4,4,256])
    val = e([a,b,c,d])

