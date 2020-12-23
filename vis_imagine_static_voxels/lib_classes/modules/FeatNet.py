# from scipy.misc import imsave
# import cv2
import tensorflow as tf
# import hyperparams as hyp

from lib_classes.modules.utils_basic import *
from lib_classes.modules import utils_improc
import ipdb 
st = ipdb.set_trace
# from encoder3D import *
# from encoder_half import *
# from encoder_decoder import *
# from encoder_decoder_normconv import *
# from encoder_normconv import *
# from encoder import *
# import utils_improc
# import utils_geom
# import utils_misc

# import tensorflow.contrib.slim as slim
# import numpy as np

EPS = 1e-4
class Encoder3D(tf.keras.Model):
    def __init__(self):
      super(Encoder3D, self).__init__()
      chans=64
      pred_dim = 32

      dims = [chans, 2*chans, 4*chans]
      strides = [2, 2, 2, 2]
      ksizes = [4, 4, 4, 4, 4]
      self.convolutions = []
      self.batchnorms = []
      for i, (dim, ksize, stride) in enumerate(zip(dims, ksizes, strides)):
        self.convolutions.append(tf.keras.layers.Conv3D(dim,ksize, strides=stride, padding='VALID', activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3)))
        self.batchnorms.append(tf.keras.layers.BatchNormalization())
      dims = dims[1:]
      dims = dims[::-1]
      ksizes = ksizes[::-1]
      strides = strides[::-1]
      self.deconvs = []
      self.batchnorms_deconv = []

      self.final_conv = tf.keras.layers.Conv3D(pred_dim, 1, strides=1, padding='VALID', activation=tf.nn.leaky_relu,\
          kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3))

      for i, (dim, ksize, stride) in enumerate(zip(dims, ksizes, strides)):
        self.deconvs.append(tf.keras.layers.Conv3DTranspose(dim, ksize, strides=stride,padding='SAME',\
            activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3)))
        self.batchnorms_deconv.append(tf.keras.layers.BatchNormalization())


    def call(self,inputs):
      feat_stack = []
      B, H, W, D, C = inputs.get_shape().as_list()
      # outputs = []
      feat = tf.identity(inputs)
      skipcons = [feat]
      for i in range(len(self.convolutions)):
          feat = tf.pad(feat, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
          feat = self.convolutions[i](feat)
          feat = self.batchnorms[i](feat)
          skipcons.append(feat)
      skipcons.pop() 

      for i in range(len(self.deconvs)):
          feat = self.deconvs[i](feat)
          if skipcons:
            feat = tf.concat([feat, skipcons.pop()], axis=-1)
          feat = self.batchnorms_deconv[i](feat)
      feat = self.final_conv(feat)
      return feat

class FeatNet(tf.keras.Model):
  def __init__(self):
    super(FeatNet, self).__init__()
    self.encoder3D = Encoder3D()

  @tf.function
  def call(self,inputs, istrain=True):
      B, H, W, D, C = inputs.get_shape().as_list()
      feats = self.encoder3D(inputs)
      feats = feats / (EPS + l2_on_axis(feats, axis=4))
      return feats

if __name__ == "__main__":
  ft = FeatNet()
  val = tf.zeros([1,64,64,64,3])
  ft(val)
