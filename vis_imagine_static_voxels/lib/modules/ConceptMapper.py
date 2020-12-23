
import tensorflow as tf
# tf.set_random_seed(1)
import ipdb
st = ipdb.set_trace
import constants as const
from collections import OrderedDict

class ConceptMapper(tf.keras.Model):
  def __init__(self, CDHW):
    super(ConceptMapper, self).__init__()
    C, D, H, W = CDHW[0], CDHW[1], CDHW[2], CDHW[3]
    self.mean_dictionary = tf.keras.layers.Dense(C*D*H*W, use_bias=False,name="mean_dictionary")
    self.std_dictionary  = tf.keras.layers.Dense(C*D*H*W, use_bias=False,name="std_dictionary")
    self.C, self.D, self.H, self.W = C, D, H, W

  @tf.function
  def call(self, x):
    word_mean = self.mean_dictionary(x)
    word_std  = self.std_dictionary(x)

    if self.H == 1 and self.W == 1:
      return [tf.reshape(word_mean,(-1, 1, 1, 1, self.C)), \
              tf.reshape(word_std,(-1, 1, 1, 1, self.C))]
    else:
      return [tf.reshape(word_mean,(-1, self.D, self.H, self.W, self.C)), \
              tf.reshape(word_std,(-1, self.D, self.H, self.W, self.C))]

class ConceptMapper_Cond(tf.keras.Model):
  def __init__(self, CDHW,latentdim=32):
    super(ConceptMapper_Cond, self).__init__()
    C, D, H, W = CDHW[0], CDHW[1], CDHW[2], CDHW[3]
    self.mean_dictionary = tf.keras.layers.Dense(C*D*H*W, use_bias=False,name="mean_dictionary")
    self.deconv1 = tf.keras.layers.Conv3DTranspose(latentdim,kernel_size= 4,strides=2,padding="same")
    self.deconv2 = tf.keras.layers.Conv3DTranspose(latentdim,kernel_size= 4,strides=2,padding="same")
    self.bn = tf.keras.layers.BatchNormalization()
    self.bn1 = tf.keras.layers.BatchNormalization()

    # self.std_dictionary  = tf.keras.layers.Dense(C*D*H*W, use_bias=False,name="std_dictionary")
    self.C, self.D, self.H, self.W = C, D, H, W

  @tf.function
  def call(self, x):
    z_dense = tf.nn.relu(self.bn(self.mean_dictionary(x)))
    z_dense = tf.reshape(z_dense, [1, self.D,self.H,self.W,self.C])
    net = tf.nn.relu(self.bn1(self.deconv1(z_dense)))
    net = tf.nn.sigmoid(self.deconv2(net))
    # word_std  = self.std_dictionary(x)
    # st()

    # if self.H == 1 and self.W == 1:
    #   return [tf.reshape(word_mean,(-1, 1, 1, 1, self.C)), \
    #           tf.reshape(word_std,(-1, 1, 1, 1, self.C))]
    # else:
    return net


           # net = tc(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
           #  net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
           #  net = tf.reshape(net, [self.batch_size, 7, 7, 128])
           #  net = tf.nn.relu(
           #      bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
           #         scope='g_bn3'))

           #  out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

def run():
  # x1 = [tf.random.normal([10,21]),tf.random.normal([10,21])] 
  x1 = tf.random.normal([10,21])
  import time
  s = time.time()
  c(x1)
  print(time.time() - s)

if __name__ == "__main__":
  c = ConceptMapper_Cond([16,4,4,4])
  for i in range(20):
    run()
  print(len(c.trainable_variables))



