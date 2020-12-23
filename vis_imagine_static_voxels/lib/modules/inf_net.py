import tensorflow as tf
class InfNet(tf.keras.Model):
  def __init__(self, hiddim):
    super(InfNet, self).__init__()
    self.infer = tf.keras.Sequential([
                      tf.keras.layers.Conv3D(hiddim, 3, 1, "same"),
                      tf.keras.layers.Activation("elu"),
                      tf.keras.layers.Conv3D(hiddim//8, 3, 1, "same"),
                      tf.keras.layers.Activation("elu"),
                      tf.keras.layers.Conv3D(hiddim//8, 1, 1, "same")])

  @tf.function
  def call(self, x):
    x = tf.expand_dims(x,0)
    x = self.infer(x)
    return x