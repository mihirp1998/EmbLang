import tensorflow as tf
import numpy as np
from keras import backend as K

class KLD(tf.keras.Model):
    def __init__(self):
        super(KLD, self).__init__()

    @tf.function
    def call(self, mu, var):
        kld = 0.5 * K.sum(K.exp(var) + K.square(mu) - 1. - var)
        return kld