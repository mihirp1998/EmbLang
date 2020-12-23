import tensorflow as tf
import keras
# from keras.models import *
# from tf.keras.layers import *
import ipdb 
st = ipdb.set_trace
from keras.initializers import he_normal
dropout      = 0.5
weight_decay = 0.0001

class Classify(tf.keras.Model):
	def __init__(self):
		super(Classify, self).__init__()
		# self.encoder = VGG()
		n_classes =1
		IMAGE_ORDERING ="channels_last"

		# build model
		self.model = tf.keras.models.Sequential()

		# Block 1
		self.model.add(tf.keras.layers.Conv3D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block1_conv1'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.Conv3D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block1_conv2'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.MaxPooling3D(2, strides=2, name='block1_pool'))

		# Block 2
		self.model.add(tf.keras.layers.Conv3D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block2_conv1'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.Conv3D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block2_conv2'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.MaxPooling3D(2, strides=2, name='block2_pool'))

		# Block 3
		self.model.add(tf.keras.layers.Conv3D(256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block3_conv1'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.Conv3D(256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block3_conv2'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.Conv3D(256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block3_conv3'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.Conv3D(256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block3_conv4'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.MaxPooling3D(2, strides=1, name='block3_pool'))

		# Block 4
		# self.model.add(tf.keras.layers.Conv3D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block4_conv1'))
		# self.model.add(tf.keras.layers.BatchNormalization())
		# self.model.add(tf.keras.layers.Activation('relu'))
		# self.model.add(tf.keras.layers.Conv3D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block4_conv2'))
		# self.model.add(tf.keras.layers.BatchNormalization())
		# self.model.add(tf.keras.layers.Activation('relu'))
		# self.model.add(tf.keras.layers.Conv3D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block4_conv3'))
		# self.model.add(tf.keras.layers.BatchNormalization())
		# self.model.add(tf.keras.layers.Activation('relu'))
		# self.model.add(tf.keras.layers.Conv3D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block4_conv4'))
		# self.model.add(tf.keras.layers.BatchNormalization())
		# self.model.add(tf.keras.layers.Activation('relu'))
		# self.model.add(tf.keras.layers.MaxPooling3D(2, strides=1, name='block4_pool'))

		# Block 5
		# self.model.add(tf.keras.layers.Conv3D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block5_conv1'))
		# self.model.add(tf.keras.layers.BatchNormalization())
		# self.model.add(tf.keras.layers.Activation('relu'))
		# self.model.add(tf.keras.layers.Conv3D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block5_conv2'))
		# self.model.add(tf.keras.layers.BatchNormalization())
		# self.model.add(tf.keras.layers.Activation('relu'))
		# self.model.add(tf.keras.layers.Conv3D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block5_conv3'))
		# self.model.add(tf.keras.layers.BatchNormalization())
		# self.model.add(tf.keras.layers.Activation('relu'))
		# self.model.add(tf.keras.layers.Conv3D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='block5_conv4'))
		# self.model.add(tf.keras.layers.BatchNormalization())
		# self.model.add(tf.keras.layers.Activation('relu'))
		# self.model.add(tf.keras.layers.MaxPooling3D(2, strides=1, name='block5_pool'))

		# self.model modification for cifar-10
		self.model.add(tf.keras.layers.Flatten(name='flatten'))
		self.model.add(tf.keras.layers.Dense(4096, use_bias = True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='fc_cifa10'))
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.Dropout(dropout))
		self.model.add(tf.keras.layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='fc2'))  
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.Dropout(dropout))      
		self.model.add(tf.keras.layers.Dense(3, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='predictions_cifa10'))        
		self.model.add(tf.keras.layers.BatchNormalization())
		self.model.add(tf.keras.layers.Activation('softmax'))
		self.final_mask = tf.keras.layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='fc3')
		self.final_mask1 = tf.keras.layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='fc3')
		self.final_mask2 = tf.keras.layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.he_normal(), name='fc3')


	# @tf.function
	def call(self,x,loading):
		if loading:
			x = self.model(x)
		else:
			x = tf.keras.models.Sequential(self.model.layers[:-3])(x)
			x = self.final_mask(x)
			x = tf.keras.layers.Activation('relu')(x)
			x = self.final_mask1(x)
			x = tf.keras.layers.Activation('relu')(x)
			x = self.final_mask2(x)
			x = tf.keras.layers.Activation('sigmoid')(x)
		# st()
		return x

if __name__ == "__main__":
	fcn = FCN()
	val = fcn(tf.zeros([1,16,16,16,32]))
	print(val.shape)

