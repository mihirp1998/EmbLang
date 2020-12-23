import tensorflow as tf
import keras
# from keras.models import *
# from tf.keras.layers import *
import ipdb 
st = ipdb.set_trace

class VGG(tf.keras.Model):
	def __init__(self):
		super(VGG, self).__init__()
		IMAGE_ORDERING ="channels_last"
		# block 1
		self.conv2d_1_1 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)
		self.conv2d_1_2 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)
		self.maxpool_1 = tf.keras.layers.MaxPooling3D(2, strides=2, name='block1_pool', data_format=IMAGE_ORDERING)

		# block 2
		self.conv2d_2_1 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)
		self.conv2d_2_2 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)
		self.maxpool_2 =  tf.keras.layers.MaxPooling3D(2, strides=2, name='block2_pool', data_format=IMAGE_ORDERING)

		# block 3
		self.conv2d_3_1 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)
		self.conv2d_3_2 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)
		self.conv2d_3_3 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)
		# self.maxpool_3 =  tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1), name='block3_pool', data_format=IMAGE_ORDERING)

		# blcok 4
		self.conv2d_4_1 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)
		self.conv2d_4_2 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)
		self.conv2d_4_3 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)
		# self.maxpool_4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool', data_format=IMAGE_ORDERING)

		# # block 5
		# self.conv2d_5_1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)
		# self.conv2d_5_2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)
		# self.conv2d_5_3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)
		# self.maxpool_5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1), name='block5_pool', data_format=IMAGE_ORDERING)

	def call(self,img_input, input_height=224 ,  input_width=224):

		assert input_height%32 == 0
		assert input_width%32 == 0
		x = self.conv2d_1_1(img_input)
		x = self.conv2d_1_2(x)
		x = self.maxpool_1(x)
		f1 = x
		# Block 2
		x = self.conv2d_2_1(x)
		x = self.conv2d_2_2(x)
		x = self.maxpool_2(x)
		f2 = x

		# Block 3
		x = self.conv2d_3_1(x)
		x = self.conv2d_3_2(x)
		x = self.conv2d_3_3(x)
		# x = self.maxpool_3(x)
		f3 = x

		# Block 4
		x = self.conv2d_4_1(x)
		x = self.conv2d_4_2(x)
		x = self.conv2d_4_3(x)
		# x = self.maxpool_4(x)
		f4 = x

		# Block 5
		# x = self.conv2d_5_1(x)
		# x = self.conv2d_5_2(x)
		# x = self.conv2d_5_3(x)
		# x = self.maxpool_5(x)
		# f5 = x

		return img_input , [f1 , f2 , f3 , f4 , f4 ]

class FCN(tf.keras.Model):
	def __init__(self):
		super(FCN, self).__init__()
		self.encoder = VGG()
		n_classes =1
		IMAGE_ORDERING ="channels_last"
		self.conv2d_1 = tf.keras.layers.Conv3D( 2048 ,  7  , activation='relu' , padding='same', data_format=IMAGE_ORDERING)
		self.conv2d_2 =   tf.keras.layers.Conv3D( 2048 , 1  , activation='relu' , padding='same', data_format=IMAGE_ORDERING)
		self.conv2d_3 =  tf.keras.layers.Conv3D( n_classes ,  1  ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING)
		self.conv2d_transpose = tf.keras.layers.Conv3DTranspose( n_classes , kernel_size=4 ,  strides=4 , use_bias=False ,  data_format=IMAGE_ORDERING)
	
	@tf.function
	def call(self,x):
		img_input , levels = self.encoder(x)
		[f1 , f2 , f3 , f4 , f5 ] = levels 
		# print(f5.shape)
		o = self.conv2d_1(f5)
		# o = Dropout(0.5)(o)

		o = self.conv2d_2(o)
		# o = Dropout(0.5)(o)

		o = self.conv2d_3(o)
		# print(o.shape)

		o = self.conv2d_transpose(o)
		o = tf.keras.activations.sigmoid(o)

		return o

if __name__ == "__main__":
	fcn = FCN()
	val = fcn(tf.zeros([1,16,16,16,32]))
	print(val.shape)

