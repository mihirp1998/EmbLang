import tensorflow as tf
from lib_classes.modules.conv_lstm import ConvLstm
import ipdb
from munch import Munch
st = ipdb.set_trace

import constants as const

class ConvLstmDecoder_f4(tf.keras.Model):
	def __init__(self):
		super(ConvLstmDecoder_f4,self).__init__()
		self.conv2d = tf.keras.layers.Conv2D(256,3,1,padding="same")
		self.conv2d_embed = tf.keras.layers.Conv2D(8,3,2,padding="same")
		if const.EMBEDDING_LOSS:
			self.convLstm = ConvLstm([['convLSTM', 128, 11, 4, 128]], stochastic = const.GQN3D_CONVLSTM_STOCHASTIC,weight_decay = 1E-5,is_training = const.mode == 'train',reuse = False,output_debug = False)
		else:
			self.convLstm = ConvLstm([['convLSTM', 128, 3, 4, 128]], stochastic = const.GQN3D_CONVLSTM_STOCHASTIC,weight_decay = 1E-5,is_training = const.mode == 'train',reuse = False,output_debug = False)


	def tanh01(self, x):
	    return (tf.keras.activations.tanh(x)+1)/2
	def poolorunpool(self, input_, targetsize):
		inputsize = input_.shape.as_list()[1]
		if inputsize == targetsize:
			return input_
		elif inputsize > targetsize:
			ratio = inputsize // targetsize
			return tf.nn.pool(
				input_,
				window_shape = [ratio, ratio],
				padding = 'SAME',
				pooling_type = 'AVG',
				strides = [ratio, ratio]
			)
		else: #inputsize < targetsize:
			ratio = targetsize // inputsize
			return tf.image.resize_nearest_neighbor(
				input_,
				tf.stack([inputsize * ratio]*2)
			)
		
	@tf.function
	def call(self,inputs,cam_posrot,output_image):
		with tf.name_scope('conv_lstm_dec') as scope:
			inputs_pooled = [self.poolorunpool(x, 16) for x in inputs]
			net = tf.concat(inputs_pooled, axis = -1)
			net = self.conv2d(net)
			out,extra = self.convLstm(net,cam_posrot,output_image)
			out_img = self.tanh01(out[:,:,:,:3])
			embedding = out[:,:,:,3:] if const.EMBEDDING_LOSS else tf.constant(0.0, dtype = tf.float32)
			if const.EMBEDDING_LOSS:
				embedding = self.conv2d_embed(embedding)

			return Munch(pred_view = out_img, embedding = embedding, kl = extra['kl_loss'])

	#     inputs = [utils.tfutil.self.poolorunpool(x, 16) for x in inputs]
	#     net = tf.concat(inputs, axis = -1)
	#     net = slim.conv2d(net, 256, [3, 3])

	#     dims = 3+const.EMBEDDING_LOSS * const.embedding_size
	#     out, extra = utils.fish_network.make_lstmConv(
	#         net,
	#         None,
	#         self.target,
	#         [['convLSTM', const.CONVLSTM_DIM, dims, const.CONVLSTM_STEPS, const.CONVLSTM_DIM]], 
	#         stochastic = const.GQN3D_CONVLSTM_STOCHASTIC,
	#         weight_decay = 1E-5,
	#         is_training = const.mode == 'train',
	#         reuse = False,
	#         output_debug = False,
	#     )

	#     out_img = utils.tfutil.self.tanh01(out[:,:,:,:3])
	#     embedding = out[:,:,:,3:] if const.EMBEDDING_LOSS else tf.constant(0.0, dtype = tf.float32)

	#     return Munch(pred_view = out_img, embedding = embedding, kl = extra['kl_loss'])
if __name__ == "__main__":
	c = ConvLstmDecoder_f4()
	val = c([tf.zeros([2,32,32,64])],None,tf.zeros([2,64,64,3]))
	print(val)

