import tensorflow as tf
import ipdb
st = ipdb.set_trace
class h_mean(tf.keras.Model):
	def __init__(self, latentdim):
		super(h_mean,self).__init__()
		self.h_mean_op = tf.keras.layers.Conv3D(latentdim,3,1,padding="same")
	
	@tf.function
	def call(self,x1):
		with tf.name_scope("h_mean") as scope:
			return self.h_mean_op(x1)

class h_var(tf.keras.Model):
	def __init__(self, latentdim):
		super(h_var,self).__init__()
		self.h_var_op = tf.keras.layers.Conv3D(latentdim,3,1,padding="same")
	
	@tf.function
	def call(self,x1):
		with tf.name_scope("h_var") as scope:
			return self.h_var_op(x1)


class h_mean_cond(tf.keras.Model):
	def __init__(self, latentdim=1):
		super(h_mean_cond,self).__init__()
		self.flatten =  tf.keras.layers.Flatten()
		self.dense = tf.keras.layers.Dense(latentdim, use_bias=False)
		# self.h_mean_op = tf.keras.layers.Conv3D(latentdim,3,1,padding="same")
	@tf.function
	def call(self,x1):
		with tf.name_scope("h_mean_cond") as scope:
			flat_val = self.flatten(tf.expand_dims(x1,0))
			return self.dense(flat_val)



class h_var_cond(tf.keras.Model):
	def __init__(self, latentdim=1):
		super(h_var_cond,self).__init__()
		self.flatten = tf.keras.layers.Flatten()
		self.dense = tf.keras.layers.Dense(latentdim, use_bias=False)
	
	@tf.function
	def call(self,x1):
		with tf.name_scope("h_var_cond") as scope:
			x1 = self.flatten(tf.expand_dims(x1,0))
			return self.dense(x1)
