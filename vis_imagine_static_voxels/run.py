import tensorflow as tf

w = tf.Variable([[1.0]])
m =tf.zeros([1,1])
print(w.shape)
with tf.GradientTape() as tape:
  k = m * w
  loss = tf.raw_ops.TensorStridedSliceUpdate(input=tf.zeros([1,1]),begin=[0,0],end=[1,1],strides=[1,1],value=k)
grad = tape.gradient(loss, w)
print(grads)
