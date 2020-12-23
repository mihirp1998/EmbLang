from lib_classes.modules.embnet3 import embnet3
import tensorflow as tf
net = embnet3(True)
val = net(tf.random.normal([4,32,32,32,8]),tf.random.normal([4,32,32,32,8]))
print(val)