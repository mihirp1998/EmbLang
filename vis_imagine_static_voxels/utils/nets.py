import tensorflow as tf
import constants as const
from . import tfpy
from . import voxel
from munch import Munch
from . import convlstm
from . import tfutil
from . import utils
from tensorflow import summary as summ
# import tensorflow.contrib.layers as tcl
# from layers import *

import ipdb
st = ipdb.set_trace
slim2 = Munch()

#some batch norm params...
bn_trainmode = ((const.mode != 'test') and (not const.rpvx_unsup))
if const.force_batchnorm_trainmode:
    bn_trainmode = True
if const.force_batchnorm_testmode:
    bn_trainmode = False

normalizer_params={
    'is_training': bn_trainmode, 
    'decay': const.bn_decay,
    'epsilon': 1e-3,
    'scale': False,
    'updates_collections': None
}

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tf.keras.layers.BatchNormalization(x), alpha)


def relu_batch_norm(x):
    return tf.nn.relu(tf.keras.layers.BatchNormalization(x))

def summary_wrap(func):
    def func_(*args, **kwargs):
        rval = func(*args, **kwargs)
        print('histogram for', rval.name)
        summ.histogram(rval.name, rval)
        return rval
    return func_

# if const.inject_summaries:
#     for funcname in ['conv2d', 'conv2d_transpose', 'conv3d', 'conv3d_transpose']:
#         func = getattr(slim, funcname)
#         setattr(slim2, funcname, summary_wrap(func))
# else:
#     slim2 = slim    

def unproject(inputs, resize = False):

    if resize:
        inputs = tf.image.resize(inputs, (const.S, const.S))
    size = int(inputs.shape[1])

    #now unproject, to get our starting point
    inputs = voxel.unproject_image(inputs)

    #in addition, add on a z-map, and a local bias
    #copied from components.py
    meshgridz = tf.range(size, dtype = tf.float32)
    meshgridz = tf.reshape(meshgridz, (1, size, 1, 1))
    meshgridz = tf.tile(meshgridz, (const.BS, 1, size, size))
    meshgridz = tf.expand_dims(meshgridz, axis = 4) 
    meshgridz = (meshgridz + 0.5) / (size/2) - 1.0 #now (-1,1)

    #get the rough outline
    unprojected_mask = utils.binarize(tf.expand_dims(inputs[:,:,:,:,0], 4), 1)
    # unprojected_depth = tf.expand_dims(inputs[:,:,:,:,1], 4)
    unprojected_depth = (tf.expand_dims(inputs[:,:,:,:,1], 4) - const.radius) * (1/const.SCENE_SIZE)

    if const.H > 32:
        outline_thickness = 0.1
    else:
        outline_thickness = 0.2
    # depth shell
    outline = tf.cast(tf.logical_and(
        unprojected_depth <= meshgridz,
        unprojected_depth + outline_thickness > meshgridz
    ), tf.float32)
    outline *= unprojected_mask
    if const.DEBUG_UNPROJECT:
        #return tf.expand_dims(inputs[:,:,:,:,0], 4) #this is the unprojected mask
        return outline

    inputs_ = [inputs]
    if const.USE_MESHGRID:
        inputs_.append(meshgridz)
    if const.USE_OUTLINE:
        inputs_.append(outline)
    inputs = tf.concat(inputs_, axis = 4)
    return inputs




def print_shape(t):
    print(t.name, t.get_shape().as_list())


# class batch_norm(object):
#     """Code modification of http://stackoverflow.com/a/33950177"""
#     def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
#         with tf.compat.v1.variable_scope(name):
#             self.epsilon = epsilon
#             self.momentum = momentum

#             self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
#             self.name = name

#     def __call__(self, x, train=True, b_reuse=False):
#         shape = x.get_shape().as_list()

#         if train:
#             with tf.compat.v1.variable_scope(self.name) as scope:
#                 self.beta = tf.compat.v1.get_variable("beta", [shape[-1]],
#                                     initializer=tf.compat.v1.initializers.constant(0.), use_resource=False)
#                 self.gamma = tf.compat.v1.get_variable("gamma", [shape[-1]],
#                                     initializer=tf.compat.v1.initializers.random_normal(1., 0.02), use_resource=False)

#                 # work around reuse=True problem
#                 with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
#                     batch_mean, batch_var = tf.nn.moments(x=x, axes=[0, 1, 2], name='moments')
#                     ema_apply_op = self.ema.apply([batch_mean, batch_var])
#                     self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

#                     with tf.control_dependencies([ema_apply_op]):
#                         mean, var = tf.identity(batch_mean), tf.identity(batch_var)

#         else:
#             mean, var = self.ema_mean, self.ema_var

#         normed = tf.nn.batch_norm_with_global_normalization(
#                 x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

#         return normed




# class Discriminator(object):
#     def __init__(self):
#         self.x_dim = 64 * 64 * 3
#         self.name = 'lsun/dcgan/d_net'

#     def __call__(self, x, reuse=True):
#         with tf.compat.v1.variable_scope(self.name) as vs:
#             if reuse:
#                 vs.reuse_variables()
#             bs = tf.shape(input=x)[0]
#             x = tf.reshape(x, [bs, 64, 64, 3])
#             conv1 = tf.keras.layers.Conv2D(x, 64, 4, 2,
#                 weights_initializer=tf.compat.v1.initializers.random_normal(stddev=0.02),
#                 activation=leaky_relu
#             )
#             conv2 = tf.keras.layers.Conv2D(conv1, 128, 4, 2,
#                 weights_initializer=tf.compat.v1.initializers.random_normal(stddev=0.02),
#                 activation=leaky_relu_batch_norm
#             )
#             conv3 = tf.keras.layers.Conv2D(conv2, 256, 4, 2,
#                 weights_initializer=tf.compat.v1.initializers.random_normal(stddev=0.02),
#                 activation=leaky_relu_batch_norm
#             )
#             conv4 = tf.keras.layers.Conv2D(conv3, 512, 4, 2,
#                 weights_initializer=tf.compat.v1.initializers.random_normal(stddev=0.02),
#                 activation=leaky_relu_batch_norm
#             )
#             conv4 = tf.keras.layers.flatten(conv4)
#             fc = tf.keras.layers.dense(conv4)
#             return fc

#     @property
#     def vars(self):
#         return [var for var in tf.compat.v1.global_variables() if self.name in var.name]

# class discrimiantor:
#     def __init__():
#         self.bn1 = batch_norm(name="bn1")
#         self.bn2 = batch_norm(name="bn2")
#         self.bn3 = batch_norm(name="bn3")
#         self.bn4 = batch_norm(name="bn4")

#     def lrelu(self,x, leak=0.2, name="lrelu"):
#         with tf.variable_scope(name):
#             f1 = 0.5 * (1 + leak)
#             f2 = 0.5 * (1 - leak)
#             return f1 * x + f2 * abs(x)
#     # CHANGE THIS is train true always
#     def create_discriminator(self,inputs, b_reuse=False,is_train=True):
#         with slim.arg_scope([slim.conv2d],
#                         padding='SAME',
#                         activation_fn=None,
#                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#                         weights_regularizer=slim.l2_regularizer(0.0005),
#                         stride=2):
#             # disc_bn2,disc_bn3,disc_bn4,disc_bn5 = disc_bn
#             noisy_inputs = inputs

#             disc = self.lrelu(slim.conv2d(noisy_inputs, 64, [5, 5], scope='conv1'))
#             print_shape(disc)

#             disc = self.lrelu(batch_norm(slim.conv2d(disc, 128, [5, 5], scope='conv2'), reuse=b_reuse,train=is_train))
#             print_shape(disc)

#             disc = self.lrelu(batch_norm(slim.conv2d(disc, 256, [5, 5], scope='conv3'), reuse=b_reuse,train=is_train))
#             print_shape(disc)

#             disc = self.lrelu(batch_norm(slim.conv2d(disc, 512, [5, 5], scope='conv4'), reuse=b_reuse,train=is_train))
#             print_shape(disc)

#             disc = self.lrelu(batch_norm(slim.conv2d(disc, 512, [5, 5], scope='conv5'), reuse=b_reuse,train=is_train))
#             print_shape(disc)

#         disc = self.lrelu(slim.fully_connected(tf.contrib.layers.flatten(disc), 1024, activation_fn=None, scope='fc6'))
#         disc_logits = slim.fully_connected(tf.contrib.layers.flatten(disc), 1, activation_fn=None, scope='fc7')

#         return disc_logits

# def voxel_net_3d(inputs, aux = None, bn = True, outsize = 128, d0 = 16):
#     #aux is used for the category input

#     #for backward compatibility
#     # normalizer_params={'is_training': bn_trainmode, 
#     #                    'decay': 0.999,
#     #                    'epsilon': 1e-5,
#     #                    'scale': True,
#     #                    'updates_collections': None}

#     normalizer_params2 = {k:v for (k,v) in normalizer_params.items()}
#     normalizer_params2[scale] = True
#     normalizer_params2[epsilon] = 1E-5
#     st()
    
#     with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
#                         activation_fn=tf.nn.relu,
#                         normalizer_fn=slim.batch_norm if bn else None,
#                         normalizer_params=normalizer_params2
#                         ):
        
#         dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
#         ksizes = [4, 4, 4, 4, 8]
#         strides = [2, 2, 2, 2, 1]
#         paddings = ['SAME'] * 4 + ['VALID']

#         if const.S == 64:
#             ksizes[-1] = 4
#         elif const.S == 32:
#             ksizes[-1] = 2

#         net = inputs

#         skipcons = [net]
#         for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
#             net = slim2.conv3d(net, dim, ksize, stride=stride, padding=padding)

#             skipcons.append(net)
        
#         #BS x 4 x 4 x 4 x 256

#         if aux is not None:
#             aux = tf.reshape(aux, (const.BS, 1, 1, 1, -1))
#             net = tf.concat([aux, net], axis = 4)

#         #fix from here..
#         chans = [8*d0, 4*d0, 2*d0, d0, 1]
#         strides = [1, 2, 2, 2, 2]
#         ksizes = [8, 4, 4, 4, 4]
#         paddings = ['VALID'] + ['SAME'] * 4
#         activation_fns = [tf.nn.relu] * 4 + [None] #important to have the last be none

#         if const.S == 64:
#             ksizes[0] = 4
#         elif const.S == 32:
#             ksizes[0] = 2
        
#         decoder_trainable = not const.rpvx_unsup #don't ruin the decoder by FTing

#         skipcons.pop() #we don't want the innermost layer as skipcon

#         foo = lambda: None
#         #we'll use this object to smuggle additional information out
        
#         for i, (chan, stride, ksize, padding, activation_fn) \
#             in enumerate(zip(chans, strides, ksizes, paddings, activation_fns)):

#             # if i == len(chans)-1:
#             #     norm_fn = None
#             # else:
#             norm_fn = slim.batch_norm

#             net = slim2.conv3d_transpose(
#                 net, chan, ksize, stride=stride, padding=padding, activation_fn = activation_fn,
#                 normalizer_fn = norm_fn, trainable = decoder_trainable
#             )

#             #now concatenate on the skip-connection
#             net = tf.concat([net, skipcons.pop()], axis = 4)

#         #one last 1x1 conv to get the right number of output channels
#         net = slim2.conv3d(
#             net, 1, 1, 1, padding='SAME', activation_fn = None,
#             normalizer_fn = None, trainable = decoder_trainable
#         )

#     logit = net
#     net = tf.nn.sigmoid(net)

#     return Munch(pred = net, features = foo, logit = logit)

# def convgru(*args, scopename='convgru', **kwargs):
#     with tf.compat.v1.variable_scope(scopename, reuse = False):
#         return convgru_(*args, **kwargs)

# def convgru_(inputs, kernel=[3, 3, 3], filters=32):
#     bs, l, h, w, d, ch = inputs.get_shape().as_list()

#     conv_gru = convlstm.ConvGRUCell(
#         shape=[h, w, d],
#         initializer=slim.initializers.xavier_initializer(),
#         kernel=kernel,
#         filters=filters
#     )
    
#     seq_length = [l]*bs
    
#     outputs, state = tf.compat.v1.nn.dynamic_rnn(
#         conv_gru,
#         inputs,
#         parallel_iterations=1,
#         sequence_length=seq_length,
#         dtype=tf.float32,
#         time_major=False,
#         swap_memory = True,
#     )
    
#     return Munch(outputs = outputs, state = state)

# def gru_aggregator(inputs):
#     inputs = tf.stack(inputs, axis = 1)
#     layers = []

#     for i in range(2):
#         layers.append(convgru(inputs, kernel = [5,5,5], filters = 12, scopename='convgru_%d' % i))
#         inputs = layers[-1].outputs

#     return sum((l.state for l in layers))
    #return tf.concat([l.state for l in layers], axis = -1)

# def encoder3D(inputs, aux = None, d0 = 16):
#     #BATCH NORM IS BROKEN HERE
#     raise Exception('you should probably not be using this')

#     net = inputs    
#     with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
#                         activation_fn=tf.nn.relu,
#                         normalizer_fn=slim.batch_norm):
#         dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
#         ksizes = [4, 4, 4, 4, 8]
#         strides = [2, 2, 2, 2, 1] 
#         paddings = ['SAME'] * 4 + ['VALID']

#         skipcons = [net]
        
#         for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
#             net = slim2.conv3d(net, dim, ksize, stride=stride, padding=padding)
#             skipcons.append(net)
        
#         if aux is not None:
#             aux = tf.reshape(aux, (const.BS, 1, 1, 1, -1))
#             net = tf.concat([aux, net], axis = 4)

#         chans = [8*d0, 4*d0, 2*d0, d0, 1]
#         strides = [1, 2, 2, 2, 2]
#         ksizes = [8, 4, 4, 4, 4]
#         paddings = ['VALID'] + ['SAME'] * 4
#         activation_fns = [tf.nn.relu] * 5

#         skipcons.pop() #we don't want the innermost layer as skipcon
#         skipcons.pop(0)

#         #skipcons contains feature tensors of sizes: 64, 32, 16, 8
#         return skipcons


# def depth_channel_net_v2(feature):
#     # pool by 8 on the depth axis
#     # then flatten last two axes and do a 1x1 conv

#     if False:
#         return DRC(feature)
    
#     if const.W == const.H == 128:
#         K = 8
#     elif const.W == const.H == 64:
#         K = 4
        
#     feature = tf.nn.pool(input=feature, window_shape=[1,1,K], pooling_type='MAX', padding='SAME', strides = [1,1,K])
    
#     bs, h, w, d_, c = map(int, feature.shape)
#     feature = tf.reshape(feature, (bs, h, w, d_*c))

#     S = voxel.getsize(feature)

#     if const.W == const.H == 128:
#         dim = 4096//S
#     elif const.W == const.H == 64:
#         dim = 2048//S

#     with tf.compat.v1.variable_scope("depthconv_%d" % S):
#         feature = slim2.conv2d(feature, dim, 1, stride=1, padding='SAME')

#     return feature

# foo = None
# def DRC(feature):
#     global foo
#     #take 1x1 conv to get occupancy
#     occ = slim2.conv3d(feature, 1, 1, activation_fn = None)
#     occ = tf.nn.sigmoid(occ - 0.5) #start with a low bias!!
#     not_occ = 1.0-occ
#     not_occ_cum = tf.math.cumprod(not_occ, axis = 3, exclusive = True)
#     hit_here = not_occ_cum * occ

#     if occ.shape.as_list()[1] == 32:
#         print('asdf')
#         foo = occ
#     return tf.reduce_sum(input_tensor=hit_here * feature, axis = 3)
    
# def decoder2D(features, apply_3to2 = True):
#     features = features[::-1]
#     if apply_3to2:
#         features = [depth_channel_net_v2(feature) for feature in features]

#     if True:
#         # dims = [256, 128, 64, 32] #original
#         dims = [32]
#     else:
#         dims = [128, 64, 32, 16] #with feature projection
#     # CHANGE THIS
#     ksizes = [4]*1
#     strides = [2]*1
#     paddings = ['SAME']*1

#     with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
#                         activation_fn=tf.nn.relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params=normalizer_params):

#         net = 0
#         for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
#             if features:
#                 net += features.pop()
                
#             #net = slim2.conv2d_transpose(net, dim, ksize, stride = stride, padding = padding)
#             #net = slim2.conv2d(net, dim, ksize, stride = 1, padding = 'SAME')

#             net = slim2.conv2d(net, dim, ksize, stride = 1, padding = padding)
#             net = tf.image.resize(net, [net.shape.as_list()[1]*2]*2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#     net = slim2.conv2d(net, 3, 3, stride = 1, padding = 'SAME', activation_fn = None)
#     net = tfutil.tanh01(net)
#     return net

# def encoder2D(net):
#     #return in 64, 32, 16, and 8

#     outputs = []

#     with slim.arg_scope([slim.conv2d],
#                         activation_fn=tf.nn.relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params = normalizer_params,
#                         padding = 'SAME'):
        
#         dims = [32, 64, 128, 256]
#         ksizes = [3, 3, 3, 3]

#         for i, (dim, ksize) in enumerate(zip(dims, ksizes)):
#             net = slim2.conv2d(net, dim, ksize, stride=2)
#             net = slim2.conv2d(net, dim, ksize, stride=1)
#             outputs.append(net)

#     return outputs


# def encoder_decoder3D(inputs, aux = None):
#     inputs = inputs[::-1]

#     d0 = 64
#     outputs = []
#     net = inputs.pop()
#     skipcons = [net]

#     with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
#                         activation_fn=tf.nn.relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params = normalizer_params):

#         if const.H == const.W == 128:
#             #64 -> 32 -> 16 -> 8 -> 4 -> 1            
#             dims = [d0, 2*d0, 4*d0, 8*d0, 16*d0]
#             ksizes = [4, 4, 4, 4, 4]
#             strides = [2, 2, 2, 2, 4]
#             paddings = ['SAME'] * 4 + ['VALID']
#         elif const.H == const.W == 64:
#             #32 -> 16 -> 8 -> 4 -> 1
#             dims = [d0, 2*d0, 4*d0, 8*d0]
#             ksizes = [4, 4, 4, 4]
#             strides = [2, 2, 2, 4]
#             paddings = ['SAME'] * 3 + ['VALID']

#         for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
#             net = slim2.conv3d(net, dim, ksize, stride=stride, padding=padding)
#             if inputs:
#                 #net = tf.concat([net, inputs.pop()], axis = -1)
#                 net += inputs.pop()
#             skipcons.append(net)

#         #return skipcons[:-2][::-1] #if we can also truncate the encoder here
#         skipcons.pop() #we don't want the innermost layer as skipcon

#         #1 -> 4 -> 8 -> 16 -> 32 -> 64
#         dims = dims[:-1][::-1] + [d0//2]
#         ksizes = ksizes[::-1]
#         strides = strides[::-1]
#         paddings = paddings[::-1]
        
#         for i, (dim, ksize, stride, padding) in enumerate(zip(dims, ksizes, strides, paddings)):
#             net = slim2.conv3d_transpose(net, dim, ksize, stride=stride, padding=padding)
#             if skipcons:
#                 net += skipcons.pop()
#                 #net = tf.concat([net, skipcons.pop()], axis = -1)
#             outputs.append(net)

#     if const.H == const.W == 128:
#         outputs.pop(0) #don't want 4x4x4

#     return outputs

# def MnistAE(net):
#     with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
#                         activation_fn = tf.nn.relu,
#                         normalizer_fn = slim.batch_norm,
#                         padding = 'SAME'):
        
#         net = slim2.conv2d(net, 8, [3,3], stride = [2,2])
#         net = slim2.conv2d(net, 16, [3,3], stride = [2,2])
#         net = slim2.conv2d(net, 64, [7,7], stride = [1,1], padding = 'VALID')
        
#         net = slim2.conv2d_transpose(net, 16, [7, 7], stride = [1, 1], padding= 'VALID')
#         net = slim2.conv2d_transpose(net, 8, [3,3], stride = [2,2])
#         net = slim2.conv2d_transpose(net, 1, [3,3], stride = [2,2],
#                                      normalizer_fn = None, activation_fn = None)
        
#     return tf.nn.sigmoid(net)
        
    
# def MnistAEconvlstm(x):
#     from .network_down import make_lstmConv

#     net = x
#     net = slim2.conv2d(net, 8, [3,3], stride = [2,2])
#     net = slim2.conv2d(net, 16, [3,3], stride = [2,2])
#     net = slim2.conv2d(net, 64, [7,7], stride = [1,1], padding = 'VALID')
#     #B x 1 x 1 x 64

#     out, extra = make_lstmConv(
#         net,
#         None, 
#         x, 
#         [['convLSTM', 64, 1, 8, 64]],
#         stochastic = const.MNIST_CONVLSTM_STOCHASTIC,
#         weight_decay = 0.0001,
#         is_training = True,
#         reuse = False,
#         output_debug = False
#     )
#     out = tf.nn.sigmoid(out)

#     out.loss = extra['kl_loss']
#     return out

# def embedding_network(img):
#     #input: BxHxWx3
#     #output: BxHxWxC

#     dims = [const.embedding_size]*const.embedding_layers
#     ksizes = [4]*4
#     paddings = ['SAME']*4

#     with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
#                         activation_fn=tf.nn.relu,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params = normalizer_params):

#         embedding = img
#         for i, (dim, ksize, padding) in enumerate(zip(dims, ksizes, paddings)):
#             embedding = slim2.conv2d(embedding, dim, ksize, stride = 1, padding = padding)
            
#         embedding = slim2.conv2d(embedding, const.embedding_size, 1, stride = 1, activation_fn = None)

#     return embedding

#fix in nets.py as well
