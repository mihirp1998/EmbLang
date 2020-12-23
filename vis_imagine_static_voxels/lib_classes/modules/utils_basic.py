import tensorflow as tf
# import hyperparams as hyp
import os
import numpy as np
from os.path import isfile
from scipy.misc import imread, imsave, imresize
EPS = 1e-6
# do not import any other utils!

def print_shape(t):
    print( t.get_shape().as_list())

def normalize_single(d):
    dmin = tf.reduce_min(d)
    dmax = tf.reduce_max(d)
    d = (d-dmin)/(EPS+(dmax-dmin))
    return d

def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    return tf.map_fn(normalize_single, (d), dtype=tf.float32)

def meshgrid3D_py(H, W, D):
    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    z = np.linspace(0, D-1, D)
    xv, yv, zv = np.meshgrid(x, y, z)
    return xv, yv, zv

def meshgrid2D(B, Y, X):
    with tf.variable_scope("meshgrid2D"):

        grid_y = tf.linspace(0.0, Y-1, Y)
        grid_y = tf.reshape(grid_y, [1, Y, 1])
        grid_y = tf.tile(grid_y, [B, 1, X])
        
        grid_x = tf.linspace(0.0, X-1, X)
        grid_x = tf.reshape(grid_x, [1, 1, X])
        grid_x = tf.tile(grid_x, [B, Y, 1])

        # outputs are B x Y x X
        return grid_y, grid_x

def add_loss(total_loss, loss, coeff, name):
    tf.summary.scalar('unscaled_%s' % name, loss)
    tf.summary.scalar('scaled_%s' % name, coeff*loss)
    total_loss += coeff*loss
    return total_loss

def meshgrid3D(B, Y, X, Z):
    # returns a meshgrid sized B x Y x X x Z
    # this ordering makes sense since usually Y=height, X=width, Z=depth
    with tf.variable_scope("meshgrid3D"):

        grid_y = tf.linspace(0.0, Y-1, Y)
        grid_y = tf.reshape(grid_y, [1, Y, 1, 1])
        grid_y = tf.tile(grid_y, [B, 1, X, Z])
        
        grid_x = tf.linspace(0.0, X-1, X)
        grid_x = tf.reshape(grid_x, [1, 1, X, 1])
        grid_x = tf.tile(grid_x, [B, Y, 1, Z])

        grid_z = tf.linspace(0.0, Z-1, Z)
        grid_z = tf.reshape(grid_z, [1, 1, 1, Z])
        grid_z = tf.tile(grid_z, [B, Y, X, 1])
        
        return grid_y, grid_x, grid_z

def gridcloud3D(B, Y, X, Z):
    # we want to sample for each location in the grid
    grid_y, grid_x, grid_z = meshgrid3D(B, Y, X, Z)
    x = tf.reshape(grid_x, [B, -1])
    y = tf.reshape(grid_y, [B, -1])
    z = tf.reshape(grid_z, [B, -1])
    # these are B x N
    XYZ = tf.stack([x, y, z], axis=2)
    # this is B x N x 3
    return XYZ

    
def meshgrid4D(B, R, Y, X, Z):
    # returns a meshgrid sized B x R x Y x X x Z
    # this ordering makes sense since usually Y=height, X=width, Z=depth
    # and R is sometimes collapsed with B
    with tf.variable_scope("meshgrid4D"):

        grid_r = tf.linspace(0.0, R-1, R)
        grid_r = tf.reshape(grid_r, [1, R, 1, 1, 1])
        grid_r = tf.tile(grid_r, [B, 1, Y, X, Z])
        
        grid_y = tf.linspace(0.0, Y-1, Y)
        grid_y = tf.reshape(grid_y, [1, 1, Y, 1, 1])
        grid_y = tf.tile(grid_y, [B, R, 1, X, Z])
        
        grid_x = tf.linspace(0.0, X-1, X)
        grid_x = tf.reshape(grid_x, [1, 1, 1, X, 1])
        grid_x = tf.tile(grid_x, [B, R, Y, 1, Z])

        grid_z = tf.linspace(0.0, Z-1, Z)
        grid_z = tf.reshape(grid_z, [1, 1, 1, 1, Z])
        grid_z = tf.tile(grid_z, [B, R, Y, X, 1])
        
        return grid_r, grid_y, grid_x, grid_z
    
def assert_same_shape(a, b):
    # tf.assert_equal(a.shape, b.shape)
    a_shape = a.get_shape().as_list()
    b_shape = b.get_shape().as_list()
    assert(np.all(a_shape==b_shape))
    
def reduce_masked_sum(x, mask, do_assert=True):
    # x and mask are the same shape
    # returns []
    # if do_assert:
    #     assert_same_shape(x, mask)
    prod = x*mask
    numer = tf.reduce_sum(prod)
    return numer
    
def reduce_masked_mean(x, mask, axis=None, do_assert=True, keepdims=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes

    prod = x*mask
    numer = tf.reduce_sum(prod, axis=axis, keepdims=keepdims)
    denom = EPS+tf.reduce_sum(mask, axis=axis, keepdims=keepdims)
    mean = numer/denom
    return mean

def l1_on_chans(x):
    return tf.reduce_sum(tf.abs(x), 3, keepdims=True)

def l1_on_axis(x, axis):
    return tf.reduce_sum(tf.abs(x), axis, keepdims=True)

def huber_on_axis(x, axis, delta=1.0, keepdims=True):
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops
    abs_error = math_ops.abs(x)
    quadratic = math_ops.minimum(x, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = math_ops.subtract(abs_error, quadratic)
    losses = math_ops.add(
        math_ops.multiply(
            ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
            math_ops.multiply(quadratic, quadratic)),
        math_ops.multiply(delta, linear))
    losses = tf.reduce_sum(losses, axis, keepdims=keepdims)
    return losses

def sql2_on_chans(x):
    return tf.reduce_sum(tf.square(x), 3, keepdims=True)

def l2_on_chans(x):
    return tf.sqrt(sql2_on_chans(x))

def sql2_on_axis(x, axis, keepdims=True):
    return tf.reduce_sum(tf.square(x), axis, keepdims=keepdims)

def l2_on_axis(x, axis, keepdims=True):
    return tf.sqrt(sql2_on_axis(x, axis, keepdims=keepdims))

# def batch_l2_norm(x):
#     # x should be B x H x W
#     # calculates the norm along the W dimension
#     # returns size B x H
#     return tf.map_fn(l2_norm, (x), dtype=tf.float32)

# def batch_l1_norm(x):
#     # x should be B x H x W
#     # calculates the norm along the W dimension
#     # returns size B x H
#     return tf.map_fn(l1_norm, (x), dtype=tf.float32)

def l2_norm(x, keepdims=False):
    # gets l2 norm on last axis
    # if x is H x W, returns norm sized H
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=keepdims))
        
def l1_norm(x, keepdims=False):
    # gets l1 norm on last axis
    # if x is H x W, returns norm sized H
    return tf.reduce_sum(tf.abs(x), axis=-1, keepdims=keepdims)
    
def batch_gather(tensor, inds):
    # tensor is B x M x maybe more
    # inds is B x N (i.e., N indices to grab within each ex)
    # M > N
    # this gathers N values in each ex according to the inds
    return tf.map_fn(single_gather, (tensor, inds), dtype=tf.float32)

def batch_sort(tensor):
    # t is B x N
    # this sorts each tensor

    N = int(tensor.get_shape()[1])
    # assert(N==H*W) # otw problem in single_sort
    
    return tf.map_fn(single_sort, tensor, dtype=tf.float32)

def batch_gather_int32(tensor, inds):
    # t is B x whatever
    # inds is B x N
    # this gathers in each tensor using inds
    return tf.map_fn(single_gather, (tensor, inds), dtype=tf.int32)

def single_gather(value):
    tensor, inds = value
    return tf.gather(tensor, inds)

def single_sort(tensor):
    N = int(tensor.get_shape()[0])
    # TypeError: __int__ returned non-int (type NoneType)
    # N = H*W
    sorted_tensor, inds = tf.nn.top_k(tensor, k=N)
    return sorted_tensor
    
def sub2ind(height, width, y, x):
    # for this to work correctly, y and x should be int32
    # x = tf.cast(x, tf.int32)
    # y = tf.cast(y, tf.int32)
    return y*width + x

# def sub2ind3D(height, width, depth, y, x, z):
#     # for this to work correctly, y and x should be int32
    
#     # note that
#     # the x-dimension is the fastest,
#     # then y,
#     # then z.
#     # so when gathering/scattering with these inds, the tensor should be Z x Y x X
    
#     return z*height*width + y*width + x

def sub2ind3D_zyx(depth, height, width, d, h, w):
    # same as sub2ind3D, but inputs in zyx order
    # when gathering/scattering with these inds, the tensor should be Z x Y x X
    return d*height*width + h*width + w

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def argmax2D(tensor):
    # input format: B x H x W x C
    # assert rank(tensor) == 4
    B = int(tensor.get_shape()[0])
    H = int(tensor.get_shape()[1])
    W = int(tensor.get_shape()[2])
    C = int(tensor.get_shape()[3])
    
    # flatten the Tensor along the height and width axes
    flat_tensor = tf.reshape(tensor, [B, -1, C])
    
    # argmax of the flat tensor
    argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
    
    # convert the indices into 2D coordinates
    argmax_y = argmax // W # row
    argmax_x = argmax % W # col

    argmax_y = tf.reshape(argmax_y, [B, C])
    argmax_x = tf.reshape(argmax_x, [B, C])
    return argmax_y, argmax_x
    
def argmax3D(tensor):
    # input format: B x H x W x D x C
    # assert rank(tensor) == 5
    B = int(tensor.get_shape()[0])
    H = int(tensor.get_shape()[1])
    W = int(tensor.get_shape()[2])
    D = int(tensor.get_shape()[3])
    C = int(tensor.get_shape()[4])

    # this is easier for my brain if things are in ZYX order
    tensor = tf.transpose(tensor, perm=[0,3,1,2,4])
    # tensor is B x D x H x W x C
    
    # flatten the Tensor along the spatial axes
    flat_tensor = tf.reshape(tensor, [B, -1, C])
    
    # argmax of the flat tensor
    argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

    # convert the indices into 3D coordinates
    argmax_z = argmax // (H*W)
    argmax_y = (argmax % (H*W)) // W
    argmax_x = (argmax % (H*W)) % W

    argmax_z = tf.reshape(argmax_z, [B, C])
    argmax_y = tf.reshape(argmax_y, [B, C])
    argmax_x = tf.reshape(argmax_x, [B, C])
    return argmax_y, argmax_x, argmax_z
    
def rank(tensor):
    # return the rank of a Tensor
    return len(tensor.get_shape())
  
def matmul2(mat1, mat2):
    return tf.matmul(mat1, mat2)

def matmul3(mat1, mat2, mat3):
    return tf.matmul(mat1, tf.matmul(mat2, mat3))

def second_grad_1D(x, absolute=False, square=False):
    # x should be B x T x C

    # extract left, middle, right slices
    xl = x[:,  :-2, :]
    xm = x[:, 1:-1, :]
    xr = x[:, 2:, :]

    d2 = xr - 2.0*xm + xl
    
    if absolute:
        d2 = tf.abs(d2)
        d2 = tf.reduce_sum(d2, axis=2)
    if square:
        d2 = tf.square(d2)
        d2 = tf.reduce_sum(d2, axis=2)
    return d2

def get_median(tensor):
    # tensor is B x whatever
    # returns a vector sized [B], with the median of each tensor
    B = int(tensor.get_shape()[0])
    tensor_median = tf.map_fn(get_median_tensor_single, (tensor), dtype=tf.float32)
    # enforce the shape
    tensor_median = tf.reshape(tensor_median, [B])
    return tensor_median

def get_median_tensor_single(z):
    z = tf.reshape(z, [-1])
    def med(t):
        if t.size:
            t = np.median(t)
        else:
            t = 0.0
        t = np.float32(t)
        return t
    median = tf.py_func(med, [z], tf.float32)
    return median

def gradient2D(x, absolute=False, square=False):
    # x should be B x H x W x C
    dy = x[:, 1:, :, :] - x[:, :-1, :, :]
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    zeros = tf.zeros_like(x)
    zero_row = tf.expand_dims(zeros[:, 0, :, :], axis=1)
    zero_col = tf.expand_dims(zeros[:, :, 0, :], axis=2)
    dy = tf.concat([dy, zero_row], axis=1)
    dx = tf.concat([dx, zero_col], axis=2)
    if absolute:
        dx = tf.abs(dx)
        dy = tf.abs(dy)
    if square:
        dx = tf.square(dx)
        dy = tf.square(dy)
    return dx, dy

def gradient3D(x, absolute=False, square=False):
    # x should be B x H x W x D x C
    dy = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dz = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    zeros = tf.zeros_like(x)
    zero_y = zeros[:, 0:1, :, :]
    zero_x = zeros[:, :, 0:1, :]
    zero_z = zeros[:, :, :, 0:1]
    dy = tf.concat([dy, zero_y], axis=1)
    dx = tf.concat([dx, zero_x], axis=2)
    dz = tf.concat([dz, zero_z], axis=3)
    if absolute:
        dy = tf.abs(dy)
        dx = tf.abs(dx)
        dz = tf.abs(dz)
    if square:
        dy = tf.square(dy)
        dx = tf.square(dx)
        dz = tf.square(dz)
    return dy, dx, dz

def pack_seqdim(tensor, B):
    shapelist = tensor.get_shape().as_list()
    B_, S = shapelist[:2]
    # assert(B==B_)
    otherdims = shapelist[2:]
    tensor = tf.reshape(tensor, [B*S] + otherdims)
    return tensor
    
def unpack_seqdim(tensor, B):
    shapelist = tensor.get_shape().as_list()
    BS = shapelist[0]
    otherdims = shapelist[1:]
    # assert(BS % B == 0)
    S = int(BS/B)
    tensor = tf.reshape(tensor, [B, S] + otherdims)
    return tensor

def assert_batch_shape(tensor, B):
    shapelist = tensor.get_shape().as_list()
    otherdims = shapelist[1:]
    tensor = tf.reshape(tensor, [B] + otherdims)
    return tensor
    
