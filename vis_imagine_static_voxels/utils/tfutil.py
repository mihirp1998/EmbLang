import tensorflow as tf
import constants as const
import ipdb
st = ipdb.set_trace

def current_scope():
    return tf.compat.v1.get_variable_scope().name


def current_scope_and_vars():
    scope = current_scope()

    #need both!
    collections = [tf.compat.v1.GraphKeys.MODEL_VARIABLES, tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES]

    vars_ = []
    for collection in collections:
        vars_.extend(tf.compat.v1.get_collection(collection, scope))

    vars_ = list(set(vars_))

    #for z in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope):
    #    assert z in vars_
        
    return (scope, vars_)


def add_scope_to_dct(dct, name):
    dct[name] = current_scope_and_vars()


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64s_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def leaky_relu(alpha):
    return lambda x: tf.nn.relu(x) + tf.nn.relu(-x) * alpha


def upscale(feats):
    _shape = feats.get_shape()
    h = _shape[1]
    w = _shape[2]
    return tf.image.resize(feats, tf.stack([h * 2, w * 2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def mv_unstack(x):
    return tf.reshape(x, [const.BS, const.H, const.W, const.V, -1])


def mv_stack(x):
    return tf.reshape(x, [const.BS, const.H, const.W, -1])


def mv_shape4(chans=1):
    return (const.BS, const.H, const.W, const.V * chans)


def mv_shape5(chans=1):
    return (const.BS, const.H, const.W, const.V, chans)


def rand_elem(tensor, axis):
    n = int(tensor.get_shape()[axis])
    r = tf.rank(tensor)
    idx = tf.random.categorical(logits=[[1.0] * n], num_samples=1)[0, 0]
    idx_ = [0] * axis + idx + [0] * r - axis - 1
    size = [-1] * axis + 1 + [-1] * r - axis - 1
    return tf.slice(tensor, tf.stack(idx_), size)


def tf_random_bg(N, darker = False):
    color = tf.random.uniform((3,))
    if darker:
        color /= 2.0
    color = tf.tile(tf.reshape(color, (1, 1, 1, 3)), (N, const.Hdata, const.Wdata, 1))
    return color

def add_feat_to_img(img, feat):
    # img is BS x H x W x C
    # feat is BS x D
    # output is BS x H x W x (C+D)

    bs, h, w, _ = list(map(int, img.get_shape()))
    feat = tf.reshape(feat, (bs, 1, 1, -1))
    tilefeat = tf.tile(feat, (1, h, w, 1))
    return tf.concat([img, tilefeat], axis=3)


def cycle(tensor, idx, axis):
    r = len(tensor.get_shape())
    n = tensor.get_shape()[axis]
    # move idx elements from the front to the back on axis
    start_idx = [0] * r

    head_size = [-1]*r
    head_size[axis] = idx
    head_size = tf.stack(head_size)

    mid_idx = [0]*r
    mid_idx[axis] = idx
    mid_idx = tf.stack(mid_idx)

    tail_size = [-1] * r

    head = tf.slice(tensor, start_idx, head_size)
    tail = tf.slice(tensor, mid_idx, tail_size)
    return tf.concat([tail, head], axis=axis)


def meshgrid2D(bs, height, width):
    with tf.compat.v1.variable_scope("meshgrid2D"):
        grid_x = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(a=tf.expand_dims(tf.linspace(0.0, width - 1, width), 1), perm=[1, 0])
        )

        grid_y = tf.matmul(tf.expand_dims(tf.linspace(0.0, height - 1, height), 1),
                           tf.ones(shape=tf.stack([1, width])))
        grid_x = tf.tile(tf.expand_dims(grid_x, 0), [bs, 1, 1], name="grid_x")
        grid_y = tf.tile(tf.expand_dims(grid_y, 0), [bs, 1, 1], name="grid_y")
        return grid_x, grid_y


def batch(tensor):
    return tf.expand_dims(tensor, axis=0)


def interleave(x1, x2, axis):
    x1s = tf.unstack(x1, axis=axis)
    x2s = tf.unstack(x2, axis=axis)
    outstack = []
    for (x1_, x2_) in zip(x1s, x2s):
        outstack.append(x1_)
        outstack.append(x2_)
    return tf.stack(outstack, axis=axis)


def bin_indices(Z, num_bins):
    raise Exception('percentile() not implemented in tf 2.0. Replace percentiles variable later')
    bin_size = 100.0 / num_bins

    percentiles = 0.5 
    # percentiles = tf.stack([tf.contrib.distributions.percentile(Z, bin_size * p)
    #                         for p in range(num_bins)])


    Z_ = tf.expand_dims(Z, 1)
    in_bin = tf.cast(Z_ > percentiles, tf.float32)
    in_bin *= tf.constant(list(range(1, num_bins + 1)), dtype=tf.float32)
    bin_idx = tf.cast(tf.argmax(input=in_bin, axis=1), tf.int32)
    return bin_idx


def prob_round(z):
    ''' probabilistically rounds z to floor(z) or ceil(z) '''

    zf = tf.floor(z)
    p = z - zf
    zf = tf.cast(zf, tf.int32)
    zc = zf + 1
    #if p ~= 0, then condition ~= 0 -> zf selected
    #if p ~= 1, then condition ~= 1 -> zc selected
    return tf.where(tf.random.uniform(tf.shape(input=p)) < p, zc, zf)


def select_p(tensor, p):
    ''' select entries of tensor with probability p'''
    d1 = tf.expand_dims(tf.shape(input=tensor)[0], axis=0)  # a weird approach
    keep = tf.random.uniform(d1) < p
    return tf.boolean_mask(tensor=tensor, mask=keep), keep


def extract_axis3(tensor, index):
    tensor_t = tf.transpose(a=mv_unstack(tensor), perm=(3, 0, 1, 2, 4))
    base = tf.gather(tensor_t, index)
    base = tf.squeeze(base, axis=1)  # 0 is batch axis
    return base


def rank(tensor):
    return len(tensor.get_shape())

def variable_in_shape(shape, name = 'variable'):
    return tf.compat.v1.get_variable(name, shape, use_resource=False)

def norm01(t):
    t -= tf.reduce_min(input_tensor=t)
    t /= tf.reduce_max(input_tensor=t)
    return t

def round2int(t):
    return tf.cast(tf.round(t), tf.int32)

def randidx(N, mask = None, seed = None):
    probs = tf.constant([1.0] * N, dtype = tf.float32)

    if mask is not None:
        probs += tf.cast(mask, tf.float32) * 100

    probs = tf.expand_dims(probs, 0)
    
    return tf.cast(tf.random.categorical(logits=probs, num_samples=1, seed = seed)[0, 0], tf.int32)

def match_placeholders_to_inputs(phs, inps):

    listortuple = lambda x: isinstance(x, list) or isinstance(x, tuple)
    
    if isinstance(phs, dict) and isinstance(inps, dict):
        rval = {}
        for name in phs:
            rval.update(match_placeholders_to_inputs(phs[name], inps[name]))
        return rval
    elif listortuple(phs) and listortuple(inps):
        rval = {}
        for (ph, inp) in zip(phs, inps):
            rval.update(match_placeholders_to_inputs(ph, inp))
        return rval
    elif 'tensorflow' in phs.__class__.__module__ and 'numpy' in inps.__class__.__module__:
        return {phs:inps}
    else:
        raise Exception('unsupported type...')

def pool3d(x, factor = 2, rank4 = False):
    if rank4:
        assert rank(x) == 4
        x = tf.expand_dims(x, axis = 4)
    return tf.nn.max_pool3d(
        x,
        ksize = [1, factor, factor, factor, 1],
        strides = [1, factor, factor, factor, 1],
        padding = 'VALID'
    )

def unitize(tensor, axis = -1):
    norm = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(tensor), axis = axis, keepdims = True) + const.eps)
    return tensor / norm

def set_batch_size(tensor, bs = const.BS):
    sizelist = list(tensor.shape)
    sizelist[0] = bs
    tensor.set_shape(sizelist)

def resize_by_axis(image, dim_1, dim_2, ax):
    resized_list = []
    unstack_img_depth_list = tf.unstack(image, axis = ax)
    for i in unstack_img_depth_list:
        resized_list.append(tf.image.resize(i, [dim_1, dim_2]))
    stack_img = tf.stack(resized_list, axis=ax)  
    return stack_img

def resize_voxel(vox,dims):
    vox = tf.squeeze(vox)
    dim_1,dim_2,dim_3 = dims
    resized_along_depth = resize_by_axis(vox,dim_1,dim_2,2) 
    resized_along_width = resize_by_axis(resized_along_depth,dim_1,dim_3,1)
    # resized_along_width = tf.expand_dims(resized_along_width,0)
    return resized_along_width

def make_opt_op(optimizer, non_trainable_weight_dict, fn, net):
    if const.eager:
        with tf.GradientTape() as tape:
            loss = fn()
            # if const.load_name:
            #     filtered_trainable_var = [var for var in net.trainable_variables if var.name not in non_trainable_weight_dict]
            # else:
            filtered_trainable_var = net.trainable_variables
            if const.frozen_decoder:
                filtered_trainable_var = [i for i in filtered_trainable_var if  not ("conv_lstm_dec" in i.name or "in_dec_2d" in i.name)]
            if const.mode != "test":
                gradients = tape.gradient(loss, filtered_trainable_var)
                clipped_gradients = [tf.clip_by_value(grad, -1., 1.) if grad is not None else grad for grad in gradients]
                return optimizer.apply_gradients(zip(clipped_gradients, filtered_trainable_var))
            else:
                return 
    else:
        if const.summ_grads:
            x = fn()
            grads = tf.gradients(ys=x, xs=tf.compat.v1.trainable_variables())
            grads = list(zip(grads, tf.compat.v1.trainable_variables()))
            for grad, var in grads:
                tf.compat.v1.summary.histogram(var.name + '/gradient', grad)
            return optimizer.minimize(x)
        else:
            if const.LOSS_GAN:
                dis_optimizer, gen_optimizer = optimizer
                d_loss,g_loss = fn()
                d_vars = [i for i in tf.compat.v1.trainable_variables() if "lsun/dcgan" in i.name]
                g_vars = [i for i in tf.compat.v1.trainable_variables() if "lsun/dcgan" not in i.name]
                dis = dis_optimizer.minimize(d_loss, var_list=d_vars)
                gen = gen_optimizer.minimize(g_loss, var_list=g_vars)
                return [dis,gen]
            else:
                return optimizer.minimize(fn())
    
def read_eager_tensor(x):
    if x is not None:
        return x.numpy()
    else:
        return None

def concat_apply_split(inputs, fn):
    num_split = len(inputs)

    inputs = tf.concat(inputs, axis = 0)
    outputs = fn(inputs)

    if isinstance(outputs, list):
        return [tf.split(output, num_split, axis = 0) for output in outputs]
    else:
        return tf.split(outputs, num_split, axis = 0)

def tanh01(x):
    return (tf.nn.tanh(x)+1)/2

def poolorunpool(input_, targetsize):
    inputsize = input_.shape.as_list()[1]
    if inputsize == targetsize:
        return input_
    elif inputsize > targetsize:
        ratio = inputsize // targetsize
        return tf.nn.pool(
            input=input_,
            window_shape = [ratio, ratio],
            padding = 'SAME',
            pooling_type = 'AVG',
            strides = [ratio, ratio]
        )
    else: #inputsize < targetsize:
        ratio = targetsize // inputsize
        return tf.image.resize(
            input_,
            tf.stack([inputsize * ratio]*2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
