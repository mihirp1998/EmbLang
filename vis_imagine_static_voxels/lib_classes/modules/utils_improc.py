import tensorflow as tf
# import hyperparams as hyp
from lib_classes.modules.utils_basic import *
# import cv2
import os
# import utils_misc

# import utils_rpn_tf

import matplotlib
import matplotlib.cm



EPS = 1e-6
MAXWIDTH = 1800 # in tensorboard

# B = hyp.B
# H = hyp.H
# W = hyp.W
# N = hyp.N

# S = hyp.S
# T = hyp.T

def summ_boxes(name, rgbRs, boxes3D, scores, tids, pix_T_cams):
    B, S, H, W, C = rgbRs.get_shape().as_list()
    boxes3D_vis = [draw_boxes3D_on_image(rgbRs[:,s],
                                         boxes3D[:,s],
                                         scores[:,s],
                                         tids[:,s],
                                         pix_T_cams[:,s])
               for s in range(S)]
    summ_rgbs(name, boxes3D_vis)

def summ_boxes_on_mem(name, rgbRs, boxes3D, scores, tids, pix_T_cams):
    B, S, H, W, C = rgbRs.get_shape().as_list()
    boxes3D_vis = [draw_boxes3D_on_image(rgbRs[:,s],
                                         boxes3D[:,s],
                                         scores[:,s],
                                         tids[:,s],
                                         pix_T_cams[:,s])
               for s in range(S)]
    summ_rgbs(name, boxes3D_vis)

def summ_box(name, rgbR, boxes3D, scores, tids, pix_T_cam):
    B, H, W, C = rgbR.get_shape().as_list()
    boxes3D_vis = draw_boxes3D_on_image(rgbR,
                                        boxes3D,
                                        scores,
                                        tids,
                                        pix_T_cam)
    summ_rgb(name, boxes3D_vis)

def preprocess_color(x):
    return tf.cast(x,tf.float32) * 1./255 - 0.5

# def preprocess_depth(x):
#     return tf.cast(x,tf.float32)

# def preprocess_valid(x):
#     return 1-tf.cast(x,tf.float32)

def back2color(i, blacken_zeros=False):
    if blacken_zeros:
        i = tf.where(tf.equal(i, 0.0), i-0.5, i)
        return back2color(i)
    else:
        return tf.cast((i+0.5)*255, tf.uint8)

def draw_boxes3D_on_image_py(rgb, corners_pix, scores, tids, boxes3D, thickness=1):
    # rgb is H x W x 3
    # corners_pix is N x 8 x 2
    # scores is N
    # tids is N
    # boxes3D is N x 9
    # pix_T_cam is 4 x 4

    # rgb = rgb.copy()
    rgb = rgb.numpy()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    H, W, C = rgb.shape
    assert(C==3)
    N, D, E = corners_pix.shape
    assert(D==8)
    assert(E==2)

    rx = boxes3D[:,6]
    ry = boxes3D[:,7]
    rz = boxes3D[:,8]

    color_map = matplotlib.cm.get_cmap('tab20')
    color_map = color_map.colors
    
    # draw
    for ind, corners in enumerate(corners_pix):
        # corners is 8 x 3
        if not np.isclose(scores[ind], 0.0):
            color_id = tids[ind] % 20
            color = color_map[color_id]
            color = np.array(color)*255.0
            # print 'tid = %d; score = %.3f' % (tids[ind], scores[ind])

            cv2.putText(rgb,
                        # '%d (%.2f)' % (tids[ind], scores[ind]), 
                        # '%d (%.2f)' % (tids[ind], ry[ind]), 
                        '%d (%.2f,%.2f,%.2f)' % (ind,
                                                 rx[ind],
                                                 ry[ind],
                                                 rz[ind]), 
                        (np.min(corners[:,0]), np.min(corners[:,1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, # font size
                        color,
                        1) # font weight

            xs = np.array([-1/2., -1/2., -1/2., -1/2., 1/2., 1/2., 1/2., 1/2.])
            ys = np.array([-1/2., -1/2., 1/2., 1/2., -1/2., -1/2., 1/2., 1/2.])
            zs = np.array([-1/2., 1/2., -1/2., 1/2., -1/2., 1/2., -1/2., 1/2.])
            xs = np.reshape(xs, [8, 1])
            ys = np.reshape(ys, [8, 1])
            zs = np.reshape(zs, [8, 1])
            offsets = np.concatenate([xs, ys, zs], axis=1)
            
            corner_inds = range(8)
            combos = list(combinations(corner_inds, 2))

            for combo in combos:
                pt1 = offsets[combo[0]]
                pt2 = offsets[combo[1]]
                # draw this if it is an in-plane edge
                eqs = pt1==pt2
                if np.sum(eqs)==2:
                    i, j = combo
                    pt1 = (corners[i, 0], corners[i, 1])
                    pt2 = (corners[j, 0], corners[j, 1])
                    retval, pt1, pt2 = cv2.clipLine((0, 0, W, H), pt1, pt2)
                    if retval:
                        cv2.line(rgb, pt1, pt2, color, thickness, cv2.LINE_AA)
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)

def draw_boxes3D_on_image(img, boxes3D, scores, tids, pix_T_cam):
    # first we need to get rid of invalid gt boxes
    # gt_boxes3D = trim_gt_boxes(gt_boxes3D)
    B, H, W, C = img.get_shape().as_list()
    assert(C==3)
    _, N, D = boxes3D.get_shape().as_list()
    assert(D==9)
    
    img = back2color(img)

    corners_cam = utils_geom.transform_boxes3D_to_corners(boxes3D)
    corners_cam_ = tf.reshape(corners_cam, [B, N*8, 3])
    corners_pix_ = utils_geom.apply_pix_T_cam(pix_T_cam, corners_cam_)
    corners_pix = tf.reshape(corners_pix_, [B, N, 8, 2])

    out = tf.py_function(func=draw_boxes3D_on_image_py, inp=[img[0], corners_pix[0], scores[0], tids[0], boxes3D[0]], Tout=tf.uint8)
    # out = tf.py_func(draw_boxes3D_on_image_py, [img[0], corners_pix[0], scores[0], tids[0]], tf.uint8)
    out = tf.expand_dims(out, axis=0)
    out = preprocess_color(out)
    out = tf.reshape(out, [1, H, W, C])
    return out

def draw_boxes3D_on_mem(img, boxes3D, scores, tids, mem_T_cam, already_mem=False):
    # first we need to get rid of invalid gt boxes
    # gt_boxes3D = trim_gt_boxes(gt_boxes3D)
    B, H, W, C = img.get_shape().as_list()
    assert(C==3)
    _, N, D = boxes3D.get_shape().as_list()
    assert(D==9)

    assert(not already_mem) # this mode does not work yet; see the cotrain branch for a working version of that
    
    img = back2color(img)

    if not already_mem:
        corners_cam = utils_geom.transform_boxes3D_to_corners(boxes3D)
        corners_cam_ = tf.reshape(corners_cam, [B, N*8, 3])
        corners_mem_ = utils_geom.apply_4x4(mem_T_cam, corners_cam_)
        # corners_mem_ = voxelizer.Ref2Mem(corners_cam_, H, W, D)
        corners_mem = tf.reshape(corners_mem_, [B, N, 8, 3])
        corners_mem_x = corners_mem[:,:,:,0]
        corners_mem_z = corners_mem[:,:,:,2]
        # we need zx order to match the vis tensor
        corners_pix = tf.stack([corners_mem_z, corners_mem_x], axis=3)
    else:
        corners_mem = utils_geom.transform_boxes3D_to_corners(boxes3D)
        corners_mem_x = corners_mem[:,:,:,0]
        corners_mem_z = corners_mem[:,:,:,2]
        # we need zx order to match the vis tensor
        corners_pix = tf.stack([corners_mem_z, corners_mem_x], axis=3)

    out = tf.py_function(func=draw_boxes3D_on_image_py, inp=[
        img[0], corners_pix[0], scores[0], tids[0], boxes3D[0]], Tout=tf.uint8)
    # out = tf.py_func(draw_boxes3D_on_image_py, [img[0], corners_pix[0], scores[0], tids[0]], tf.uint8)
    out = tf.expand_dims(out, axis=0)
    out = preprocess_color(out)
    out = tf.reshape(out, [1, H, W, C])
    return out

# def vis_target_box(rgb, target_box3D, target_score, target_tid,
#                    pix_T_cam, name='target_box'):
#     boxes3D = tf.expand_dims(target_box3D, axis=1)
#     scores = tf.expand_dims(target_score, axis=1)
#     tids = tf.expand_dims(target_tid, axis=1)
#     vis = draw_boxes3D_on_image(rgb, boxes3D, scores, tids, pix_T_cam)
#     target_box3D_on_rgb_ = utils_rpn_tf.draw_lidar_box3d_on_image(rgb,
#                                                                    boxes3D,
#                                                                    scores,
#                                                                    tids,
#                                                                    pix_T_rect,
#                                                                    rect_T_cam,
#                                                                    cam_T_velo)
#     summ_rgb(name, target_box3D_on_rgb_)


    

# def preprocess_for_resnet(im):
#     # takes an RGB uint8 image as input
#     # produces a mean-subbed BGR image as output
#     im_r, im_g, im_b = tf.split(axis=3, num_or_size_splits=3, value=im)
#     im = tf.cast(tf.concat(axis=3, values=[im_b, im_g, im_r]), dtype=tf.float32)
#     im_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
#     im_mean = tf.reshape(tf.constant(im_mean), [1, 1, 1, 3])
#     im -= im_mean
#     return im

def back2gray(i):
    r, g, b = tf.split(i, 3, axis=3)
    rgb = tf.tile((r+g+b)/3,[1,1,1,3])
    return back2color(rgb)

def oned2inferno(d, norm=True):
    if len(d.get_shape())==3:
        d = tf.expand_dims(d, axis=3)
    # convert a 1chan input to a 3chan image output
    if norm:
        d = normalize(d)
        rgb = colorize(d, cmap='inferno')
    else:
        rgb = colorize(d, vmin=0., vmax=1., cmap='inferno')
    rgb = tf.cast(255.0*rgb, tf.uint8)
    # rgb = tf.reshape(rgb, [-1, hyp.H, hyp.W, 3])
    # rgb = tf.expand_dims(rgb, axis=0)
    return rgb


# def reframe_masks(rgb, masks, scores, boxes, nums, vis=True):
#     # rgb is B x H x W x 3
#     # masks is B x N x 33 x 33
#     # boxes is B x N x 4
#     # returns all_masks, shaped [B, N, H, W, 1]

#     # maybe the mask dim is not actually 33
#     M = int(masks.get_shape()[2])
    
#     h = tf.cast(H*tf.ones([B], tf.float32), tf.int32)
#     w = tf.cast(W*tf.ones([B], tf.float32), tf.int32)
#     masks = tf.reshape(masks, [B, N, M, M])
#     scores = tf.reshape(scores, [B, N, 1, 1])
#     boxes = tf.reshape(boxes, [B, N, 4])
#     nums = tf.reshape(nums, [B])

#     masks = masks*scores
    
#     masks = tf.map_fn(ops_reframe, (masks, boxes, h, w, nums), tf.float32)
#     # masks is B x N x H x W

#     all_masks = tf.reshape(masks, [B, N, H, W, 1]) # return this

#     ## vis the masks in a single image
#     masks = tf.reshape(masks, [B, N, H, W, 1])
#     masks = tf.reduce_max(masks, axis=1)
    
#     masks = tf.reshape(masks, [B, H, W, 1])
#     # tf.summary.histogram('masks', masks)
#     # tf.summary.image('masks', oned2inferno(masks), max_outputs=1)

#     rgb = tf.reshape(rgb, [B, H, W, 3])
#     gray = back2gray(rgb)
#     red = oned2inferno(masks)
#     im = merge_images(gray, red)
#     if vis:
#         tf.summary.image('reframed_masks', im, max_outputs=1)
#     return all_masks
    
# def ops_reframe((masks, boxes, h, w, num)):
#     # masks is N x 33 x 33
#     # boxes is N x 4
    
#     # before calling this google function, we need to unpad
#     masks = masks[:num]
#     boxes = boxes[:num]
#     masks = ops.reframe_box_masks_to_image_masks(masks, boxes, h, w)
    
#     # and then pad again
#     masks = tf.pad(masks, [[0, N-num], [0, 0], [0, 0]])
#     masks = tf.reshape(masks, [N, H, W])
#     return masks

# def merge_images(im1, im2):
#     # im1 and im2 should already be uint8 in [0, 255]
#     return tf.cast((tf.cast(im1,tf.float32)+tf.cast(im2,tf.float32))/2, tf.uint8)

# def get_image_crops(image, boxes, size=64):
#     B = int(image.get_shape()[0])
#     N = int(boxes.get_shape()[1])
#     # image is B x H x W x C
#     # boxes is B x N x 4

#     boxes = tf.reshape(boxes, [B*N, 4])

#     ## careful reshape!
#     box_ind = tf.tile(tf.reshape(tf.range(0, B), [B, 1]), [1, N])
#     box_ind = tf.cast(tf.reshape(box_ind, [-1]), tf.int32)
    
#     size = tf.constant([size, size], tf.int32)
#     image_crops = tf.image.crop_and_resize(image, boxes, box_ind, size,
#                                            extrapolation_value=0.0)
#     # image_crops is B*N x D x D x C
   
#     return image_crops

# def get_uniform_image_crops(image, boxes, size=64, box_dim0=0.5, box_dim1=0.5):
#     B = int(image.get_shape()[0])
#     H = int(image.get_shape()[1])
#     W = int(image.get_shape()[2])
#     N = int(boxes.get_shape()[1])
#     # image is B x H x W x C
#     # boxes is B x N x 4
    
#     # like the other one, but we first make all the boxes the same size
#     centroids = utils_misc.get_centroids(boxes)

#     ratio = (1.0*H)/W
#     dim0 = box_dim0*tf.ones([B, N])
#     dim1 = ratio*box_dim1*tf.ones([B, N])
#     dims = tf.stack([dim0, dim1], axis=2)
#     boxes_ = utils_misc.get_boxes(centroids, dims, clip=False)
#     return get_image_crops(image, boxes_, size=size)

# def get_square_uniform_image_crops(image, boxes, size=64):
#     B = int(image.get_shape()[0])
#     H = int(image.get_shape()[1])
#     W = int(image.get_shape()[2])
#     N = int(boxes.get_shape()[1])
#     # image is B x H x W x C
#     # boxes is B x N x 4
    
#     # extracts boxes of a fixed size, with no rescaling
    
#     centroids = utils_misc.get_centroids(boxes)

#     # tf needs the boxes in [0, 1]
#     # i want the boxes to be exactly size x size in pixels

#     box_h = float(size)/float(H)
#     box_w = float(size)/float(W)

#     dim0 = box_h*tf.ones([B, N])
#     dim1 = box_w*tf.ones([B, N])
#     dims = tf.stack([dim0, dim1], axis=2)

#     # this seems not quite exact, but i think crop_and_resize is differentiable
#     # (since it does a bit of bilinear interp for sub-pix boxes)
#     # whereas taking an "exact" crop would not be so
    
#     boxes_ = utils_misc.get_boxes(centroids, dims, clip=False)
#     return get_image_crops(image, boxes_, size=size)

# def atan2_ocv(y, x):
#     with tf.variable_scope("atan2_ocv"):
#         # constants
#         DBL_EPSILON = 2.2204460492503131e-16
#         atan2_p1 = 0.9997878412794807 * (180 / np.pi)
#         atan2_p3 = -0.3258083974640975 * (180 / np.pi)
#         atan2_p5 = 0.1555786518463281 * (180 / np.pi)
#         atan2_p7 = -0.04432655554792128 * (180 / np.pi)
#         ax, ay = tf.abs(x), tf.abs(y)
#         c = tf.where(tf.greater_equal(ax, ay), tf.div(ay, ax + DBL_EPSILON),
#                       tf.div(ax, ay + DBL_EPSILON))
#         c2 = tf.square(c)
#         angle = (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c
#         angle = tf.where(tf.greater_equal(ax, ay), angle, 90.0 - angle)
#         angle = tf.where(tf.less(x, 0.0), 180.0 - angle, angle)
#         angle = tf.where(tf.less(y, 0.0), 360.0 - angle, angle)
#         return angle
                                                                                            
# def cart_to_polar_ocv(x, y, angle_in_degrees=False):
#     with tf.variable_scope("cart_to_polar_ocv"):
#         v = tf.sqrt(tf.add(tf.square(x), tf.square(y)))
#         ang = atan2_ocv(y, x)
#         scale = 1 if angle_in_degrees else np.pi / 180
#         return v, tf.multiply(ang, scale)
    
def depth2inferno(depth, valid=None, already_inv=False, already_log=False, maxdepth=60.0):
    # depth should be [B x H x W x 1]
    max_depth = np.log(maxdepth)
    if already_inv:
        depth = 1./depth
    if not already_log:
        depth = tf.log(depth)
    depth = tf.clip_by_value(depth, 0, max_depth)

    if valid is not None:
        depth = tf.where(tf.greater(valid, 0.0),
                         depth, max_depth*tf.ones_like(depth))
        
    depth = depth/max_depth
    depth_im = oned2inferno(depth, norm=False)
    # depth = tf.cast((1-depth)*255,tf.uint8)
    # depth_im = tf.tile(depth, [1,1,1,3])


    # max_depth = 80.0
    # if already_inv:
    #     depth = 1./depth
    # depth = tf.clip_by_value(depth, 0, max_depth)
    # depth = depth/max_depth
    # depth_im = oned2inferno(1.0-depth, norm=False)
    return depth_im

# # def depth2invinferno(depth, already_inv=False):
# #     # depth should be [B x H x W x 1]
# #     # max_depth = 5.0 # e^5 ~= 150 m
# #     max_depth = 4.38 # ln(80)
# #     if already_inv:
# #         depth = 1./depth
# #     depth = tf.log(depth)
# #     depth = tf.clip_by_value(depth, 0, max_depth)
# #     depth = depth/max_depth
# #     depth_im = oned2inferno(1.0-depth, norm=False)
# #     # depth = tf.cast((1-depth)*255,tf.uint8)
# #     # depth_im = tf.tile(depth, [1,1,1,3])
# #     return depth_im

def oned2gray(d, norm=True):
    # convert a 1chan input to a 3chan image output
    # (really this is gray)
    if norm:
        d = normalize(d)
    return tf.cast(tf.tile(255*d,[1,1,1,3]),tf.uint8)

def overlay_1on2(im1, im2):
    # reduce the intensity of im2
    im2 = tf.cast(tf.cast(im2, tf.float32)*0.5, tf.uint8)

    # where im1 has nothing, put im2
    im1_ = tf.tile(tf.reduce_max(im1, axis=3, keepdims=True), [1, 1, 1, 3])
    all_z = tf.equal(im1_, tf.zeros_like(im1_))
    im1 = tf.where(all_z, im2, im1)

    return im1

def seq2color_on_oned(d, oned, norm=True, colormap='RdBu'):
    d_im = seq2color(d, norm=norm, colormap=colormap)
    o_im = oned2gray(oned, norm=norm)
    d_im = overlay_1on2(d_im, o_im)
    return d_im

def seq2color(d, norm=True, colormap='RdBu'):
    B = int(d.get_shape()[0])
    H = int(d.get_shape()[1])
    W = int(d.get_shape()[2])
    C = int(d.get_shape()[3])
    # the C dim is sequenced
    # we want to convert this to a nicely colorized image output

    # prep a mask of the valid pixels, so we can blacken the invalids later
    mask = tf.reduce_max(d, axis=3, keepdims=True)

    # turn the C dim into an explicit sequence
    coeffs = np.linspace(1.0, float(C), C).astype(np.float32)
    # increase the spacing from the center
    coeffs[:C/2] -= 2.0
    coeffs[C/2+1:] += 2.0
    coeffs = tf.constant(coeffs, tf.float32)
    coeffs = tf.reshape(coeffs, [1, 1, 1, C])
    coeffs = tf.tile(coeffs, [B, H, W, 1])
    # scale each channel by the right coeff
    d = d * coeffs
    # now d is in [1, C], except for the invalid parts which are 0
    # keep the highest valid coeff at each pixel
    d = tf.reduce_max(d, axis=3, keepdims=True)
    # d = tf.where(tf.equal(d, 0.0), tf.ones_like(d), d)

    # note the range here is -2 to C+2, since we shifted away from the center
    rgb = colorize(d, vmin=0.0-2.0, vmax=float(C+2.0), vals=255, cmap=colormap)

    # blacken the invalid pixels, instead of using the 0-color
    rgb = rgb*mask
    
    rgb = tf.cast(255.0*rgb, tf.uint8)
    return rgb

def oned2red(d, norm=True):
    # convert a 1chan input to a 3chan image output
    if norm:
        d = normalize(d)
    red = tf.cast(255*d,tf.uint8)
    # put ones in all chans of the colored pixels
    ones = red/128
    rgb = tf.concat([red, ones, ones], axis=3)
    return rgb

# def depth2color(depth, already_inv=False):
#     # depth should be [B x H x W x 1]
#     # maxDepth = 5.0 # e^5 ~= 150 m
#     maxDepth = 4.38 # ln(80)
#     if already_inv:
#         depth = 1./depth
#     depth = tf.log(depth)
#     depth = tf.clip_by_value(depth, 0, maxDepth)
#     depth = depth/maxDepth
#     depth = tf.cast((1-depth)*255,tf.uint8)
#     depth_im = tf.tile(depth, [1,1,1,3])
#     return depth_im

# def texturedness(rgb, scale=1.0):
#     # rgb should be B x H x W x 3
#     # shows where the RGB is not smooth
#     B = int(rgb.get_shape()[0])
#     H = int(rgb.get_shape()[1])
#     W = int(rgb.get_shape()[2])
#     if not scale==1.0:
#         h = tf.cast(H*scale, tf.int32)
#         w = tf.cast(W*scale, tf.int32)
#         rgb = tf.image.resize_images(rgb, [h, w])
#     R_dx, R_dy = gradient2D(rgb)
#     # R_dx = tf.exp(-tf.reduce_sum(tf.abs(R_dx), axis=3, keepdims=True))
#     # R_dy = tf.exp(-tf.reduce_sum(tf.abs(R_dy), axis=3, keepdims=True))
#     # t = R_dx+R_dy
    
#     tex = tf.exp(-tf.reduce_sum(tf.abs(R_dx) + tf.abs(R_dy), axis=3, keepdims=True))
#     if not scale==1.0:
#         tex = tf.image.resize_images(tex, [H, W])
#     return tex


def flow2color(flow, clip=50.0):
    """
    :param flow: Optical flow tensor.
    :return: RGB image normalized between 0 and 1.
    """
    with tf.name_scope('flow_visualization'):
        # B, H, W, C dimensions.
        abs_image = tf.abs(flow)
        flow_mean, flow_var = tf.nn.moments(abs_image, axes=[1, 2, 3])
        flow_std = tf.sqrt(flow_var)

        if clip:
            mf = clip
            flow = tf.clip_by_value(flow, -mf, mf)/mf
        else:
            # Apply some kind of normalization. Divide by the perceived maximum (mean + std)
            flow = flow / tf.expand_dims(tf.expand_dims(
                tf.expand_dims(flow_mean + flow_std + 1e-10, axis=-1), axis=-1), axis=-1)

        radius = tf.sqrt(tf.reduce_sum(tf.square(flow), axis=-1))
        radius_clipped = tf.clip_by_value(radius, 0.0, 1.0)
        angle = tf.atan2(-flow[..., 1], -flow[..., 0]) / np.pi

        hue = tf.clip_by_value((angle + 1.0) / 2.0, 0.0, 1.0)
        saturation = tf.ones(shape=tf.shape(hue), dtype=tf.float32) * 0.75
        value = radius_clipped
        hsv = tf.stack([hue, saturation, value], axis=-1)
        flow = tf.image.hsv_to_rgb(hsv)
        flow = tf.cast(flow*255.0, tf.uint8)
        return flow

# def flow2color(flow, clip=True, half_intensity=False):
#     with tf.variable_scope("flow2color"):
#         shape = flow.get_shape()
#         B, H, W, C = shape
#         if clip:
#             maxFlow = 20.0
#             flow = tf.clip_by_value(flow, -maxFlow, maxFlow)
#         else:
#             maxFlow = tf.reduce_max(flow)/2.0
#         if half_intensity:
#             maxFlow = maxFlow*2
#         # add some temp values to reach the maxes
#         flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[maxFlow*tf.ones([B,H,1,1]),-maxFlow*tf.ones([B,H,1,1])]),flow])
#         flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[-maxFlow*tf.ones([B,H,1,1]),maxFlow*tf.ones([B,H,1,1])]),flow])
#         flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[maxFlow*tf.ones([B,H,1,1]),tf.zeros([B,H,1,1])]),flow])
#         flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[-maxFlow*tf.ones([B,H,1,1]),tf.zeros([B,H,1,1])]),flow])
#         flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([B,H,1,1]),maxFlow*tf.ones([B,H,1,1])]),flow])
#         flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([B,H,1,1]),-maxFlow*tf.ones([B,H,1,1])]),flow])
#         flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([B,H,1,1]),tf.zeros([B,H,1,1])]),flow])
#         flow = tf.concat(axis=2,values=[tf.zeros([B,H,1,2]),flow])
#         flow = tf.concat(axis=2,values=[maxFlow*tf.ones([B,H,1,2]),flow])
#         flow = tf.concat(axis=2,values=[-maxFlow*tf.ones([B,H,1,2]),flow])
#         fx, fy = flow[:, :, :, 0], flow[:, :, :, 1]
#         fx = tf.clip_by_value(fx, -maxFlow, maxFlow)
#         fy = tf.clip_by_value(fy, -maxFlow, maxFlow)
#         v, ang = cart_to_polar_ocv(fx, fy)
#         h = normalize(tf.multiply(ang, 180 / np.pi))
#         s = tf.ones_like(h)
#         v = normalize(v)
#         hsv = tf.stack([h, s, v], 3)
#         rgb = tf.image.hsv_to_rgb(hsv) * 255
#         rgb = tf.slice(rgb,[0,0,10,0],[-1,-1,-1,-1])
#         # rgb = rgb[0,0,1:,:]
#         return tf.cast(rgb, tf.uint8)


# def resize_object_masks(masks, size):
#     B = int(masks.get_shape()[0])
#     N = int(masks.get_shape()[1])
#     H_ = int(masks.get_shape()[2])
#     W_ = int(masks.get_shape()[3])
#     masks = tf.transpose(tf.reshape(masks, [B, N, H_, W_]), perm=[0, 2, 3, 1])
#     H, W = size
#     masks = tf.image.resize_images(masks, [H, W])
#     masks = tf.transpose(masks, perm=[0, 3, 1, 2])
#     masks = tf.reshape(masks, [B, N, H, W, 1])
#     return masks
    
# def summarize_bboxes(rgbs, boxes, name, ind=0):
#     boxes = tf.clip_by_value(boxes, 0.0, 1.0)
#     boxes_on_im = tf.image.draw_bounding_boxes(tf.cast(back2color(rgbs), tf.float32)/255., boxes)
#     tf.summary.image('bboxes_%s' % name, tf.expand_dims(boxes_on_im[ind], axis=0), max_outputs=1)

def xy2mask(xy, proto, norm=False):
    # xy is B x N x 2, in either pixel coords or normalized coordinates (depending on norm)
    # proto is B x H x W x 1, showing how big to make the mask
    # returns a mask the same size as proto, with a 1 at each specified xy
    if norm:
        # convert to pixel coords
        x, y = tf.unstack(xy, axis=2)
        shape = proto.get_shape()
        H = int(shape[0])
        W = int(shape[1])
        x = x*float(W)
        y = y*float(H)
        xy = tf.stack(xy, axis=2)
    mask = tf.map_fn(xy2mask_single, (xy, proto), dtype=tf.float32)
    return mask

# def xyv2mask(xy, v, proto, norm=False):
#     # xy is B x N x 2, in either pixel coords or normalized coordinates (depending on norm)
#     # v is B x N; it tells what value in [0,1] to fill with at the corresponding coord
#     # proto is B x H x W x 1, showing how big to make the mask
#     # returns a mask the same size as proto, with a 1 at each specified xy
#     if norm:
#         # convert to pixel coords
#         x, y = tf.unstack(xy, axis=2)
#         shape = proto.get_shape()
#         H = int(shape[0])
#         W = int(shape[1])
#         x = x*float(W)
#         y = y*float(H)
#         xy = tf.stack(xy, axis=2)
#     mask = tf.map_fn(xyv2mask_single, (xy, v, proto), dtype=tf.float32)
#     return mask

def xy2mask_single(val):
    xy, proto = val
    # xy is N x 2
    # proto is H x W x 1
    x, y = tf.unstack(xy, axis=1)
    x = tf.cast(x, tf.int32)
    y = tf.cast(y, tf.int32)

    shape = proto.get_shape()
    H = int(shape[0])
    W = int(shape[1])
    x = tf.clip_by_value(x, 0, W-1)
    y = tf.clip_by_value(y, 0, H-1)
    
    inds = sub2ind(H, W, y, x)

    ok = tf.squeeze(tf.where(tf.logical_and(tf.less(inds, H*W), tf.greater_equal(inds, 0))))
    inds = tf.squeeze(tf.gather(inds, ok))
    
    mask = tf.sparse_to_dense(inds, [H*W], sparse_values=1.0, default_value=0.0, validate_indices=False)
    mask = tf.reshape(mask,[H,W,1])
    return mask

# def xyv2mask_single((xy, vals, proto)):
#     # xy is N x 2
#     # vals is N
#     # proto is H x W x 1
#     x, y = tf.unstack(xy, axis=1)
#     x = tf.cast(x, tf.int32)
#     y = tf.cast(y, tf.int32)

#     shape = proto.get_shape()
#     H = int(shape[0])
#     W = int(shape[1])
#     x = tf.clip_by_value(x, 0, W-1)
#     y = tf.clip_by_value(y, 0, H-1)
    
#     inds = sub2ind(H, W, y, x)

#     ok = tf.squeeze(tf.where(tf.logical_and(tf.less(inds, H*W), tf.greater_equal(inds, 0))))
#     inds = tf.squeeze(tf.gather(inds, ok))
#     vals = tf.squeeze(tf.gather(vals, ok))
    
#     mask = tf.sparse_to_dense(inds, [H*W], sparse_values=vals, default_value=0.0, validate_indices=False)
#     mask = tf.reshape(mask,[H,W,1])
#     return mask

# def draw_line_py(x1, y1, x2, y2, canvas):
#     # coords are shaped [1]
#     # canvas is shaped H x W (x 1)
#     x1 = x1.astype(np.int32)
#     y1 = y1.astype(np.int32)
#     x2 = x2.astype(np.int32)
#     y2 = y2.astype(np.int32)
    
#     # x2, y2 = p2.astype(np.int32)
#     rr, cc, v = line_aa(y1, x1, y2, x2)
#     # rr, cc, v = line_aa(y1[0], x1[0], y2[0], x2[0])
#     canvas[rr, cc] = 1.0
#     return canvas

# def draw_lines(xy1, xy2, scores, H, W):
#     # draw lines from each xy1 to the corresp xy2
#     # do it in an image sized like proto

#     # xy1 is B x N x 2
#     # xy2 is B x N x 2
#     # scores is B x N 

#     B = int(xy1.get_shape()[0])
#     N = int(xy1.get_shape()[1])
#     canvas = tf.zeros([B*N, H, W, 1])

#     xy1 = tf.reshape(xy1, [B*N, 2])
#     xy2 = tf.reshape(xy2, [B*N, 2])
#     scores = tf.reshape(scores, [B*N])
    
#     x1, y1 = tf.unstack(xy1, axis=1)
#     x2, y2 = tf.unstack(xy2, axis=1)
#     x1 = tf.clip_by_value(x1, 0, W-1)
#     x2 = tf.clip_by_value(x2, 0, W-1)
#     y1 = tf.clip_by_value(y1, 0, H-1)
#     y2 = tf.clip_by_value(y2, 0, H-1)

#     canvas = tf.map_fn(draw_lines_single,
#                        (x1, y1, x2, y2, canvas), tf.float32)
#     canvas = tf.reshape(canvas, [B, N, H, W, 1])
#     scores = tf.reshape(scores, [B, N, 1, 1, 1])
#     canvas *= scores
#     canvas = tf.reduce_max(canvas, axis=1)

#     # canvas is B x H x W x 1
#     return canvas
    
# def draw_lines_single((x1, y1, x2, y2, canvas)):
#     # xy1 is 2
#     # xy2 is 2
#     # canvas is H x W x 1

#     x1 = tf.cast(x1, tf.int32)
#     x2 = tf.cast(x2, tf.int32)
#     y1 = tf.cast(y1, tf.int32)
#     y2 = tf.cast(y2, tf.int32)
    
#     canvas = tf.py_func(draw_line_py, [x1, y1, x2, y2, canvas], tf.float32)
#     return canvas
    
def vis_corners_on_im(pix_T_cam, corners_cam, rgb):
    # corners_cam is B x N x 8 x 3
    
    # visualize box3D corners on an rgb image
    # corners are provided in velodyne coords
    
    B = int(corners_cam.get_shape()[0])
    N = int(corners_cam.get_shape()[1])
    H = int(rgb.get_shape()[1])
    W = int(rgb.get_shape()[2])
    
    corners_cam = tf.reshape(corners_cam, [B, N*8, 3])
    corners_pix = utils_geom.apply_pix_T_cam(pix_T_cam, corners_cam)
    # corners_pix = tf.reshape(corners_pix, [B, N, 8, 3])

    x, y = tf.unstack(corners_pix, axis=2)
    # vis these into a quarter-size image, then upsample
    x = x/W
    y = y/H
    H_ = 0.5*H
    W_ = 0.5*W
    x = x*W_
    y = y*H_
    xy = tf.stack([x, y], axis=2)
    mask = xy2mask(xy, tf.zeros([B, int(H_), int(W_)]))
    mask = tf.image.resize_images(mask, [H, W])

    # mask3 = tf.tile(mask, [1, 1, 1, 3])
    # rgb = rgb*mask
    # mask_im = oned2red(mask)
    # # rgb_im = back2gray(rgb)
    # rgb_im = back2color(rgb, blacken_zeros=True)
    # merge = merge_images(mask_im, rgb_im)
    
    merge = multicolor_heat_on_im(mask, rgb)
    # merge = heat_on_im(mask, rgb)
    return merge

# def multicolor_heat(lm):
#     # lm is B x H x W x L
#     B = int(lm.get_shape()[0])
#     H = int(lm.get_shape()[1])
#     W = int(lm.get_shape()[2])
#     L = int(lm.get_shape()[3])
    
#     lm = normalize(lm)
#     lm = tf.where(lm > 0.5, tf.ones_like(lm), tf.zeros_like(lm))

#     lm_flat = tf.reduce_max(lm, axis=3, keepdims=True)
#     gray = oned2gray(lm_flat)
#     # gray is B x H x W x 3
    

#     # vmin = tf.reduce_min(lm)
#     # vmax = tf.reduce_max(lm)
#     # lm = (lm - vmin) / (vmax - vmin) # vmin..vmax

#     # # squeeze last dim if it exists
#     # lm = tf.squeeze(lm)

#     m = tf.linspace(1.0, L, L)
#     m = tf.reshape(m, [1, 1, 1, L])
#     m = tf.tile(m, [B, H, W, 1])
#     lm *= m

#     lm = tf.reduce_max(lm, axis=3)
    
#     lm = tf.map_fn(multicolor_heat_single, lm, dtype=tf.float32)
#     gray = tf.cast(gray, tf.float32)/255.0
#     lm = tf.cast(255.0*lm*gray, tf.uint8)
#     # lm = tf.cast(255.0*lm, tf.uint8)
    
#     # # quantize
#     # indices = tf.to_int32(tf.round(lm * 255))

#     # # gather
#     # cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
#     # colors = tf.constant(cm.colors, dtype=tf.float32)
    
#     # lm = tf.gather(colors, indices)
#     # lm = tf.cast(lm, tf.uint8)
    
#     # lm = tf.map_fn(multicolor_heat_single, (lm, gray), dtype=tf.uint8)
#     # # lm is B x H x W x 3
#     return lm

def multicolor_heat_on_im(lm, im):
    # lm is B x H x W x L
    # im is B x H x W x 3

    B = int(lm.get_shape()[0])
    H = int(lm.get_shape()[1])
    W = int(lm.get_shape()[2])
    L = int(lm.get_shape()[3])
    
    lm = normalize(lm)
    lm = tf.where(lm > 0.8, tf.ones_like(lm), tf.zeros_like(lm))

    lm_flat = tf.reduce_max(lm, axis=3, keepdims=True)
    gray = tf.cast(oned2gray(lm_flat), tf.float32)/255.0
    im = tf.cast(back2color(im), tf.float32)/255.0
    im = im*(1.0-gray)
    
    m = tf.linspace(1.0, L, L)
    m = tf.reshape(m, [1, 1, 1, L])
    m = tf.tile(m, [B, H, W, 1])
    lm *= m
    lm = tf.reduce_max(lm, axis=3, keepdims=True)

    # lm = tf.map_fn(multicolor_heat_single, lm, dtype=tf.float32)
    lm = colorize(lm, vmin=0.0, vmax=L, cmap='tab20', vals=19)
    lm = lm*gray
    im = 0.5*im+lm
    im = tf.cast(255.0*im, tf.uint8)
    im = preprocess_color(im)
    return im

# # def multicolor_heat_single(lm):
# #     # lm is H x W
# #     out = colorize(lm, vmin=0.0, vmax=20, cmap='tab20', vals=19)
# #     out = tf.reshape(out, [H, W, 3])
# #     return out
# #     # return tf.cast(colorize(lm, cmap='viridis'), tf.float32)

def heat_on_im(heat, im):
    h = oned2red(heat)
    i = back2gray(im)
    h = tf.cast(h, tf.float32)
    i = tf.cast(i, tf.float32)
    im = 0.5*h+0.5*i
    im = tf.cast(im, tf.uint8)
    im = preprocess_color(im)
    return im

# def heat_on_oned(heat, im):
#     h = oned2red(heat)
#     i = oned2gray(im)
#     h = tf.cast(h, tf.float32)
#     i = tf.cast(i, tf.float32)
#     im = 0.5*h+0.5*i
#     im = tf.cast(im, tf.uint8)
#     return im

def prep_birdview_vis(image):
    # # vox is B x Y x X x Z x C

    # # discard the vertical dim
    # image = tf.reduce_mean(vox, axis=1)

    # right now, X will be displayed vertically, which is confusing... 

    # make "forward" point up, and make "right" point right
    image = tf.map_fn(tf.image.rot90, image)

    return image

# def summarize_birdview_images(image, name='bird_view_image', max_outputs=1):
#     # image is uint8 and already in bird view, in the velodyne coordinate system

#     # print 'TEMPORARILY DISABLED BIRDVIEW VIS PREP'
    
#     image = prep_birdview_vis(image)
    
#     # to make the vis more readable,
#     # make "forward" point up, and make "right" point right
#     tf.summary.image(name, image, max_outputs=max_outputs)


# def vis_corners_on_bird(boxes3D_corners, rgb):
#     # boxes3D_corners is B x N x 8 x 3
#     # visualize box3D corners on an rgb image
#     # corners are provided in velodyne coords
    
#     B = int(boxes3D_corners.get_shape()[0])
#     N = int(boxes3D_corners.get_shape()[1])
#     H = int(rgb.get_shape()[1])
#     W = int(rgb.get_shape()[2])
    
#     corners_velo = tf.reshape(boxes3D_corners, [B, N*8, 3])
#     corners_bird = utils_geom.Velo2Bird(corners_velo)
    
#     # ones = tf.ones([B, N*8, 1])
#     # corners_velo = tf.concat([corners_velo, ones], axis=2)
#     # corners_velo = tf.transpose(corners_velo, perm=[0,2,1])
#     # corners_cam = tf.matmul(cam_T_velo, corners_velo)
#     # corners_rect = tf.matmul(rect_T_cam, corners_cam)
#     # corners_pix = tf.matmul(pix_T_rect, corners_rect)
#     # corners_pix = tf.transpose(corners_pix, perm=[0,2,1])
#     # corners_pix = corners_pix[:,:,:2]/(EPS+tf.expand_dims(corners_pix[:,:,2], axis=2))

#     x, y = tf.unstack(corners_bird, axis=2)
#     # normalize
#     x = x/W
#     y = y/H

#     # these are B x N*8
#     x_ = tf.reshape(x, [B, N, 8])
#     y_ = tf.reshape(y, [B, N, 8])
#     xmin = tf.reduce_min(x_, axis=2)
#     xmax = tf.reduce_max(x_, axis=2)
#     ymin = tf.reduce_min(y_, axis=2)
#     ymax = tf.reduce_max(y_, axis=2)
#     # these are B x N
    
#     x = tf.concat([xmin, xmax, xmin, xmax], axis=1)
#     y = tf.concat([ymin, ymin, ymax, ymax], axis=1)

#     boxes = tf.stack([ymin, xmin, ymax, xmax], axis=2)
    
#     boxes = tf.clip_by_value(boxes, 0.0, 1.0)
#     im = tf.image.draw_bounding_boxes(
#         tf.cast(back2color(rgb), tf.float32)/255., boxes)
#     im = preprocess_color(im * 255.)
    
#     # put them into a bigger image
#     H_ = 0.25*H
#     W_ = 0.25*W
#     x = x*W_
#     y = y*H_
#     xy = tf.stack([x, y], axis=2)
#     mask = xy2mask(xy, tf.zeros([B, int(H_), int(W_)]))
#     mask = tf.image.resize_images(mask, [H, W])

#     # mask3 = tf.tile(mask, [1, 1, 1, 3])
#     # rgb = rgb*mask
#     # mask_im = oned2red(mask)
#     # # rgb_im = back2gray(rgb)
#     # rgb_im = back2color(rgb, blacken_zeros=True)
#     # merge = merge_images(mask_im, rgb_im)
    
#     # merge = multicolor_heat_on_im(mask, rgb)
#     # merge = heat_on_im(mask, rgb)
#     merge = heat_on_im(mask, im)
#     return merge

    
def xy2heatmap(xyPos, sigma, all_x, all_y, norm=False):
    # xyPos is B x N x 2, containing tf.float32 x and y coordinates of N things (in pixel coords)
    # all_x and all_y are B x H x W x N
    
    # print 'xy2heatmap'

    B = int(all_x.get_shape()[0])
    H = int(all_x.get_shape()[1])
    W = int(all_x.get_shape()[2])
    N = int(all_x.get_shape()[3])
    
    mu_x = xyPos[:,:,0]
    mu_y = xyPos[:,:,1]

    # mu_x = tf.Print(mu_x, [mu_x], 'mu_x', summarize=24)
    # mu_y = tf.Print(mu_y, [mu_y], 'mu_y', summarize=24)

    # -1.0 in xyPos is reserved for "unknown". set the gaussian to all zeros here
    mu_x = tf.where(tf.equal(mu_x, -1.0), -100000.0*tf.ones_like(mu_x), mu_x)
    mu_y = tf.where(tf.equal(mu_y, -1.0), -100000.0*tf.ones_like(mu_y), mu_y)
    
    mu_x = tf.reshape(mu_x, [B, 1, 1, N])
    mu_y = tf.reshape(mu_y, [B, 1, 1, N])
    mu_x = tf.tile(mu_x, [1, H, W, 1])
    mu_y = tf.tile(mu_y, [1, H, W, 1])
    
    sigma_sq = sigma*sigma;
    sq_diff_x = tf.square(all_x - mu_x)
    sq_diff_y = tf.square(all_y - mu_y)

    term1 = 1./2.*np.pi*sigma_sq
    term2 = tf.exp(-(sq_diff_x+sq_diff_y)/(2.*sigma_sq))
    gauss = term1*term2

    if norm:
        # normalize so each gaussian peaks at 1
        gsum = tf.reduce_max(gauss, axis=[1, 2], keepdims=True)
        gauss = gauss/(EPS+gsum)
    return gauss

def xy2heatmaps(xy, B, H, W, sigma=30.0):
    # xy is B x N x 2

    B = int(xy.get_shape()[0])
    N = int(xy.get_shape()[1])
    
    grid_y, grid_x = meshgrid2D(B, H, W)
    # grid_x and grid_y are B x H x W
    grid_x = tf.tile(tf.expand_dims(grid_x, -1), [1, 1, 1, N])
    grid_y = tf.tile(tf.expand_dims(grid_y, -1), [1, 1, 1, N])
    # grid_x and grid_y are B x H x W x N
    heat = xy2heatmap(xy, sigma, grid_x, grid_y, norm=True)
    # heat is B x H x W x N     
    # heat = tf.reshape(heat, [B, H, W, N])
    return heat

# def put_heatmaps_on_batch_dim(lm):
#     # lm is B x H x W x L
#     # returns L x H x W x 3 visualization
#     lm0 = lm[0,:,:,:]
#     lm0 = tf.transpose(lm0, [2, 0, 1])
#     lm0 = tf.expand_dims(lm0, axis=3)
#     lm0 = oned2inferno(lm0)
#     return lm0

# def put_channels_on_batch_dim(lm):
#     # lm is B x H x W x L
#     # returns L x H x W x 1 
#     lm0 = lm[0,:,:,:]
#     # lm0 is H x W x L
#     lm0 = tf.transpose(lm0, perm=[2, 0, 1])
#     # lm0 is L x H x W
#     lm0 = tf.expand_dims(lm0, axis=3)
#     # lm0 is L x H x W x 1
#     return lm0
    
# def summarize_valid_boxes(boxes, scores, rgb, bird=True, name='boxes'):
#     # boxes is B x N x 4
#     # scores is B x N
#     # rgb is B x H x W x 3 (where H=BY and W=BX)
#     B = int(rgb.get_shape()[0])
#     H = int(rgb.get_shape()[1])
#     W = int(rgb.get_shape()[2])
#     N = int(boxes.get_shape()[1])
    
#     # we'll do this just for the first el of the batch
#     scores = scores[0] # N
#     boxes = boxes[0] # N x 4
#     rgb = rgb[0] # H x W x 3
    
#     inds = tf.where(tf.greater(scores, 0.0))
#     boxes = tf.gather(boxes, inds)
#     boxes = tf.reshape(boxes, [1, -1, 4])
#     rgb = tf.reshape(rgb, [1, H, W, 3])
    
#     boxes = tf.clip_by_value(boxes, 0.0, 1.0)
#     im = tf.image.draw_bounding_boxes(
#         tf.cast(back2color(rgb, blacken_zeros=True), tf.float32)/255., boxes)
    
#     if bird:
#         summarize_birdview_images(im, name)
#     else:
#         tf.summary.image(name, im)
        
# def rotate_images(image, degrees):
#     image = tf.contrib.image.rotate(
#         image,
#         np.pi/180.0*degrees,
#         interpolation='BILINEAR',
#     )
#     return image

# def draw_boxes2D_on_image(rgb, boxes, scores):
#     boxes = boxes[0]
#     scores = scores[0]
#     inds = tf.where(tf.greater(scores, 0.0))
#     boxes = tf.gather(boxes, inds)
#     boxes = tf.reshape(boxes, [1, -1, 4])
#     boxes = tf.clip_by_value(boxes, 0.0, 1.0)
#     rgb = tf.image.draw_bounding_boxes(
#         tf.cast(back2color(rgb), tf.float32)/255., boxes)
#     return rgb

# def colorize_seg(seg, cmap='vkitti', vals=20):
#     # seg is B x H x W x 1
#     # this should be 1-indexed, so that 0 can be reserved for 'unknown'
#     # rgb = colorize(seg, normalize=False, cmap='tab20', vals=20)
#     rgb = colorize(seg, normalize=False, cmap=cmap, vals=vals)
#     rgb = tf.cast(255.0*rgb, tf.uint8)
#     return rgb

# def summarize_argmaxes(heat, scores, F, boxes_g=None, name='heat_argmax',
#                        bird=False, soft=False, justone=False):
#     # heat is B x H x W x N
#     # scores is B x N
#     B = int(heat.get_shape()[0])
#     H = int(heat.get_shape()[1])
#     W = int(heat.get_shape()[2])
#     N = int(heat.get_shape()[3])
#     # draw boxes around the argmaxes on the heatmap
#     # we draw the boxes F x F, which should be the filter size

#     if soft:
#         heat_ = tf.reshape(heat, [B, -1, N])
#         heat_ = tf.transpose(heat_, perm=[0, 2, 1])
#         spatial_weights = tf.nn.softmax(heat_, axis=2)
#         # this is B x N x H*W

#         ## this vis revealed to me that the current bad softmax is #
#         # probably just due to numerical issues; 
#         # the exp squishes small values very close together (at ~1)

#         # spatial_weights_ = tf.reshape(spatial_weights, [B, N, H, W])
#         # spatial_weights_ = tf.transpose(spatial_weights_, perm=[0, 2, 3, 1])
#         # spat_zer = oned2inferno(tf.expand_dims(spatial_weights_[:,:,:,0], axis=3))
#         # spat_one = oned2inferno(tf.expand_dims(spatial_weights_[:,:,:,1], axis=3))
#         # spat_all = oned2inferno(tf.reduce_max(spatial_weights_, axis=3, keepdims=True))
#         # if bird:
#         #     summarize_birdview_images(spat_zer, name+'_spat_zer')
#         #     summarize_birdview_images(spat_one, name+'_spat_one')
#         #     summarize_birdview_images(spat_all, name+'_spat_all')
#         # # else:
#         # #     tf.summary.image('%s_zer' % name, im_zer)

#         grid_y, grid_x = meshgrid2D(B, H, W)
#         # these are B x H x W
#         grid_x = tf.reshape(grid_x, [B, 1, H*W])
#         grid_y = tf.reshape(grid_y, [B, 1, H*W])
#         x = tf.reduce_sum(spatial_weights*grid_x, axis=2)
#         y = tf.reduce_sum(spatial_weights*grid_y, axis=2)
#     else:
#         y, x = argmax2D(heat)
#         # these are B x N
    
#     # these are B x N
#     x = tf.cast(tf.reshape(x, [B, N]), tf.float32)/float(W)
#     y = tf.cast(tf.reshape(y, [B, N]), tf.float32)/float(H)
#     dimY = float(F)/float(H)*tf.ones([B, N])
#     dimX = float(F)/float(W)*tf.ones([B, N])
#     # these are B x N
#     dims = tf.stack([dimY, dimX], axis=2)
#     centroids = tf.stack([y, x], axis=2)
#     # these are B x N x 2
#     boxes = utils_misc.get_boxes(centroids, dims, clip=True)
#     # boxes is B x N x 4

#     # scores is B x N
#     # use this to gather up the valid boxes before vis
#     # we'll do this just for the first el of the batch
#     boxes = boxes[0] # N x 4
#     heat = heat[0] # H x W x N
    
#     heat = tf.transpose(heat, perm=[2, 0, 1])
#     if not justone:
#         scores = scores[0] # N
#         inds = tf.where(tf.greater(scores, 0.0))
#         boxes = tf.gather(boxes, inds)
#         heat = tf.gather(heat, inds)
#     else:
#         boxes = boxes[0]
#         heat = heat[0]
#     boxes = tf.reshape(boxes, [1, -1, 4])
#     heat = tf.reshape(heat, [-1, H, W, 1])
#     heat = tf.transpose(heat, perm=[3, 1, 2, 0])

#     # pad to make sure we have stuff to draw
#     heat = tf.pad(heat, [[0, 0], [0, 0], [0, 0], [0, 2]])
#     boxes = tf.pad(boxes, [[0, 0], [0, 2], [0, 0]])

#     heat_zer = oned2inferno(tf.expand_dims(heat[:,:,:,0], axis=3))
#     heat_one = oned2inferno(tf.expand_dims(heat[:,:,:,1], axis=3))
#     heat_all = oned2inferno(tf.reduce_max(heat, axis=3, keepdims=True))

#     boxes0 = tf.expand_dims(boxes[:,0], axis=1)
#     boxes1 = tf.expand_dims(boxes[:,1], axis=1)
    
#     boxes = tf.clip_by_value(boxes, 0.0, 1.0)
#     boxes0 = tf.clip_by_value(boxes0, 0.0, 1.0)
#     boxes1 = tf.clip_by_value(boxes1, 0.0, 1.0)
#     im_zer = tf.image.draw_bounding_boxes(tf.cast(heat_zer, tf.float32)/255., boxes0)
#     im_one = tf.image.draw_bounding_boxes(tf.cast(heat_one, tf.float32)/255., boxes1)
#     im_all = tf.image.draw_bounding_boxes(tf.cast(heat_all, tf.float32)/255., boxes)
    
#     if boxes_g is not None:
#         boxes_g = boxes_g[0]
#         if not justone:
#             boxes_g = tf.gather(boxes_g, inds)
#         else:
#             boxes_g = boxes_g[0]
#         boxes_g = tf.reshape(boxes_g, [1, -1, 4])
#         boxes_g= tf.pad(boxes_g, [[0, 0], [0, 2], [0, 0]])

#         boxes_g0 = tf.expand_dims(boxes_g[:,0], axis=1)
#         boxes_g1 = tf.expand_dims(boxes_g[:,1], axis=1)
        
#         boxes_g = tf.clip_by_value(boxes_g, 0.0, 1.0)
#         boxes_g0 = tf.clip_by_value(boxes_g0, 0.0, 1.0)
#         boxes_g1 = tf.clip_by_value(boxes_g1, 0.0, 1.0)
#         im_zer = tf.image.draw_bounding_boxes(im_zer, boxes_g0)
#         im_one = tf.image.draw_bounding_boxes(im_one, boxes_g1)
#         im_all = tf.image.draw_bounding_boxes(im_all, boxes_g)
        
#     if bird:
#         summarize_birdview_images(im_zer, name+'_zer')
#         if not justone:
#             summarize_birdview_images(im_one, name+'_one')
#         summarize_birdview_images(im_all, name+'_all')
#         return prep_birdview_vis(im_all)
#     else:
#         tf.summary.image('%s_zer' % name, im_zer)
#         if not justone:
#             tf.summary.image('%s_one' % name, im_one)
#         tf.summary.image('%s_all' % name, im_all)
#         return im_all


# def summarize_argmaxes_zoom(heat, scores, box3D, F, boxes_g=None, name='heat_argmax',
#                             bird=False, soft=False):
#     # heat is B x H x W x N
#     # scores is B x N
#     B = int(heat.get_shape()[0])
#     H = int(heat.get_shape()[1])
#     W = int(heat.get_shape()[2])
#     N = int(heat.get_shape()[3])
#     # draw boxes around the argmaxes on the heatmap
#     # we draw the boxes F x F, which should be the filter size

#     if soft:
#         heat_ = tf.reshape(heat, [B, -1, N])
#         heat_ = tf.transpose(heat_, perm=[0, 2, 1])
#         spatial_weights = tf.nn.softmax(heat_, axis=2)
#         # this is B x N x H*W

#         grid_y, grid_x = meshgrid2D(B, H, W)
#         # these are B x H x W
#         grid_x = tf.reshape(grid_x, [B, 1, H*W])
#         grid_y = tf.reshape(grid_y, [B, 1, H*W])
#         x = tf.reduce_sum(spatial_weights*grid_x, axis=2)
#         y = tf.reduce_sum(spatial_weights*grid_y, axis=2)
#     else:
#         y, x = argmax2D(heat)
#         # these are B x N

#     # print 'got argmaxes...'
#     # x = tf.Print(x, [x], 'x', summarize=24)
#     # y = tf.Print(y, [y], 'y', summarize=24)

#     x = tf.cast(tf.reshape(x, [B, N]), tf.float32)
#     y = tf.cast(tf.reshape(y, [B, N]), tf.float32)
#     z = tf.tile(tf.constant([0.0], shape=[1, 1]), [B, N])

#     XYZ = tf.stack([x, y, z], axis=2)
#     # XYZ is B x N x 3

#     x_, y_, z_, _, _, _, _ = tf.unstack(box3D, axis=1)
#     xyz = tf.stack([x_, y_, z_], axis=1)
    
#     XYZ = utils_geom.Zoom2Bird(XYZ, xyz)
#     x, y, _ = tf.unstack(XYZ, axis=2)
#     # these are B x N

#     # x = tf.Print(x, [x], 'x', summarize=24)
#     # y = tf.Print(y, [y], 'y', summarize=24)

#     # put them into normalized coords
#     # x = x/float(W)
#     # y = y/float(H)

#     print 'GET RID OF HYP HERE'
#     x = x/float(hyp.BX)
#     y = y/float(hyp.BY)
    
    
#     dimY = float(F)/float(H)*tf.ones([B, N])
#     dimX = float(F)/float(W)*tf.ones([B, N])
#     # these are B x N
#     dims = tf.stack([dimY, dimX], axis=2)
#     centroids = tf.stack([y, x], axis=2)
#     # these are B x N x 2
#     boxes = utils_misc.get_boxes(centroids, dims, clip=True)
#     # boxes is B x N x 4

#     # scores is B x N
#     # use this to gather up the valid boxes before vis
#     # we'll do this just for the first el of the batch
#     scores = scores[0] # N
#     boxes = boxes[0] # N x 4
#     heat = heat[0] # H x W x N

#     inds = tf.where(tf.greater(scores, 0.0))
#     boxes = tf.gather(boxes, inds)
#     boxes = tf.reshape(boxes, [1, -1, 4])
#     # boxes is 1 x M x H x W
#     heat = tf.transpose(heat, perm=[2, 0, 1])
#     heat = tf.gather(heat, inds)
#     heat = tf.reshape(heat, [-1, H, W, 1])
#     heat = tf.transpose(heat, perm=[3, 1, 2, 0])
#     # heat is 1 x H x W x M

#     # pad to make sure we have stuff to draw
#     heat = tf.pad(heat, [[0, 0], [0, 0], [0, 0], [0, 2]])
#     boxes = tf.pad(boxes, [[0, 0], [0, 2], [0, 0]])

#     heat_zer = oned2inferno(tf.expand_dims(heat[:,:,:,0], axis=3))
#     heat_one = oned2inferno(tf.expand_dims(heat[:,:,:,1], axis=3))
#     heat_all = oned2inferno(tf.reduce_max(heat, axis=3, keepdims=True))

#     boxes0 = tf.expand_dims(boxes[:,0], axis=1)
#     boxes1 = tf.expand_dims(boxes[:,1], axis=1)
    
#     boxes = tf.clip_by_value(boxes, 0.0, 1.0)
#     boxes0 = tf.clip_by_value(boxes0, 0.0, 1.0)
#     boxes1 = tf.clip_by_value(boxes1, 0.0, 1.0)
#     im_zer = tf.image.draw_bounding_boxes(tf.cast(heat_zer, tf.float32)/255., boxes0)
#     im_one = tf.image.draw_bounding_boxes(tf.cast(heat_one, tf.float32)/255., boxes1)
#     im_all = tf.image.draw_bounding_boxes(tf.cast(heat_all, tf.float32)/255., boxes)
    
#     if boxes_g is not None:
#         boxes_g = boxes_g[0]
#         boxes_g = tf.gather(boxes_g, inds)
#         boxes_g = tf.reshape(boxes_g, [1, -1, 4])
#         boxes_g= tf.pad(boxes_g, [[0, 0], [0, 2], [0, 0]])

#         boxes_g0 = tf.expand_dims(boxes_g[:,0], axis=1)
#         boxes_g1 = tf.expand_dims(boxes_g[:,1], axis=1)
        
#         boxes_g = tf.clip_by_value(boxes_g, 0.0, 1.0)
#         boxes_g0 = tf.clip_by_value(boxes_g0, 0.0, 1.0)
#         boxes_g1 = tf.clip_by_value(boxes_g1, 0.0, 1.0)
#         im_zer = tf.image.draw_bounding_boxes(im_zer, boxes_g0)
#         im_one = tf.image.draw_bounding_boxes(im_one, boxes_g1)
#         im_all = tf.image.draw_bounding_boxes(im_all, boxes_g)
        
#     if bird:
#         summarize_birdview_images(im_zer, name+'_zer')
#         summarize_birdview_images(im_one, name+'_one')
#         summarize_birdview_images(im_all, name+'_all')
#         return prep_birdview_vis(im_all)
#     else:
#         tf.summary.image('%s_zer' % name, im_zer)
#         tf.summary.image('%s_one' % name, im_one)
#         tf.summary.image('%s_all' % name, im_all)
#         return im_all

def summ_traj_on_mem(xyz, mem, name='traj', max_outputs=1):
    # xyz is B x S x 3
    # mem is B x H x W x D x 1
    
    B = int(mem.get_shape()[0])
    H = int(mem.get_shape()[1])
    W = int(mem.get_shape()[2])
    D = int(mem.get_shape()[3])
    C = int(mem.get_shape()[4])
    S = int(xyz.get_shape()[1])

    # make it 1d (if it isn't already)
    mem = tf.reduce_mean(mem, axis=4, keepdims=True)

    xyz = voxelizer.Ref2Mem(xyz, H, W, D)
    x, y, z = tf.unstack(xyz, axis=2)

    # for vis, get rid of the height dim
    mem = tf.reduce_mean(mem, axis=1)
    # show the xz traj
    # heats = utils_misc.get_target_prior(B, W, D, x, z, sigma=2)
    heats = utils_misc.get_target_prior(B, D, W, z, x, sigma=2)
    im = seq2color_on_oned(heats, mem)
    # we cannot call summ_rgb since it is float
    # but this is easy enough:
    im = prep_birdview_vis(im)
    tf.summary.image(name, im, max_outputs=max_outputs)
    
# def summ_trajs_on_lidar(xy_e, xy_g, occM, name='traj'):
#     # xy_e and _g are both B x S x 2
#     # occM is the projected lidar of the middle frame, sized B x H x W x 1
    
#     B = int(occM.get_shape()[0])
#     S = int(xy_e.get_shape()[1])
#     H = int(occM.get_shape()[1])
#     W = int(occM.get_shape()[2])

#     x_e, y_e = tf.unstack(xy_e, axis=2)
#     x_e, y_e = utils_geom.Velo2Bird2(x_e, y_e)
    
#     x_g, y_g = tf.unstack(xy_g, axis=2)
#     x_g, y_g = utils_geom.Velo2Bird2(x_g, y_g)

#     heats_e = utils_misc.get_target_prior(B, H, W, x_e, y_e, sigma=3)
#     heats_g = utils_misc.get_target_prior(B, H, W, x_g, y_g, sigma=8)

#     # toss the "e" stuff for the past, since it is not really predicted
#     mask = tf.concat([tf.zeros([B, H, W, S/2]), tf.ones([B, H, W, S/2+1])], axis=3)
#     heats_e = heats_e*mask

#     # heats_e_vis = seq2color_on_oned(heats_e, tf.zeros_like(occM), 'bwr')
#     heats_e_vis = seq2color_on_oned(heats_e, tf.zeros_like(occM), colormap='RdYlGn')
#     heats_g_vis = seq2color_on_oned(heats_g, occM, colormap='RdBu')


#     vis = overlay_1on2(heats_e_vis, heats_g_vis)
#     tf.summary.image(name, vis, max_outputs=1)

# def summ_many_trajs_on_bird(many_xy_e, xy_g, bird_im, name='traj'):
#     # many_xy_e is K x B x S x 2
#     # xy_g is B x S x 2
#     # bird_im is the projected lidar of the reference frame, sized B x H x W x 1

#     K = int(many_xy_e.get_shape()[0])
#     B = int(many_xy_e.get_shape()[1])
#     S = int(many_xy_e.get_shape()[2])
#     H = int(bird_im.get_shape()[1])
#     W = int(bird_im.get_shape()[2])

#     # to save mem, let's only use the first el of the batch
#     # but we will preserve the batch dim for convenience
#     many_xy_e = tf.expand_dims(many_xy_e[:,0], axis=1)
#     xy_g = tf.expand_dims(xy_g[0], axis=0)
#     bird_im = tf.expand_dims(bird_im[0], axis=0)
#     B = 1
    
#     xy_e = tf.reshape(many_xy_e, [B*K, S, 2])
#     x_e, y_e = tf.unstack(xy_e, axis=2)
#     x_e, y_e = voxelizer.Velo2Bird2(x_e, y_e, H, W)
#     x_g, y_g = tf.unstack(xy_g, axis=2)
#     x_g, y_g = voxelizer.Velo2Bird2(x_g, y_g, H, W)

#     heats_e = utils_misc.get_target_prior(B*K, H, W, x_e, y_e, sigma=2)
#     heats_g = utils_misc.get_target_prior(B, H, W, x_g, y_g, sigma=6)

#     # # only keep the "e" stuff for curr and future
#     # mask = tf.concat([tf.zeros([B*K, H, W, S/2]), tf.ones([B*K, H, W, S/2+1])], axis=3)
#     # heats_e = heats_e*mask

#     heats_e = tf.reshape(heats_e, [B, K, H, W, S])
#     heats_e = tf.reduce_max(heats_e, axis=1)

#     # heats_e_vis = seq2color_on_oned(heats_e, tf.zeros_like(bird_im), 'bwr')
#     heats_e_vis = seq2color_on_oned(heats_e, tf.zeros_like(bird_im), colormap='RdYlGn')
#     heats_g_vis = seq2color_on_oned(heats_g, bird_im, colormap='RdBu')
#     vis = overlay_1on2(heats_e_vis, heats_g_vis)

#     vis = prep_birdview_vis(vis)
#     tf.summary.image(name, vis, max_outputs=1)


# def summ_seg(name='seg', im=None, cmap='vkitti', vals=20, bird=False):
#     if bird:
#         im = prep_birdview_vis(im)
#     tf.summary.image(name, colorize_seg(im, cmap, vals), max_outputs=1)
    
def summ_oned(name='oned', im=None, norm=True, is3D=False, max_outputs=1):
    if is3D:
        im = prep_birdview_vis(im)
    tf.summary.image(name, oned2inferno(im, norm=norm), max_outputs=max_outputs)

def summ_oneds(name='oned', ims=None, norm=True, is3D=False, maxwidth=MAXWIDTH):
    if is3D:
        ims = [prep_birdview_vis(im) for im in ims]
    im = gif_and_tile(ims)
    # colorize needs B x H x W x C
    B, S, H, W, C = im.get_shape().as_list()
    im = tf.reshape(im, [B*S, H, W, C])
    vis = oned2inferno(im, norm=norm)
    vis = tf.reshape(vis, [B, S, H, W, 3])
    if W > maxwidth:
        vis = vis[:,:,:,:maxwidth]
    summ_gif(name, vis, max_outputs=1, fps=8)

def convert_occ_to_height(occ):
    B, H, W, D, C = occ.get_shape().as_list()
    assert(C==1)
    # note that height increases DOWNWARD in the tensor
    # (like pixel/camera coordinates)
    
    height = tf.lin_space(float(H), 1.0, H)
    height = tf.reshape(height, [1, H, 1, 1, 1])
    height = tf.reduce_max(occ*height, axis=1)/float(D)
    height = tf.reshape(height, [B, W, D, C])
    return height
    
def summ_occ(name='occ', occ=None, maxwidth=MAXWIDTH):
    B, H, W, D, C = occ.get_shape().as_list()
    height = convert_occ_to_height(occ)
    
    # make "forward" point up, and make "right" point right
    height = tf.map_fn(tf.image.rot90, height)
    height = tf.reshape(height, [B, D, W, C])
    
    summ_oned(name=name, im=height, norm=False, is3D=False)
    
def summ_occs(name='occs', occs=None, maxwidth=MAXWIDTH):
    B, H, W, D, C = occs[0].get_shape().as_list()
    heights = [convert_occ_to_height(occ) for occ in occs]
    
    # make "forward" point up, and make "right" point right
    heights = [tf.map_fn(tf.image.rot90, height) for height in heights]
    heights = [tf.reshape(height, [B, D, W, C]) for height in heights]
    
    summ_oneds(name=name, ims=heights, norm=False, is3D=False)

def summ_images(name='images', images=None, maxwidth=MAXWIDTH):
    B, H, W, C = images[0].get_shape().as_list()
    if C==1:
        summ_oneds(name=name, ims=images)
    elif C==3:
        summ_rgbs(name=name, ims=images)
    else:
        images = [tf.reduce_mean(im, axis=3, keepdims=True) for im in images]
        summ_oneds(name=name, ims=images)

def summ_image(name='image', image=None, maxwidth=MAXWIDTH):
    B, H, W, C = image.get_shape().as_list()
    if C==1:
        summ_oned(name=name, im=image)
    elif C==3:
        summ_rgb(name=name, im=image)
    else:
        image = tf.reduce_mean(image, axis=3, keepdims=True)
        summ_oned(name=name, im=image)

def summ_feats(name='feats', feats=None, maxwidth=MAXWIDTH):
    feats = tf.stack(feats, axis=1)
    B, S, H, W, D, C = feats.get_shape().as_list()

    feats = tf.reduce_mean(tf.abs(feats), axis=5, keepdims=True)
    # feats is B x S x H x W x D x 1
    feats = tf.reduce_mean(feats, axis=2)
    # feats is B x S x W x D x 1
    feats = tf.unstack(feats, axis=1)
    
    # make "forward" point up, and make "right" point right
    feats = [tf.map_fn(tf.image.rot90, feat) for feat in feats]
    feats = [tf.reshape(feat, [B, D, W, 1]) for feat in feats]
    
    summ_oneds(name=name, ims=feats, norm=True, is3D=False)

def summ_feat(name='feat', feat=None, maxwidth=MAXWIDTH):
    B, H, W, D, C = feat.get_shape().as_list()

    feat = tf.reduce_mean(tf.abs(feat), axis=4, keepdims=True)
    # feat is B x H x W x D x 1
    feat = tf.reduce_mean(feat, axis=1)
    # feat is B x W x D x 1
    
    # make "forward" point up, and make "right" point right
    feat = tf.map_fn(tf.image.rot90, feat)
    feat = tf.reshape(feat, [B, D, W, 1])
    
    summ_oned(name=name, im=feat, norm=True, is3D=False)

def summ_unp(name='unp', unp=None, occ=None, maxwidth=MAXWIDTH):
    B, H, W, D, C = unp.get_shape().as_list()
    occ = tf.tile(occ, [1, 1, 1, 1, C])
    unp = reduce_masked_mean(unp, occ, axis=1)
    unp = tf.map_fn(tf.image.rot90, unp)
    summ_rgb(name=name, im=unp, blacken_zeros=True)

def get_unp_vis(unp, occ):
    B, H, W, D, C = unp.get_shape().as_list()
    occ = tf.tile(occ, [1, 1, 1, 1, C])
    unp = reduce_masked_mean(unp, occ, axis=1)
    return unp
    
def get_unps_vis(unps, occs):
    B, S, H, W, D, C = unps.get_shape().as_list()
    occs = tf.tile(occs, [1, 1, 1, 1, 1, C])
    unps = reduce_masked_mean(unps, occs, axis=2)
    # unps is B x S x W x D x C
    return unps
    
def summ_unps(name='unps', unps=None, occs=None, maxwidth=MAXWIDTH):
    unps = tf.stack(unps, axis=1)
    occs = tf.stack(occs, axis=1)
    B, S, H, W, D, C = unps.get_shape().as_list()

    occs = tf.tile(occs, [1, 1, 1, 1, 1, C])
    unps = reduce_masked_mean(unps, occs, axis=2)

    unps = tf.unstack(unps, axis=1)
    unps = [tf.map_fn(tf.image.rot90, unp) for unp in unps]
    summ_rgbs(name=name, ims=unps, blacken_zeros=True) 
    
# def summ_depth(name='depth', im=None, valid=None, already_inv=False, maxdepth=60.0):
#     tf.summary.image(name, depth2inferno(im, valid=valid, already_inv=already_inv, maxdepth=maxdepth), max_outputs=1)
    
def summ_depths(name='depth', ims=None, valids=None, already_inv=False, maxdepth=60.0, maxwidth=MAXWIDTH):
    im = gif_and_tile(ims)
    if valids is not None:
        valid = gif_and_tile(valids)
    else:
        valid = tf.ones_like(im)

    # colorize needs B x H x W x C
    B, S, H, W, C = im.get_shape().as_list()
    im = tf.reshape(im, [B*S, H, W, C])
    valid = tf.reshape(valid, [B*S, H, W, 1])
    vis = depth2inferno(im, valid=valid, already_inv=already_inv, maxdepth=maxdepth)
    vis = tf.reshape(vis, [B, S, H, W, 3])
    B, S, H, W, C = vis.get_shape().as_list()
    if W > maxwidth:
        vis = vis[:,:,:,:maxwidth]
    summ_gif(name, vis, max_outputs=1, fps=8)
    
def summ_rgb(name='rgb', im=None, blacken_zeros=False, is3D=False):
    if is3D:
        im = prep_birdview_vis(im)
    tf.summary.image(name, back2color(im, blacken_zeros=blacken_zeros), max_outputs=1)

def summ_flow(name='flow', im=None, clip=0.0, is3D=False):
    if is3D:
        im = prep_birdview_vis(im)
    tf.summary.image(name, flow2color(im, clip=clip), max_outputs=1)
    
def summ_rgbs_and_oned(name='rgbs_and_oned', rgb0=None, rgb1=None, oned=None, clip=True, is3D=False):
    # show a gif of the rgbs on the left, and the oned on the right
    if is3D:
        oned = prep_birdview_vis(oned)
        rgb0 = prep_birdview_vis(rgb0)
        rgb1 = prep_birdview_vis(rgb1)
    oned_vis = oned2inferno(oned)
    rgb0_vis = back2color(rgb0)
    rgb1_vis = back2color(rgb1)
    left = tf.stack([rgb0_vis, rgb1_vis], axis=1)
    right = tf.stack([oned_vis, oned_vis], axis=1)
    full = tf.concat([left, right], axis=3) # axis3 is width here
    summ_gif(name, full, max_outputs=1, fps=8)

def summ_rgbs_and_flow(name='rgbs_and_flow', rgb0=None, rgb1=None, flow=None, clip=1.0, is3D=False):
    flow_xz = tf.stack([flow[:,:,:,:,0], flow[:,:,:,:,2]], axis=4) # grab x, z
    # this is B x H x W x D x 2
    flow_xz = tf.reduce_mean(flow_xz, axis=1) # reduce over H (y)
    # this is B x W x D x 2
    
    # show a gif of the rgbs on the left, and the oned on the right
    if is3D:
        flow_xz = prep_birdview_vis(flow_xz)
        rgb0 = prep_birdview_vis(rgb0)
        rgb1 = prep_birdview_vis(rgb1)
    flow_vis = flow2color(flow_xz, clip=clip)
    rgb0_vis = back2color(rgb0)
    rgb1_vis = back2color(rgb1)
    left = tf.stack([rgb0_vis, rgb1_vis], axis=1)
    right = tf.stack([flow_vis, flow_vis], axis=1)
    full = tf.concat([left, right], axis=3) # axis3 is width here
    summ_gif(name, full, max_outputs=1, fps=8)

def gif_and_tile(ims):
    S = len(ims)
    # i want a gif in the left, and the tiled frames on the right
    # for the gif tool, this means making a B x S x H x W tensor
    # where the leftmost part is sequential and the rest is tiled
    gif = tf.stack(ims, axis=1)
    til = tf.concat(ims, axis=2)
    til = tf.tile(tf.expand_dims(til, axis=1), [1, S, 1, 1, 1])
    im = tf.concat([gif, til], axis=3)
    return im
    
def summ_rgbs(name='rgb', ims=None, blacken_zeros=False, is3D=False, maxwidth=MAXWIDTH):
    if is3D:
        ims = [prep_birdview_vis(im) for im in ims]
    im = gif_and_tile(ims)
    vis = back2color(im, blacken_zeros=blacken_zeros)
    B, S, H, W, C = vis.get_shape().as_list()
    if W > maxwidth:
        vis = vis[:,:,:,:maxwidth]
    summ_gif(name, vis, max_outputs=1, fps=8)

def encodeGifFfmpeg(images, fps):
    from subprocess import Popen, PIPE
    S,H,W,C = images.shape
    cmd = ['ffmpeg', '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-r', '%.02f' % fps,
           '-s', '%dx%d' % (W,H),
           '-pix_fmt', {1: 'gray', 3: 'rgb24'}[C],
           '-i', '-',
           '-filter_complex',
           '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
           '-r', '%.02f' % fps,
           '-f', 'gif',
           '-']
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc
    return out

def encodeGifMoviepy(images, fps):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    clip = ImageSequenceClip(list(images), fps)
    tmpfile = os.tmpnam() + '.gif'
    clip.write_gif(tmpfile, verbose=False, logger=None)
    out = open(tmpfile, 'rb').read()
    os.remove(tmpfile)
    return out

def py_gif_summary(tag, images, max_outputs, fps):
    tag = tag.numpy()
    images = images.numpy()
    max_outputs = max_outputs.numpy()
    fps = fps.numpy()
    B,S,H,W,C = images.shape
    summ = tf.Summary()
    num_outputs = min(B, max_outputs)
    for i in range(num_outputs):
        image_summ = tf.Summary.Image()
        image_summ.height     = H
        image_summ.width      = W
        image_summ.colorspace = C  # 1: grayscale, 3: RGB
        try:
            enc_im_str = encodeGifFfmpeg(images[i], fps)
            raise Exception('foo')
        except Exception as e:
            # print e
            try:
                enc_im_str = encodeGifMoviepy(images[i], fps)
            except Exception as e:
                # print e
                tf.logging.warning("Unable to encode images to a gif.")
                enc_im_str = ''
        image_summ.encoded_image_string = enc_im_str            
        summ_tag = "%s/gif_%02d" % (tag,i)
        summ.value.add(tag=summ_tag, image=image_summ)
    summ_str = summ.SerializeToString()
    return summ_str

# def summ_rgb_gif(name, tensor, max_outputs=8, fps=3):
#     tensor = back2color(tensor)
#     summ_gif(name, tensor, max_outputs=max_outputs, fps=fps)

def summ_gif(name, tensor, max_outputs=8, fps=3,
             family=None, collections=None):
    from tensorflow.python.ops import summary_op_util
    if summary_op_util.skip_summary():
        return
    assert tensor.dtype in {tf.uint8,tf.float32}
    shape = tensor.shape.as_list()
    assert len(shape) in {4,5}
    assert shape[4] in {1,3}
    if len(shape) == 4:
        tensor = tf.expand_dims(tensor, axis=0)
    if tensor.dtype == tf.float32:
        tensor = back2color(tensor)
    with summary_op_util.summary_scope(name, family=family,
                                       values=[tensor]) as (tag, scope):
        tag = tf.convert_to_tensor(str(tag))
        with tf.device('/device:CPU:*'):
            summ = tf.py_function(func=py_gif_summary,
                                  inp=[tag, tensor, max_outputs, fps],
                                  Tout=tf.dtypes.string,
                                  name=scope)
        summary_op_util.collect(summ, collections, [tf.GraphKeys.SUMMARIES])

    

def colorize(value, normalize=True, vmin=None, vmax=None, cmap=None, vals=255):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.

    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
      - vals: the number of values in the cmap minus one

    Example usage:

    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """
    value = tf.squeeze(value, axis=3)

    if normalize:
        vmin = tf.reduce_min(value) if vmin is None else vmin
        vmax = tf.reduce_max(value) if vmax is None else vmax
        value = (value - vmin) / (vmax - vmin) # vmin..vmax

        # dma = tf.reduce_max(value)
        # dma = tf.Print(dma, [dma], 'dma', summarize=16)
        # tf.summary.histogram('dma', dma) # just so tf.Print works
        
        # quantize
        indices = tf.cast(tf.round(value * float(vals)), tf.int32)
    else:
        # quantize
        indices = tf.cast(value, tf.int32)

    # 00 Unknown 0 0 0
    # 01 Terrain 210 0 200
    # 02 Sky 90 200 255
    # 03 Tree 0 199 0
    # 04 Vegetation 90 240 0
    # 05 Building 140 140 140
    # 06 Road 100 60 100
    # 07 GuardRail 255 100 255
    # 08 TrafficSign 255 255 0
    # 09 TrafficLight 200 200 0
    # 10 Pole 255 130 0
    # 11 Misc 80 80 80
    # 12 Truck 160 60 60
    # 13 Car:0 200 200 200
  
    if cmap=='vkitti':
        colors = np.array([0, 0, 0,
                           210, 0, 200,
                           90, 200, 255,
                           0, 199, 0,
                           90, 240, 0,
                           140, 140, 140,
                           100, 60, 100,
                           255, 100, 255,
                           255, 255, 0,
                           200, 200, 0,
                           255, 130, 0,
                           80, 80, 80,
                           160, 60, 60,
                           200, 200, 200,
                           230, 208, 202]);
        colors = np.reshape(colors, [15, 3]).astype(np.float32)/255.0
        colors = tf.constant(colors)
    else:
        # gather
        cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
        if cmap=='RdBu' or cmap=='RdYlGn':
            colors = cm(np.arange(256))[:, :3]
        else:
            colors = cm.colors
        colors = np.array(colors).astype(np.float32)
        colors = np.reshape(colors, [-1, 3])
        colors = tf.constant(colors, dtype=tf.float32)
    
    value = tf.gather(colors, indices)
    # value is float32, in [0,1]
    return value

def summ_3D_flow(flow, mod='', clip=1.0):
    tf.summary.histogram('flow_x%s' % mod, flow[:,:,:,:,0])
    tf.summary.histogram('flow_y%s' % mod, flow[:,:,:,:,1])
    tf.summary.histogram('flow_z%s' % mod, flow[:,:,:,:,2])
    # flow is B x H x W x D x 3; inside the 3 it's XYZ
    flow_xz = tf.stack([flow[:,:,:,:,0], flow[:,:,:,:,2]], axis=4) # grab x, z
    flow_xy = tf.stack([flow[:,:,:,:,0], flow[:,:,:,:,1]], axis=4) # grab x, y
    flow_yz = tf.stack([flow[:,:,:,:,1], flow[:,:,:,:,2]], axis=4) # grab y, z
    # these are B x H x W x D x 2
    flow_xz = tf.reduce_mean(flow_xz, axis=1) # reduce over H (y)
    flow_xy = tf.reduce_mean(flow_xy, axis=3) # reduce over D (z)
    flow_yz = tf.reduce_mean(flow_yz, axis=2) # reduce over W (x)
    summ_flow('flow_xz%s' % mod, flow_xz, clip=clip, is3D=True) # rot90 for interp
    summ_flow('flow_xy%s' % mod, flow_xy, clip=clip)
    summ_flow('flow_yz%s' % mod, flow_yz, clip=clip)
    flow_mag = tf.reduce_mean(tf.reduce_sum(tf.sqrt(EPS+tf.square(flow)), axis=4, keepdims=True), axis=1)
    summ_oned('flow_mag%s' % mod, flow_mag, is3D=True)

def summ_proj_compare(proj, proj1, name):
    summ_image('%s_proj'%(name), tf.reduce_mean(proj, axis=3))
    summ_image('%s_proj1'%(name), tf.reduce_mean(proj1, axis=3))
    summ_image('%s_proj_diff'%(name), tf.reduce_mean((proj-proj1)**2, axis=3))
    tf.summary.scalar('%s_proj_diff_scalar'%(name), tf.reduce_mean((proj-proj1)**2))
