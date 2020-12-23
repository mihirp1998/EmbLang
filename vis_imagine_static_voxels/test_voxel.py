import tensorflow as tf
# tf.enable_eager_execution()
from utils import binvox_rw
from utils import voxel
import constants as const
import numpy as np
from scipy.misc import imsave
import ipdb
import imageio

st = ipdb.set_trace

def save_voxel(voxel_, filename, THRESHOLD=0.5):
	S1 = voxel_.shape[2]
	S2 = voxel_.shape[1]
	S3 = voxel_.shape[0]
	# st()
	binvox_obj = binvox_rw.Voxels(
		np.transpose(voxel_, [2, 1, 0]) >= THRESHOLD,
		dims = [S1, S2, S3],
		translate = [0.0, 0.0, 0.0],
		scale = 1.0,
		axis_order = 'xyz'
	)   

	with open(filename, "wb") as f:
		binvox_obj.write(f)

def rotate_voxels(rep,angle):
	a = binvox_rw.read_as_3d_array(open("tmpTest/outline_scale_47.0.binvox","rb"))
	val = a.data

	val = tf.convert_to_tensor(np.expand_dims(np.expand_dims(val,0),-1))
	phi,theta = angle
	rot_mat = voxel.get_transform_matrix_tf([theta], [phi])

	proj_val = voxel.rotate_voxel(val,rot_mat)
	num = np.where(proj_val>0.5)[0]

	proj_val = np.squeeze(proj_val)
	proj_val = proj_val >0.5
	# st()
	proj_imgZ = np.mean(proj_val,0)

	imsave('{}/valRotate_phi_{}_theta_{}_fov_{}_Z.png'.format(rep,phi,theta,const.fov), proj_imgZ)

	# st()
	save_voxel(np.squeeze(proj_val),"{}/valRotate_THETA_{}_PHI_{}_fov_{}_.binvox".format(rep,theta,phi,const.fov))
# rotate_voxels
# rotate_voxels("tmpTest",[-20.0,0.0])

def project_voxel(rep):
	a = binvox_rw.read_as_3d_array(open("/Users/ashar/work/visual_imagination/prob_scene_gen/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/image_generation/output_90_20.binvox","rb"))
	val = a.data
	val = tf.convert_to_tensor(np.expand_dims(np.expand_dims(val,0),-1))
	proj_val = voxel.project_voxel(val)
	num = np.where(proj_val>0.5)[0]
	# if len(num) > 0:
	# 	print("found")
	# 	fovs_working[fov] = len(num)
	proj_val = np.squeeze(proj_val)
	proj_val = proj_val >0.5
	proj_imgZ = np.mean(proj_val,0)
	proj_imgY = np.mean(proj_val,1)
	proj_imgX = np.mean(proj_val,2)
	imsave('{}/valProject_fov_{}_Z.png'.format(rep,const.fov), proj_imgZ)
	imsave('{}/valProject_fov_{}_Y.png'.format(rep,const.fov), proj_imgY)
	imsave('{}/valProject_fov_{}_X.png'.format(rep,const.fov), proj_imgX)

	save_voxel(proj_val,"{}/valProject_fov_{}.binvox".format(rep,const.fov))

# project_voxel("tmpTest")

# unprojected_depth = (tf.expand_dims(inputs[:,:,:,:,1], 4) - const.radius) * (1/const.SCENE_SIZE)
def unproject(resize = False):
    # st()
    depth = np.array(imageio.imread("/Users/ashar/work/visual_imagination/prob_scene_gen/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/image_generation/rendered_depth_90_20.exr", format='EXR-FI'))[:,:,0]
    depth = np.array(imageio.imread("/Users/ashar/work/visual_imagination/prob_scene_gen/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/output/CLEVR_64_36_MAYHEM_AGAIN/depth/train/CLEVR_new_000000/CLEVR_new_000000_180_40.exr", format='EXR-FI'))[:,:,0]
    # depth =np.transpose(depth, [1, 0])
    inputs = depth * (100 - 0) + 0
    inputs.astype(np.float32)
        # st()
    if resize:
        inputs = tf.image.resize(inputs, (const.S, const.S))
    size = int(inputs.shape[1])
    inputs = np.expand_dims(np.expand_dims(inputs,axis=-1),0)
    #now unproject, to get our starting point
    inputs = voxel.unproject_image(inputs)

    #in addition, add on a z-map, and a local bias
    #copied from components.py
    meshgridz = tf.range(size, dtype = tf.float32)
    meshgridz = tf.reshape(meshgridz, (1, size, 1, 1))
    meshgridz = tf.tile(meshgridz, (1, 1, size, size))
    meshgridz = tf.expand_dims(meshgridz, axis = 4) 
    meshgridz = (meshgridz + 0.5) / (size/2) - 1.0 #now (-1,1)
    # st()
    #get the rough outline
    # unprojected_mask = tf.expand_dims(inputs[:,:,:,:,0], 4)
    # unprojected_depth = tf.expand_dims(inputs[:,:,:,:,0], 4)
    unprojected_depth = (tf.expand_dims(inputs[:,:,:,:,0], 4) - const.radius) * (1/const.SCENE_SIZE)
    # return unprojected_depth
    if const.H > 32:
        outline_thickness = 0.1
    else:
        outline_thickness = 0.2
    # depth shell
    outline = tf.cast(tf.logical_and(
        unprojected_depth <= meshgridz,
        unprojected_depth + outline_thickness > meshgridz
    ), tf.float32)
    # outline *= unprojected_mask
    if True:
        #return tf.expand_dims(inputs[:,:,:,:,0], 4) #this is the unprojected mask
        unprojected_depth = np.squeeze(unprojected_depth)
        val = np.squeeze(outline)
        save_voxel(val, "tmpTest/outline_scale_{}_{}.binvox".format(const.fov, 180))
        save_voxel(unprojected_depth, "tmpTest/unproj_depths_{}_{}.binvox".format(const.fov, 180))
        return outline,unprojected_depth

    inputs_ = [inputs]
    if const.USE_MESHGRID:
        inputs_.append(meshgridz)
    if const.USE_OUTLINE:
        inputs_.append(outline)
    inputs = tf.concat(inputs_, axis = 4)
    return inputs

unproject()


