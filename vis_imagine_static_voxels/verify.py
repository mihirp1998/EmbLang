import pickle
import numpy as np
import utils
from nbtschematic import SchematicFile
from ipdb import set_trace as st
import json
import pickle
from utils import binvox_rw

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


val = pickle.load(open("vis_embeddings/gqn3d_cond_tree_vae_concat_end_inf_segmentation_test/test/sphere gray large metal/000003_segmentation_masks.png","rb"))

# cube = pickle.load(open("voxel_shapes/real_shapes/cube.p","rb"))
# cylinder = pickle.load(open("voxel_shapes/real_shapes/cylinder.p","rb"))
val = pickle.load(open("voxel_shapes/real_shapes/sphere.p","rb"))

# cube_bce = utils.losses.binary_ce_loss(val,cube)
# cylinder_bce = utils.losses.binary_ce_loss(val,cylinder)
# sphere_bce = utils.losses.binary_ce_loss(val,sphere)
# print("cube",cube_bce)
# print("cylinder",cylinder_bce)
# print("spher",sphere_bce)


valrange= [i*0.05 for i in list(range(5,20))]
print(valrange)
for i in valrange:
	name = "ab_{}.binvox".format(i)
	st()
	save_voxel(val,name,THRESHOLD=i)