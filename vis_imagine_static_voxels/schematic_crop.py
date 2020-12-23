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

def read_bv_schematic(fn,scene_file):
    fileformat = 'schematic'
    with open(scene_file) as f:
        scene_json = json.load(f)
    scene_obj_list = scene_json['objects']
    print("num objs ",len(scene_obj_list))
    # check if extra object voxels need to be removed from blocks
    remove_extra_objects = True
    orig_block_id = int(scene_obj_list[0]['obj_id'].split('blockid_')[-1])
    if len(scene_obj_list) == 1:
        remove_extra_objects = True
        orig_block_id = int(scene_obj_list[0]['obj_id'].split('blockid_')[-1])
    # load image
    if fileformat == 'schematic':
        voxel_file = fn
        sf = SchematicFile.load(voxel_file)
        blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
        voxel_size = int(round(len(blocks)**(1./3)))
        blocks = blocks.reshape((voxel_size,voxel_size,voxel_size))
        blocks = blocks.copy()
        if remove_extra_objects:
            blocks[blocks != orig_block_id] = 0
    # st()
    data = np.float32(blocks)
    return data
import glob
voxels = glob.glob("voxel_shapes/real_shapes/*matic")
# path = "alsfjdal;"    
# st()
for view_path in voxels:
    tree_path = view_path.replace("schematics","trees").replace("schematic",'tree')
    scene_path = view_path.replace("schematics","scenes").replace("schematic",'json')
    tree = pickle.load(open(tree_path,"rb"))
    bx, by, bz, bx_d, by_d, bz_d  = tree.bbox
    binvox = read_bv_schematic(view_path,scene_path)
    padding = 0 
    print(tree_path,tree.word)
    binvox = np.transpose(binvox, [0, 2, 1])
    binvox = binvox[bz-padding:bz+bz_d+padding+1,by-padding:by+by_d+padding+1,bx-padding:bx+bx_d+padding+1]
    canvas = np.zeros([16,16,16])
    canvas[4:4+binvox.shape[0],4:4+binvox.shape[1],4:4+binvox.shape[2]] = binvox
    # st()
    st()
    pickle.dump(canvas,open(view_path.replace("schematic","p"),"wb"))
    save_voxel(canvas,view_path.replace("schematics","binvox_filter").replace("schematic",'binvox'))

    # st()
    # print(binvox.shape)
