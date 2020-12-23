import sys
import math
import time

from options import OptionGroup as OG
import options
override_loading =False
emb_dim = 8
USE_MEAN = False
emb_coeff = 1
emb_l1_coeff = 0.1
visualize_inf = False
RANDOMNESS_THRESH = -1
loading=True
RANK_TENSORS = False
kl_vae_loss=False
opname = 'gqn3d'
eager = True
save_custom= False
LIMITED_DATASET = True
MASKED_DATASET = False
ARITH_MODE = False
concat_3d = True
SAMPLE_ANGLES = False
# unlikely to change constants:
classification=False
customTrees=False
# classification_stop=False
randomize =False
onlycolor = False
NUM_VIEWS = 3
NUM_PREDS = 1
l2mask = False
BS_NORM=True 
EMBEDDING_LOSS_3D = False
LOAD_VAL=False
SAMPLE_VALUES=False
COND_TREE_TO_IMG = False
# COND = False
segmentation = False
segmentation_stop = False
custominf=False
L2_loss_z = False
TREE_LOSS = True
DIRECT_TREE_LOSS = False
DIRECT_TREE_TO_IMG = False
OVERFIT_ON_SUBSET = False
USE_GT_BBOX = False
FREEZE_ENC_DEC = False
L2LOSS_DIST = False
mask3d = False
KLD_LOSS = False
renderer = True
MULTI_UNPROJ = True
AGGREGATION_METHOD = 'stack'
AUX_POSE = True
DUMP_TENSOR = False
l2mask3d = False
stochasticity = True 
PHI_IDX = None
CATS = 57 #number of categories
eps = 1E-8
GoModules= True
#V = 18  # number of views for multiview -- we are keeping phi frozen for now
frozen_decoder=False
Hdata = 64
Wdata = 64
inf_net = False

inject_summaries = False
summ_grads = False

# CHANGE THIS
HV = 12
VV = 3

MINH = 0
MAXH = 360 #exclusive
MINV = 0
MAXV = 30 #exclusive

H = 256
W = 256


HDELTA = (MAXH-MINH) / HV #20
VDELTA = (MAXV-MINV) / VV #10

fov = 47.0
radius = 12.0

bn_decay = 0.999

# since the scene size is always exactly 1
# if the actual scene size is larger or smaller
# you should scale the radius instead
# a scene twice as big is equivalent to having
# a radius half as large

#this should probably never be changed for an reason
SCENE_SIZE = 8.0

# object mask for loss
MASKWEIGHT = 2.0

S = 128  # cube size
BS = 2
SS = 16
NS = BS * SS
NB_STEPS = 1000000

ORIENT = True
STNET = False


ARCH = 'unproj'
#options: 'unproj, marr'

NET3DARCH = 'marr' #or '3x3, marr'
USE_OUTLINE = True
USE_MESHGRID = True
USE_LOCAL_BIAS = False #set to false later

INPUT_RGB = False
INPUT_POSE = False
VOXNET_LATENT = 512


# test/train mode
mode = 'train'

# input constants
train_file = 'train'
val_file = 'valid'
test_file = 'test'

# optimizer consts
lr = 1E-4
mom = 0.9

# validation period
valp = 1000
savep = 5000

# important directories


tb_dir = 'log'
data_dir = 'data'
ckpt_dir = 'ckpt'
ckpt_cfg_dir = 'ckpt_cfg'
stop = False
# debug flags
FAKE_NET = False
REPROJ_SINGLE = False
ADD_FEEDBACK = False
VALIDATE_INPUT = False
DEBUG_MODE = False
DEBUG_32 = False
DEBUG_HISTS = False
DEBUG_PLACEMENT = False
DEBUG_VOXPROJ = False
DEBUG_VOXNET = False
DEBUG_REPROJ = False
DEBUG_EXPORTS = True
DEBUG_SPEED = True
DEBUG_NET = False
DEBUG_RP = False
DEBUG_FULL_TRACE = False
DEBUG_NODE_TRACE = False
DEBUG_NAN = False
DEBUG_CACHE = False
DEBUG_LOSSES = True
DEBUG_MEMORY = False
DEBUG_UNPROJECT = False

SKIP_RUN = False
SKIP_TRAIN_EXPORT = False
SKIP_VAL_EXPORT = False
SKIP_EXPORT = False

USE_GRAVITY_LOSS = False

FIX_VIEW = False
STOP_PRED_DELTA = True
STOP_REPROJ_MASK_GRADIENTS = False

USE_TIMELINE = False

rpvx_unsup = False
force_batchnorm_trainmode = False
force_batchnorm_testmode = False

RANDOMIZE_BG = True

MNIST_CONVLSTM = False
MNIST_CONVLSTM_STOCHASTIC = False

GQN3D_CONVLSTM = False
GQN3D_CONVLSTM_STOCHASTIC = False
LOSS_FN = 'L1'
CONVLSTM_DIM = 128
CONVLSTM_STEPS = 4
gan_scale =10
DEEPMIND_DATA = False
GQN_DATA_NAME = 'shepard_metzler_7_parts' # or rooms_ring_camera

#some stuff related to embeddings
embed_loss_coeff = 1.0
EMBEDDING_LOSS = False
embedding_size = 8
embedding_layers = 4
kl_loss_coeff = 1
single_layer =False
gan_d_iters = 5
# Do not turn on with anything else
LOSS_GAN=False
T0 = time.time()

exp_name = sys.argv[1].strip() if len(sys.argv) >= 2 else ''
load_name = ''
save_name = exp_name

options.data_options('doubledata', 'double', add_suffix = True)
options.data_options('doubledebugdata', 'double_single', add_suffix = False)
options.data_options('4_data', '4', add_suffix = True)
options.data_options('multi_data', 'multi', add_suffix = True)
options.data_options('arith_data', 'arith', add_suffix = True)
options.data_options('house_data', 'house', add_suffix = True)

vis_dir='vis_embeddings_new'

OG('doublemug',
   'doubledata',
   MULTI_UNPROJ = True, lr = 5E-4, BS = 2, valp = 500, H = 64, W = 64, data_dir = "double_tfrs/",
   MINV = 20, MAXV = 20, VDELTA = 20, USE_OUTLINE = False, USE_MESHGRID = False,load_name = "", GQN3D_CONVLSTM=True, EMBEDDING_LOSS=False
)

OG('doublemug_view_pred_only',
   'doubledata',
   MULTI_UNPROJ = True, lr = 5E-4, BS = 2, valp = 500, H = 64, W = 64, data_dir = "double_tfrs/",
   MINV = 20, MAXV = 20, VDELTA = 20, USE_OUTLINE = False, USE_MESHGRID = False,load_name = "", GQN3D_CONVLSTM=True, EMBEDDING_LOSS=False, DIRECT_TREE_LOSS=False, TREE_LOSS=False
)

OG('doublemug_direct_tree_to_image',
   'doubledata',
   MULTI_UNPROJ = True, lr = 5E-4, BS = 2, valp = 500, H = 64, W = 64, data_dir = "double_tfrs/",
   MINV = 20, MAXV = 20, VDELTA = 20, USE_OUTLINE = False, USE_MESHGRID = False,load_name = "", GQN3D_CONVLSTM=True, EMBEDDING_LOSS=False, DIRECT_TREE_LOSS=False, TREE_LOSS=False, DIRECT_TREE_TO_IMG=True
)

OG('doublemug_full_model_dual_loss',
   'doubledata',
   MULTI_UNPROJ = True, lr = 5E-4, BS = 2, valp = 500, H = 64, W = 64, data_dir = "double_tfrs/",
   MINV = 20, MAXV = 20, VDELTA = 20, USE_OUTLINE = False, USE_MESHGRID = False,load_name = "", GQN3D_CONVLSTM=True, EMBEDDING_LOSS=False, DIRECT_TREE_LOSS=True, TREE_LOSS=True
)

OG('doublemug_frozen',
   'doubledata',
   MULTI_UNPROJ = True, lr = 5E-4, BS = 2, valp = 500, H = 64, W = 64, data_dir = "double_tfrs/",
   MINV = 20, MAXV = 20, VDELTA = 20, USE_OUTLINE = False, USE_MESHGRID = False,load_name = "doublemug", GQN3D_CONVLSTM=True, EMBEDDING_LOSS=False, DIRECT_TREE_LOSS=True, TREE_LOSS=True
)



OG('doublemug_test',
   'doubledata',
   MULTI_UNPROJ = True, lr = 5E-4, BS = 1, valp = 500, H = 64, W = 64, data_dir = "double_tfrs/",
   MINV = 20, MAXV = 20, VDELTA = 20, USE_OUTLINE = False, USE_MESHGRID = False,load_name = "doublemug",mode="test",vis_dir = '/projects/katefgroup/mprabhud/vistest_new_4_4',DUMP_TENSOR = False,
   HV = 4, VV = 1
)

OG('doublemug_debug',
   'doublemug', 'doubledebugdata',
   DEBUG_VOXPROJ = True
)

OG('doublemug_train',
   'doublemug',
   valp = 100, savep = 10000, BS = 2
)

OG('doublemug_small',
   'doublemug_train',
   S = 64,
)

OG('doublemug_small_debug',
   'doublemug_small', 'doubledebugdata',
   DEBUG_VOXPROJ = False, DEBUG_UNPROJECT = True, valp = 50, BS = 4,
)

#what is voxproj vs unproj?
# works fine for
# no debug voxproj (single data) +
# debug voxproj? 
# no debug voxproj (all data) +
# debug unproj ??? seems fishy -- not sure if it works
OG('doublemug_small2_debug',
   'doublemug_small_debug',
   S = 32, H = 64, W = 64,
)

OG('doublemug_train_gru',
   'doublemug_train',
   AGGREGATION_METHOD = 'gru', BS = 1
)

#works w/ depth/mask
#works w/o depth/mask
OG('querytask',
   'doublemug_train_gru',
   opname = 'query',
   RANDOMIZE_BG = False, AGGREGATION_METHOD = 'average', BS = 2, lr = 1E-4,
   USE_OUTLINE = False, USE_MESHGRID = False, AUX_POSE = False
)

OG('querytask_debug',
   'querytask', 'doubledebugdata',
   lr = 1E-4
)

OG('size64',
   H = 64, W = 64, BS = 8,
)

#works
OG('querytask_debug64',
   'querytask_debug', 'size64'
)

#not sure if works w/ depth/mask
#doesn't work w/o depth/mask
#wait... this works!
OG('querytask64',
   'querytask', 'size64'
)

OG('querytask_eager',
   'querytask_debug', 
   eager = True, BS = 1, NUM_VIEWS = 2
)

OG('gqnbase',
   'querytask',
   NUM_PREDS = 1, H = 64, W = 64, S = None, savep = 10000, 
)

OG('gqn2d',
   'gqnbase', 
   opname = 'gqn2d', BS = 8, 
)

OG('gqntower',
   'gqnbase',
   opname = 'gqntower',
   BS = 8, CONVLSTM_DIM = 128, CONVLSTM_STEPS = 4, LOSS_FN = 'CE',
)


OG('gqntower2',
   'gqntower', load_name = 'gqntower'
)

OG('gqntower2_eval',
   'gqntower2', load_name = 'gqntower2', mode = 'test', BS = 1,
)


OG('gqntower_debug',
   'gqntower', 'doubledebugdata'
)
   

OG('gqn3d',
   'gqnbase',
   opname = 'gqn3d', BS = 8,savep = 5000
)

OG('gqn3d_ce', 'gqn3d', LOSS_FN='CE')

#converges to 0 quickly
OG('gqn3d_convlstm',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
)

OG('gqn3d_convlstm_stoch',
   'gqn3d_convlstm',
   GQN3D_CONVLSTM_STOCHASTIC = True,
)

OG('gqn3d_convlstm_big',
   'gqn3d_convlstm',
   CONVLSTM_DIM = 256,
   CONVLSTM_STEPS = 6,
)

#pretrain from 4 views
OG('grnn_shapenet_1view',
   'gqn3d_convlstm_big', load_name = 'gqn3d_convlstm_big',
   NUM_VIEWS=1,
)

#does not work
OG('gqn3d_cameratest1', 'gqn3d_convlstm_big', radius = 2.0)
#does not work
OG('gqn3d_cameratest2', 'gqn3d_convlstm_big', radius = 2.0, fov = 60.0)
#this works
OG('gqn3d_cameratest3', 'gqn3d_convlstm_big', fov = 15.0)
#ok what about this?
OG('gqn3d_cameratest4', 'gqn3d_convlstm_big', radius = 3.0)

OG('gqn3d_convlstm_big_eval',
   'gqn3d_convlstm_big', load_name = 'gqn3d_convlstm_big',
   mode = 'test', BS = 1
)

OG('gqn3d_convlstm_4obj_eval',
   'gqn3d_convlstm_big', '4_data',
   load_name = 'gqn3d_convlstm_big',
   mode = 'test', BS = 1, data_dir = '4_tfrs', 
)

OG('gqntower_4obj_eval',
   'gqntower', '4_data',
   load_name = 'gqntower',
   mode = 'test', BS = 1, data_dir = '4_tfrs', 
)

#suncg
OG('gqn3d_suncg_base', 'gqn3d_convlstm_big', 'house_data', load_name = 'gqn3d_convlstm_big', data_dir = 'house_tfrs',
   HV = 8, VV = 3, MINV=20, MAXV=80, fov=30
)

OG('gqn3d_suncg_r4', 'gqn3d_suncg_base', radius = 4)
OG('gqn3d_suncg_r3', 'gqn3d_suncg_base', radius = 3)
OG('gqn3d_suncg_r2', 'gqn3d_suncg_base', radius = 2)
OG('gqn3d_suncg_r1', 'gqn3d_suncg_base', radius = 1)
OG('gqn3d_suncg_r0.5', 'gqn3d_suncg_base', radius = 0.5)

OG('gqn3d_suncg_r2_gru', 'gqn3d_suncg_base', radius = 2, AGGREGATION_METHOD='gru')

OG('gqn3d_suncg_masked', 'gqn3d_suncg_base', radius = 2, MASKED_DATASET = True)

OG('gqn3d_multi', 'gqn3d_convlstm_big', 'multi_data', load_name = 'gqn3d_convlstm_big', data_dir = 'multi_tfrs')


OG('arith', 'gqn3d_multi', 'arith_data',
   data_dir = 'arith_tfrs', ARITH_MODE = True, load_name = 'gqn3d_multi',
   opname = 'gqn3d', BS = 4, #important!!,
   mode = 'test', FIX_VIEW = True,
)

OG('arith2', 'gqntower', 'arith_data',
   data_dir = 'arith_tfrs', ARITH_MODE = True, load_name = 'gqntower',
   opname = 'gqntower', BS = 4, #important!!,
   mode = 'test', FIX_VIEW = True,
)

OG('gqntower_eval',
   'gqntower', load_name = 'gqntower',
   mode = 'test', BS = 1
)

OG('gqn3d_deepmind',
   'gqn3d_convlstm_big',
   DEEPMIND_DATA = True, 
)

OG('gqn3d_rooms',
   'gqn3d_deepmind', GQN_DATA_NAME = 'rooms_ring_camera'
)

OG('gqn3d_rooms2', 'gqn3d_rooms', radius = 2.0, savep=50000)
OG('gqn3d_rooms3', 'gqn3d_rooms', load_name = 'gqn3d_rooms', fov = 20.0, savep=2000)


#also run gqn3d_rooms for a long time, and compare on test set
OG('gqn3d_rooms4', 'gqn3d_rooms', fov = 60.0, radius = 1.2, savep=5000)
OG('gqn3d_rooms6', 'gqn3d_rooms', fov = 60.0, radius = 0.95, savep=5000)
OG('gqn3d_rooms6_resume', 'gqn3d_rooms6', load_name='gqn3d_rooms6')
OG('gqn3d_rooms6_eval', 'gqn3d_rooms6', load_name='gqn3d_rooms6_resume', BS = 1, mode='test')
#note that at a distance of 1 -- the far end of the scene, we have a coverage of siez 1
#we can also try the more typical radius of 2, at the risk of not covering the scene properly

#basicall we run the model for 1000 steps with a lower decay value so that update ops does its job
OG('fixbn', bn_decay = 0.99, savep = 1000)
   
OG('gqn3d_rooms_fixbn', 'gqn3d_rooms6', 'fixbn', load_name='gqn3d_rooms6_resume')

OG('gqn3d_rooms5', 'gqn3d_rooms', fov = 60.0, radius = 2.0, savep=5000)

#.....
OG('gqn3d_rooms6_embed', 'gqn3d_rooms5', EMBEDDING_LOSS = True)
OG('gqn3d_rooms_embed', 'gqn3d_rooms6_embed', load_name = 'gqn3d_rooms6_embed')
OG('gqn3d_rooms_embed2', 'gqn3d_rooms_embed', load_name = 'gqn3d_rooms_embed')

OG('gqn3d_deepmind2',
   'gqn3d_deepmind', load_name = 'gqn3d_deepmind'
)

OG('gqntower_deepmind',
   'gqntower',
   DEEPMIND_DATA = True, 
)

OG('gqntower_room',
   'gqntower_deepmind', GQN_DATA_NAME = 'rooms_ring_camera'
)

OG('gqntower_room4', 'gqntower_room', BS = 32, CONVLSTM_STEPS = 12)
OG('gqntower_room_eval', 'gqntower_room', load_name = 'gqntower_room3', BS = 1, mode = 'test')

OG('gqntower_deepmind2',
   'gqntower_deepmind', load_name = 'gqntower_deepmind',
   DEEPMIND_DATA = True, 
)

OG('gqn3d_deepmind_eval', 'gqn3d_deepmind', load_name = 'gqn3d_deepmind2', mode = 'test', BS = 1)
OG('gqntower_deepmind_eval', 'gqntower_deepmind', load_name = 'gqntower_deepmind2', mode = 'test', BS = 1)

OG('gqntest',
   'gqnbase',
   opname = 'gqntest', DEEPMIND_DATA = True,
)


########

OG('gqn3dv2', 'gqn3d', lr = 5E-4)
OG('gqn3dv3', 'gqn3d', lr = 2E-5)


OG('gqn3d_debug',
   'gqn3d', 'doubledebugdata',
   FIX_VIEW = False
)

OG('mnist',
   opname = 'mnist', BS = 64, valp = 200,
)

OG('mnist_convlstm',
   'mnist',
   MNIST_CONVLSTM = True,
)

OG('mnist_convlstm_stoch',
   'mnist_convlstm',
   MNIST_CONVLSTM_STOCHASTIC = True,
)



OG('gqn3d_convlstm_view_pred_only_highres',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   EMBEDDING_LOSS= False,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   valp=100,
   savep=1000,
   BS=3
)


OG('gqn3d_convlstm_view_pred_only_3d',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   EMBEDDING_LOSS= False,
   EMBEDDING_LOSS_3D = True,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   emb_dim=32,
   valp=100,
   savep=1000,
   BS=3
)

OG('gqn3d_convlstm_view_pred_only_3d_test',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   EMBEDDING_LOSS= False,
   EMBEDDING_LOSS_3D = True,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   emb_dim=32,
   valp=100,
   savep=1000,
   BS=3
)

OG('gqn3d_convlstm_view_pred_only_test',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   EMBEDDING_LOSS= True,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   valp=100,
   savep=1000,
   BS=5
)


OG('gqn3d_convlstm_view_pred_only',
   'gqn3d', #'doubledebugdata',
   load_name="gqn3d_convlstm_view_pred_only",
   mask3d=True,
   valp=1,
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   EMBEDDING_LOSS= False,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   # valp=100,
   savep=5000,
   BS=4
)

OG('gqn3d_direct_tree_to_image',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   DIRECT_TREE_TO_IMG=True
)

OG('gqn3d_full_model_dual_loss',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=True,
   TREE_LOSS=True,
   BS=4,
)

OG('gqn3d_cond_tree_vae',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   COND_TREE_TO_IMG = True,
   GoModules=True,
   BS=3)

OG('gqn3d_cond_tree_vae_concat',
   'gqn3d', #'doubledebugdata',
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_new",
   FREEZE_ENC_DEC= True,
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   COND_TREE_TO_IMG = True,
   # Concat=True,
   GoModules=False,
   BS=3
)

OG('gqn3d_cond_tree_vae_concat_end_kl','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=True,L2_loss_z=True,
   USE_MEAN=True,BS_NORM=True,load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_end_nokl','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=True,L2_loss_z=True,
   USE_MEAN=True,BS_NORM=True,load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",kl_vae_loss=False,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_kl_frozen','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=True,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   load_name="gqn3d_cond_tree_vae_concat_begin_kl_frozen",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   valp=1,
   frozen_decoder=True,
   concat_3d=False,
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_kl_nonfrozen','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=True,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   load_name="gqn3d_cond_tree_vae_concat_begin_kl_nonfrozen",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   valp=1,
   frozen_decoder=False,
   concat_3d=False,
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)
OG('gqn3d_cond_tree_vae_onlycolor','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=True,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   load_name="gqn3d_convlstm_view_pred_only",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   valp=100,
   stop = True,
   onlycolor=True,
   frozen_decoder=False,
   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_onlycolor_inf','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_onlycolor_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   inf_net= True,
   stop = False,
   valp=100,
   onlycolor=True,
   frozen_decoder=False,
   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_inf','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_begin_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   inf_net= True,
   stop = False,
   valp=100,
   concat_3d=True,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_inf_go','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   load_name="gqn3d_cond_tree_vae_concat_begin_inf_go",
   # load_name="gqn3d_convlstm_view_pred_only",
   # load_name="gqn3d_cond_tree_vae_concat_begin_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",

   GoModules=True,
   RANDOMNESS_THRESH=0.8,
   inf_net= True,
   stop = True,
   valp=100,
   concat_3d=True,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_inf_go_notstop','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   load_name="gqn3d_cond_tree_vae_concat_begin_inf_go",
   # load_name="gqn3d_convlstm_view_pred_only",
   # load_name="gqn3d_cond_tree_vae_concat_begin_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   GoModules=True,
   RANDOMNESS_THRESH=0.8,
   inf_net= True,
   stop = False,
   valp=100,
   concat_3d=True,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_inf_go_noadj','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   load_name="gqn3d_cond_tree_vae_concat_begin_inf_go",
   # load_name="gqn3d_convlstm_view_pred_only",
   # load_name="gqn3d_cond_tree_vae_concat_begin_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   GoModules=True,
   RANDOMNESS_THRESH=1.1,
   inf_net= True,
   stop = True,
   valp=100,
   concat_3d=True,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_inf_go_sample','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   load_name="gqn3d_cond_tree_vae_concat_begin_inf_go",

   # load_name="gqn3d_convlstm_view_pred_only",
   # load_name="gqn3d_cond_tree_vae_concat_begin_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   SAMPLE_ANGLES =True,
   GoModules=True,
   single_layer=False,
   randomize=True,
   RANDOMNESS_THRESH=-1,
   customTrees=True,
   inf_net= True,
   SAMPLE_VALUES = True,
   stop = True,
   BS=1,
   mode="test",
   valp=100,
   concat_3d=True,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)


OG('gqn3d_cond_tree_vae_concat_begin_custom_go','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   load_name="gqn3d_cond_tree_vae_concat_begin_custom_go",
   # load_name="gqn3d_convlstm_view_pred_only",
   # load_name="gqn3d_cond_tree_vae_concat_begin_custom",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   RANDOMNESS_THRESH=0.8,
   inf_net= True,
   stop = True,
   custominf=True,
   GoModules=True,
   valp=100,
   concat_3d=True,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)



OG('gqn3d_cond_tree_vae_concat_end_inf','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_end_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   inf_net= True,
   kl_loss_coeff=0.01,
   stop = True,
   RANDOMNESS_THRESH = 0.8,
   valp=100,
   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_end_inf_segmentation','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_end_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   segmentation=True,
   segmentation_stop = True,
   inf_net= True,
   kl_loss_coeff=0.01,
   stop = True,
   RANDOMNESS_THRESH = -1,
   valp=100,
   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_end_inf_segmentation_test','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_end_inf_segmentation",
   loading=False,
   mode="test",
   BS=1,
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   segmentation=True,
   segmentation_stop = True,
   inf_net= True,
   kl_loss_coeff=0.01,
   stop = True,
   RANDOMNESS_THRESH = -1,
   valp=100,
   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)


OG('gqn3d_cond_tree_vae_concat_end_inf_classification','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_end_inf_classification",
   # load_name="gqn3d_cond_tree_vae_concat_end_inf_classification",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   segmentation=True,
   segmentation_stop = True,
   loading=True,
   override_loading = True,
   classification=True,
   # classification_stop=True,
   inf_net= True,
   kl_loss_coeff=0.01,
   stop = True,
   RANDOMNESS_THRESH = -1,
   valp=100,
   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)


OG('gqn3d_cond_tree_vae_concat_end_single_adj','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_end_single_adj",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   single_layer=True,
   inf_net= True,
   kl_loss_coeff=0.01,
   stop = True,
   RANDOMNESS_THRESH = 0.8,
   valp=100,
   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_end_inf_sample','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   mode="test",
   BS=1,
   single_layer=False,
   randomize = True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_end_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   inf_net= True,
   stop = True,
   valp=100,
   concat_3d=False,  
   SAMPLE_ANGLES=False,
   SAMPLE_VALUES = True,
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_inf_sample','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   mode="test",
   BS=1,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_begin_inf",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   inf_net= True,
   stop = True,
   valp=100,
   concat_3d=True,
   SAMPLE_VALUES = True,
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_end_custom_sample','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   mode="test",
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_end_custom",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   single_layer=False,
   randomize = True,

   inf_net= True,
   stop = False,
   valp=100,
   BS=1,
   custominf=True,
   SAMPLE_VALUES = True,
   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_custom_sample','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   mode="test",
   BS=1,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_begin_custom",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   inf_net= True,
   stop = False,
   custominf=True,
   valp=100,
   SAMPLE_VALUES = True,
   concat_3d=True,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_end_custom','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_end_custom",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   inf_net= True,
   RANDOMNESS_THRESH = 0.8,
   kl_loss_coeff=0.01,

   stop = True,
   valp=100,
   custominf=True,

   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)
OG('gqn3d_cond_tree_vae_concat_begin_custom','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=False,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   # load_name="gqn3d_convlstm_view_pred_only",
   load_name="gqn3d_cond_tree_vae_concat_begin_custom",
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   inf_net= True,
   stop = False,
   custominf=True,
   valp=100,
   concat_3d=True,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)




OG('gqn3d_cond_tree_vae_onlycolor_inf_visualize','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=True,L2_loss_z=True,
   # load_name="gqn3d_cond_tree_vae_concat_begin_kl",
   load_name="gqn3d_cond_tree_vae_onlycolor_inf",
   visualize_inf = True,
   # load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   inf_net = True,
   # stop = True,
   valp=100,
   onlycolor=True,
   concat_3d=False,  
   USE_MEAN=True,BS_NORM=True,kl_vae_loss=True,
   save_custom=False)

OG('gqn3d_cond_tree_vae_concat_begin_nokl','gqn3d_cond_tree_vae_concat',
   l2mask=False,FREEZE_ENC_DEC=True,L2_loss_z=True,
   concat_3d=False,
   USE_MEAN=True,BS_NORM=True,load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",kl_vae_loss=False,
   save_custom=False)


OG('gqn3d_cond_tree_vae_concat_pix_norm','gqn3d_cond_tree_vae_concat',
   l2mask=False,BS_NORM=False,load_name="gqn3d_cond_tree_vae_concat_pix_norm")

OG('gqn3d_full_model_dual_loss_3dmask_l2',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   L2LOSS_DIST = True,
   mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=True,
   TREE_LOSS=True,
   renderer = False,
   BS=4,
)

OG('gqn3d_full_model_dual_loss_3dmask_l2_new',
   'gqn3d_full_model_dual_loss_3dmask_l2', #'doubledebugdata',
   )


OG('unset_everything','gqn3d',
   COND_TREE_TO_IMG=False,
   TREE_LOSS=False,
   DIRECT_TREE_LOSS=False,
   SAMPLE_VALUES=False,
   RANK_TENSORS=False
)

OG('gqn3d_full_model_dual_loss_3dmask_l2_z',
   'gqn3d', #'doubledebugdata',
   load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   # savep=1,
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   # L2LOSS_DIST = True,
   L2_loss_z = True,
   mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=True,
   TREE_LOSS=True,
   renderer = False,
   BS=4,
)
OG('gqn3d_full_model_dual_loss_3dmask_l2_z_test',
   'gqn3d', #'doubledebugdata',
   load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   SAMPLE_VALUES=True,
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   mode="test",
   # L2LOSS_DIST = True,
   L2_loss_z = True,
   mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=True,
   TREE_LOSS=True,
   renderer = False,
   BS=1,
)

OG('gqn3d_full_model_dual_loss_3dmask_l2_z_retrieval',
   'gqn3d', #'doubledebugdata',
   load_name="gqn3d_full_model_dual_loss_3dmask_l2_z",
   SAMPLE_VALUES=False,
   RANK_TENSORS = True,
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   mode="test",
   # L2LOSS_DIST = True,
   L2_loss_z = True,
   mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=True,
   TREE_LOSS=True,
   renderer = False,
   BS=5,
)


OG('gqn3d_full_model_dual_loss_3dmask_kl_cond',
   'gqn3d', #'doubledebugdata',
   load_name="gqn3d_full_model_dual_loss_3dmask_l2_new",
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   # L2LOSS_DIST = True,
   DIRECT_TREE_TO_IMG = True,
   KLD_LOSS = True,
   mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=True,
   TREE_LOSS=True,
   renderer = False,
   BS=1,
)


OG('gqn3d_full_model_dual_loss_3dmask_kl',
   'gqn3d', #'doubledebugdata',
   load_name="gqn3d_full_model_dual_loss_3dmask_l2_new",
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   # L2LOSS_DIST = True,
   KLD_LOSS = True,
   mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=True,
   TREE_LOSS=True,
   renderer = False,
   BS=4,
)



OG('gqn3d_full_model_dual_loss_3dmask_l2_nodirect',
   'gqn3d', #'doubledebugdata',
   load_name= "gqn3d_full_model_dual_loss_3dmask_l2_nodirect",
   mode= "train",
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   L2LOSS_DIST = True,
   KLD_LOSS = False,
   mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=False,
   valp=200,
   renderer = False,
   TREE_LOSS=True,
   BS=4,
)
OG('gqn3d_full_model_dual_loss_3dmask_kl_nodirect_kl',
   'gqn3d', #'doubledebugdata',
   load_name= "gqn3d_full_model_dual_loss_3dmask_l2_nodirect",
   mode= "train",
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   L2LOSS_DIST = False,
   KLD_LOSS = True,
   mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=False,
   valp=200,
   renderer = False,
   TREE_LOSS=True,
   BS=4,
)

OG('gqn3d_full_model_dual_loss_3dmask_l2_nodirect_render',
   'gqn3d', #'doubledebugdata',
   load_name= "gqn3d_full_model_dual_loss_3dmask_l2_nodirect_render",
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   L2LOSS_DIST = True,
   mask3d = False,
   l2mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=False,
   renderer = True,
   TREE_LOSS=True,
   BS=4,
)



OG('gqn3d_full_model_dual_loss_3dmask_l2_no_stoch',
   'gqn3d', #'doubledebugdata',
   load_name= "gqn3d_full_model_dual_loss_3dmask_l2_no_stoch",
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   L2LOSS_DIST = True,
   mask3d = True,
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=True,
   TREE_LOSS=True,
   renderer = False,
   stochasticity = False,
   BS=4,
)




OG('gqn3d_direct_tree_to_image_overfit',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   DIRECT_TREE_TO_IMG=True,
   OVERFIT_ON_SUBSET = True
)

OG('gqn3d_direct_tree_to_image_overfit_gt_bbox',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   DIRECT_TREE_TO_IMG=True,
   OVERFIT_ON_SUBSET = True,
   USE_GT_BBOX = True
)

OG('gqn3d_direct_tree_to_image_overfit_gt_bbox_no_poe',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=False,
   TREE_LOSS=False,
   DIRECT_TREE_TO_IMG=True,
   OVERFIT_ON_SUBSET = True,
   USE_GT_BBOX = True
)

OG('gqn3d_frozen',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
   DIRECT_TREE_LOSS=True,
   TREE_LOSS=True,
   load_name='gqn3d_convlstm_view_pred_only',
   BS=4,
   FREEZE_ENC_DEC=True
)




#########

generate_views = False
ELEV_GRANULARITY = 1
AZIMUTH_GRANULARITY = 24*4
MIN_ELEV = 0
MAX_ELEV = 80
GEN_FRAMERATE = 24
# generate all possible output views

#rooms dataset
OG('gqn3d_rooms_gen', 'gqn3d_rooms6_eval', generate_views = True)
OG('gqn3d_rooms_gen_fixbn', 'gqn3d_rooms_fixbn', generate_views = True,
   BS = 1, mode = 'test', load_name = 'gqn3d_rooms_fixbn')

OG('gqntower_rooms_gen', 'gqntower_room_eval', generate_views = True)

#SM shapes
OG('gqn3d_sm_gen', 'gqn3d_deepmind_eval', generate_views = True, ELEV_GRANULARITY = 3, MIN_ELEV = -30, MAX_ELEV = 30)
OG('gqntower_sm_gen', 'gqntower_deepmind_eval', generate_views = True, ELEV_GRANULARITY = 3, MIN_ELEV = -30, MAX_ELEV = 30)

#shapenet shapes
#these dont' seem to work yet...
OG('gqn3d_shapenet_gen', 'gqn3d_convlstm_big_eval', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8)

OG('gqntower_shapenet_gen', 'gqntower2_eval', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8)

#...
OG('gqn3d_shapenet4_gen', 'gqn3d_convlstm_4obj_eval', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8)

OG('gqntower_shapenet4_gen', 'gqntower_4obj_eval', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8)

#suncg gen views
OG('gqn3d_suncg_gen', 'gqn3d_suncg_masked', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8,
   mode='test', BS=1, load_name='gqn3d_suncg_masked')


def _verify_(key, value):
    #print(key, '<-', value)
    print('{0:20} <--  {1}'.format(key, value))
    assert key in globals(), ('%s is new variable' % key)

if exp_name not in options._options_:
    print('*' * 10 + ' WARNING -- no option group active ' + '*' * 10)
else:
    print('running experiment', exp_name)
    for key, value in options.get(exp_name).items():
        _verify_(key, value)
        globals()[key] = value


def set_experiment(exp_name):
    print('running experiment', exp_name)
    for key, value in options.get(exp_name).items():
        _verify_(key, value)
        globals()[key] = value 

#stuff which must be computed afterwards, because it is a function of the constants defined above

# camera stuffs
fx = W / 2.0 * 1.0 / math.tan(fov * math.pi / 180 / 2)
fy = fx
focal_length = fx / (W / 2.0) #NO SCALING

x0 = W / 2.0
y0 = H / 2.0

#scene stuff
near = radius - SCENE_SIZE
far = radius + SCENE_SIZE

#other
GEN_NUM_VIEWS = ELEV_GRANULARITY * AZIMUTH_GRANULARITY
