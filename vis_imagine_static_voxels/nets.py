import tensorflow as tf
from keras import backend as K
import random
import constants as const
import utils
import os.path as path
import numpy as np
import inputs
from tensorflow import summary as summ
from munch import Munch
import ipdb
import math
import csv
# from utils import nets
from lib_classes.modules.output_encoder2d import OutputEncoder_f1
from lib_classes.modules.output3d import Output3D_f2
from lib_classes.modules.input_decoder2d import InputDecoder_f3
# from lib_classes.modules.output_decoder2d import OutputDecoder_f5
from lib_classes.modules.conv_lstm_decoder import ConvLstmDecoder_f4
from lib_classes.modules.embedding_network import embedding_network
import pickle
st = ipdb.set_trace
from lib.modules.DistributionRender import DistributionRender,DistributionRender_Z
from lib.modules.Transform import Transform
from lib.modules.Combine import Combine_Vis,Combine_Pos,Combine_Vis_Z
from lib.modules.Describe import Describe_Vis,Describe_Pos,Describe_Vis_Z
from lib.modules.ConceptMapper import ConceptMapper, ConceptMapper_Cond
from lib.modules.VAE import VAE
from lib.modules.latent_gen import h_mean,h_var
from lib.BiKLD import BiKLD
from lib.reparameterize import reparameterize
from lib.modules.latent_gen import h_mean_cond,h_var_cond
from lib.modules.kld import KLD
from lib.modules.kld import KLD
from lib.modules.Segmentation import FCN
from lib.modules.Classification import Classify

from lib_classes.modules.embnet2 import embnet2
from lib_classes.modules.embnet3 import embnet3
from double_pool import  DoublePool
from lib_classes.modules.FeatNet import FeatNet
from lib.modules.inf_net import InfNet
import evaluate
import random
import glob
class Net(tf.keras.Model):
    def __init__(self, input_):
        super(Net, self).__init__()
        # self.weights_new = {}
        self.num =0 
        self.inputNew = input_
        #this is used to select whether to pull data from train, val, or test set
        self.data_selector = input_.q_ph
        self.output_encoder2d_f1 = OutputEncoder_f1()
        self.output3d_f2 = Output3D_f2()
        self.input_decoder2d_f3 = InputDecoder_f3()
        self.conv_lstm_decoder_f4 = ConvLstmDecoder_f4()
        self.embedding_network = embedding_network()
        if const.LOSS_GAN:
            self.discrim = utils.nets.Discriminator()
        if const.EMBEDDING_LOSS or const.EMBEDDING_LOSS_3D:
            self.embed2d_loss =  embnet2(True)
            self.embed3d_loss =  embnet3(True)
        self.double_pool_tree = DoublePool(10)
        self.double_pool_ricson = DoublePool(10)
        downsample = 1
        im_size = 64
        latentdim = 32
        word_size = 16
        pos_size = [8, 1, 1, 1]
        op=["gPoE", "CAT_gPoE"]
        # op = ['PROD', 'PROD']
        self.im_size = im_size
        self.pos_beta = 1
        self.kl_beta = 5
        self.kl_beta = 2
        self.z_dim = 50
        self.one_gaussian = tf.keras.backend.random_normal(shape=[1,self.z_dim], mean=0., stddev=1.,seed=random.randint(1,100))

        self.loading = False
        self.feat3D = FeatNet()
        if const.EMBEDDING_LOSS or const.EMBEDDING_LOSS_3D:
            self.dictionary = ['brown','cylinder','cube','left-front','yellow','sphere','right','right-front','right-behind','cyan','blue','gray','rubber','purple','metal','left-behind','green','red','left','small','large','cup','inside']
        else:
            self.dictionary = ['brown','cylinder','cube','left-front','yellow','sphere','right','right-front','right-behind','cyan','blue','gray','rubber','purple','metal','left-behind','green','red','left','small','large']

        self.dictionary_colors = ['brown','yellow','cyan','blue','gray','purple','green','red']
        self.dictionary_shapes = ['cylinder','cube','sphere']
        self.dictionary_locations = ['left-front','right-front','right-behind','right','left-behind','left']
        self.dictionary_sizes = ['small','large']
        self.dictionary_textures = ['rubber','metal']
        # color shape location size textures
        self.dictionaries = {'color':self.dictionary_colors,'shape':self.dictionary_shapes,'location':self.dictionary_locations,'size':self.dictionary_sizes,'textures':self.dictionary_textures}
        # not sure the right value for this
        if const.segmentation:
            self.cube = tf.cast(tf.expand_dims(tf.reshape(pickle.load(open("voxel_shapes/real_shapes/cube.p","rb")),[-1]),0),tf.float32)
            self.cylinder = tf.cast(tf.expand_dims(tf.reshape(pickle.load(open("voxel_shapes/real_shapes/cylinder.p","rb")),[-1]),0),tf.float32)
            self.sphere = tf.cast(tf.expand_dims(tf.reshape(pickle.load(open("voxel_shapes/real_shapes/sphere.p","rb")),[-1]),0),tf.float32)
            self.segmentation_target_dict = {"cube":self.cube,"cylinder":self.cylinder,"sphere":self.sphere}
            if const.classification:
                self.classification_target_dict = {"cube":tf.cast([[1,0,0]],tf.float32),"cylinder":tf.cast([[0,1,0]],tf.float32),"sphere":tf.cast([[0,0,1]],tf.float32)}


        self.ds = 2 ** downsample 
        self.lmap_size=im_size //self.ds

        self.word_size = [latentdim, word_size, word_size, word_size]
        self.latent_canvas_size = [1, self.lmap_size, self.lmap_size, self.lmap_size, latentdim]

        self.colors2dict = tf.keras.layers.Dense(21, use_bias=False,name="colors")
        self.shapes2dict = tf.keras.layers.Dense(21, use_bias=False,name="shapes")
        self.locations2dict = tf.keras.layers.Dense(21, use_bias=False,name="locations")
        self.sizes2dict = tf.keras.layers.Dense(21, use_bias=False,name="sizes")
        self.textures2dict = tf.keras.layers.Dense(21, use_bias=False,name="textures")

        self.kld = KLD()

        self.h_mean = h_mean(latentdim)
        self.h_var = h_var(latentdim)

        self.h_mean_cond_vis =  h_mean_cond(latentdim=self.z_dim)
        self.h_var_cond_vis =  h_var_cond(latentdim=self.z_dim)
        self.inf_net = InfNet(32)

        self.inf_net_shape = InfNet(32)
        self.inf_net_text = InfNet(32)
        self.inf_net_size = InfNet(32)
        self.inf_net_color = InfNet(32)

        self.fcn = FCN()
        self.classify = Classify()
        # self.h_mean_cond_pos =  h_mean_cond(latentdim=8)
        # self.h_var_cond_pos =  h_var_cond(latentdim=8)

        # if const.GoModules:
        #     self.vis_dist_cond = ConceptMapper_Cond([i//4 for i in self.word_size],32)
        # else:
        self.vis_dist_cond = ConceptMapper_Cond([i//4 for i in self.word_size],8)
        self.vis_dist_cond_color = ConceptMapper_Cond([i//4 for i in self.word_size],8)
        self.vis_dist_cond_size = ConceptMapper_Cond([i//4 for i in self.word_size],8)
        self.vis_dist_cond_text = ConceptMapper_Cond([i//4 for i in self.word_size],8)
        self.vis_dist_cond_shape = ConceptMapper_Cond([i//4 for i in self.word_size],8)

        self.THRESHOLD = const.RANDOMNESS_THRESH

        self.vis_dist = ConceptMapper(self.word_size)
        self.pos_dist = ConceptMapper(pos_size)

        self.latent_canvas_size = [1, self.lmap_size, self.lmap_size, self.lmap_size, latentdim]

        self.combine_vis_z = Combine_Vis_Z(hiddim_v=latentdim//4, op=op[0])
        self.combine_vis = Combine_Vis(hiddim_v=latentdim, op=op[0])
        self.combine_pos = Combine_Pos(hiddim_p=pos_size[0], op=op[0])
        
        self.box_vae = VAE("box_vae",indim=3, latentdim=pos_size[0])
        self.offset_vae = VAE("offset_vae",indim=6, latentdim=pos_size[0])
        self.sampler = reparameterize()
        
        self.describe_vis = Describe_Vis(hiddim_v=latentdim, op=op[1])
        self.describe_pos = Describe_Pos(hiddim_p=pos_size[0], op=op[1])
        self.describe_vis_z = Describe_Vis_Z(hiddim_v=latentdim//4, op=op[1])

        self.transform = Transform(matrix='default')
        self.renderer_z = DistributionRender_Z(hiddim=latentdim)
        self.renderer = DistributionRender(hiddim=latentdim)
        self.pos_criterion = tf.losses.MeanSquaredError()
        self.kl_beta
        self.transform = utils.tfutil.resize_voxel
            
        self.bikld = BiKLD()
        self.recordfile = open('records.tsv', 'w')
        self.record = csv.writer(self.recordfile, delimiter='\t', lineterminator='\n')
        # self.record.writerow(["one","two"])
        # self.test = open("read.txt","w")
        self.metadatafile = open('metadata.tsv', 'w')
        self.metadata = csv.writer(self.metadatafile, delimiter='\t', lineterminator='\n')

    # def add_weights(self, name):
    #     self.weights[name] = utils.tfutil.current_scope_and_vars()
        # change this
    def optimize(self, fn, non_trainable_weight_dict):
        if const.LOSS_GAN:
            self.dis_optimizer = tf.compat.v1.train.AdamOptimizer(const.lr, const.mom)
            self.gen_optimizer = tf.compat.v1.train.AdamOptimizer(const.lr, const.mom)
            self.opt = utils.tfutil.make_opt_op([self.dis_optimizer, non_trainable_weight_dict, self.gen_optimizer], fn)
        else:
            self.optimizer = tf.optimizers.Adam(const.lr, const.mom)
            self.opt = utils.tfutil.make_opt_op(self.optimizer, non_trainable_weight_dict, fn,self)


    def call(self, index = None,kl_coeff=None,loading=False, non_trainable_weight_dict={}):
        #index is passed to the data_selector, to control which data is used
        self.index = index
        self.kl_coeff = kl_coeff
        self.loading = loading
        if self.loading:
            self.THRESHOLD = -1
        else:
            self.THRESHOLD = const.RANDOMNESS_THRESH
        self.optimize(lambda: self.go_up_to_loss(index), non_trainable_weight_dict)
        self.assemble()

    def go_up_to_loss(self, index = None):
        #should save the loss to self.loss_
        #and also return the loss
        raise NotImplementedError

    def build_vis(self):
        #should save a Munch dictionary of values to self.vis
        raise NotImplementedError
    
    def assemble(self):
        #define all the summaries, visualizations, etc
        if const.LOSS_GAN:
            with tf.compat.v1.name_scope('summary'):
                summ.scalar('gen_loss', self.gen_loss) #just to keep it consistent
                summ.scalar('discrim_loss', self.dis_loss)

            if not const.eager:
                self.summary = tf.compat.v1.summary.merge_all()
            else:
                self.summary = None

            self.evaluator = Munch(gen_loss = self.gen_loss,discrim_loss=self.dis_loss)
            self.build_vis()

            #these are the tensors which will be run for a single train, test, val step
            self.test_run = Munch(evaluator = self.evaluator, vis = self.vis, summary = self.summary)
            self.train_run = Munch(opt = self.opt, summary = self.summary)
            self.val_run = Munch(gen_loss = self.gen_loss,discrim_loss=self.dis_loss, summary = self.summary, vis = self.vis)

        else:
            # with tf.compat.v1.name_scope('summary'):
            #     summ.scalar('loss', self.loss_) #just to keep it consistent

            if not const.eager:
                self.summary = tf.compat.v1.summary.merge_all()
            else:
                self.summary = self.loss_

            self.evaluator = Munch(loss = self.loss_)
            self.build_vis()

            #these are the tensors which will be run for a single train, test, val step
            self.test_run = Munch(evaluator = self.evaluator, vis = self.vis, summary = self.summary)
            self.train_run = Munch(opt = self.opt, summary = self.summary)
            self.val_run = Munch(loss = self.loss_, summary = self.summary, vis = self.vis)

    
class TestInput:
    def __init__(self, inputs):
        self.weights = {}
        self.data_selector = inputs.q_ph

        def foo(x):
            total = 0
            if isinstance(x, dict):
                for val in x.values():
                    total += foo(val)
            elif isinstance(x, tuple):
                for y in x:
                    total += foo(y)
            else:
                total += tf.reduce_sum(input_tensor=x)
            return total

        bar = foo(data)
        
        self.test_run = Munch(bar = bar)
        self.val_run = Munch(bar = bar)
        self.train_run = Munch(bar = bar)

    def go(self):
        pass

        
class MnistAE(Net):

    def go_up_to_loss(self, index = None):
        self.prepare_data(index)

        if const.MNIST_CONVLSTM:
            net_fn = utils.nets.MnistAEconvlstm
        else:
            net_fn = utils.nets.MnistAE
            
        self.pred = net_fn(self.img)

        self.loss_ = utils.losses.binary_ce_loss(self.pred, self.img)
        
        if const.MNIST_CONVLSTM_STOCHASTIC:
            self.loss_ += self.pred.loss #kl
        
        return self.loss_
        
    def prepare_data(self, index):
        data = self.inputNew.data(index)
        s.update(dataelf.__dict__) #so we can do self.images, etc.
        
    def build_vis(self):
        self.vis = {
            'in': self.img,
            'out': self.pred
        }

class MultiViewNet(Net):

    def go_up_to_loss(self, index = None):
        # awkward, but necessary in order to record gradients using tf.eager
        
        self.prepare_data(index)
        self.pred_aux_input = self.construct_pred_aux_input()
        self.pred_main_input = self.construct_pred_main_input()
        self.predict()
        self.reproject()
        self.loss()
        return self.loss_

    def prepare_data(self, index):
        data = self.inputNew.data(index)
        self.__dict__.update(data) #so we can do self.images, etc.
        
        #for convenience:
        self.phis_oh = [tf.one_hot(phi, depth = const.VV) for phi in self.phis]
        self.thetas_oh = [tf.one_hot(theta, depth = const.HV) for theta in self.thetas]

    def predict(self):
        with tf.compat.v1.variable_scope('3DED'):
            self.predict_(self.pred_main_input, self.pred_aux_input)

    def get_views_for_prediction(self):
        #order *must* be mask, depth, norms, else this will fail!
        # subtract 4 from the depth so that it is (-1,1)
        # (the shapenet data is all centered with camera at distance of 4)
        self.depths_ = [d-const.radius for d in self.depths]
        return [
            self.masks[:const.NUM_VIEWS],
            self.depths_[:const.NUM_VIEWS],
            self.images[:const.NUM_VIEWS],
        ]
            
    def construct_pred_main_input(self):
        pred_inputs = self.get_views_for_prediction()
        
        pred_inputs = list(zip(*pred_inputs))
        pred_inputs = [tf.concat(x, axis = 3) for x in pred_inputs]

        pred_inputs = self.preunprojection(pred_inputs)

        def stack_unproject_unstack(_pred_inputs):
            _pred_inputs = tf.stack(_pred_inputs, axis = 0)
            _pred_inputs = tf.map_fn(
                lambda x: utils.nets.unproject(x, self.__class__.resize_unproject),
                _pred_inputs, parallel_iterations = 1
            )
            _pred_inputs = tf.unstack(_pred_inputs, axis = 0)
            return _pred_inputs
        
        if isinstance(pred_inputs[0], list):
            rval = [stack_unproject_unstack(inp) for inp in pred_inputs]
        else:
            rval = stack_unproject_unstack(pred_inputs)
        return rval

    def preunprojection(self, pred_inputs):
        return pred_inputs

    def construct_pred_aux_input(self):
        self.poses = [
            tf.concat([phi, theta], axis = 1) for phi, theta in
            zip(self.phis_oh[:const.NUM_VIEWS], self.thetas_oh[:const.NUM_VIEWS])
        ]
        if const.MULTI_UNPROJ:
            return tf.concat(self.poses, axis = 1)
        else:
            return self.poses[0]

    def aggregate_inputs(self, inputs):
        if const.AGGREGATION_METHOD == 'stack':
            return tf.concat(inputs, axis = -1)
        elif const.AGGREGATION_METHOD == 'gru':
            return utils.nets.gru_aggregator(inputs)
        else:
            raise Exception('unknown aggregation method')

    def bin2theta(self, bin):
        return tf.cast(bin, tf.float32) * const.HDELTA + const.MINH

    def bin2phi(self, bin):
        return tf.cast(bin, tf.float32) * const.VDELTA + const.MINV
    
    def i2theta(self, idx):
        return self.bin2theta(self.thetas[idx])

    def i2phi(self, idx):
        return self.bin2phi(self.phis[idx])

    def translate_views_single(self, vid1, vid2, vox):
        return self.translate_views_multi([vid1], [vid2], [vox])[0]

    def translate_views_multi(self, vid1s, vid2s, voxs):
        """
        # num_views x batch x D X H X W X C
        rotates the 5d tensor `vox` from one viewpoint to another
        vid1s: indices of the bin corresponding to the input view
        vid2s: indices of the bin corresponding to the output view
        """
        # num_views x 8
        dthetas = [self.i2theta(vid2) - self.i2theta(vid1) for (vid2, vid1) in zip(vid2s, vid1s)]
        phi1s = list(map(self.i2phi, vid1s))
        phi2s = list(map(self.i2phi, vid2s))

        dthetas = tf.stack(dthetas, 0)
        phi1s = tf.stack(phi1s, 0)
        phi2s = tf.stack(phi2s, 0)
        
        voxs = tf.stack(voxs, 0)
        
        f = lambda x: utils.voxel.translate_given_angles(*x)
        out = tf.map_fn(f, [dthetas, phi1s, phi2s, voxs], dtype = tf.float32)
        return tf.unstack(out, axis = 0)
    
class MultiViewReconstructionNet(MultiViewNet):

    resize_unproject = True

    def predict_(self, pred_main_inputs, pred_aux_input):
        
        pred_main_inputs_ = [pred_main_inputs[0]]

        pred_main_inputs__ = self.translate_views_multi(
            list(range(1, const.NUM_VIEWS)),
            [0]*(const.NUM_VIEWS-1),
            pred_main_inputs[1:]
        )

        pred_main_inputs_.extend(pred_main_inputs__)
        #[8x32x32x32x32, 8x16x16x16x64, 8x8x8x8x12, 8x4x4x4x256]
        pred_main_input = self.aggregate_inputs(pred_main_inputs_)

        net_out = utils.nets.voxel_net_3d(
            pred_main_input, aux = pred_aux_input, outsize = const.S, d0 = 16
        )
        
        self.pred_voxel = net_out.pred
        self.pred_logit = net_out.logit
        self.pred_features = net_out.features

        if const.DEBUG_UNPROJECT:
            #visualize raw outlines...
            if const.MULTI_UNPROJ:
                self.pred_voxel = tf.reduce_max(input_tensor=pred_main_input, axis = 4, keepdims = True)
            else:
                self.pred_voxel = pred_main_input

        if const.FAKE_NET:
            self.pred_voxel = tf.nn.tanh(tf.Variable(
                np.zeros(self.pred_voxel.get_shape()),
            dtype = tf.float32)) * 0.5 + 0.5


    def flatten(self, voxels):
        pred_depth = utils.voxel.voxel2depth_aligned(voxels)
        pred_mask = utils.voxel.voxel2mask_aligned(voxels)
        
        #replace bg with grey
        hard_mask = tf.cast(pred_mask > 0.5, tf.float32)
        pred_depth *= hard_mask
        pred_depth += const.radius * (1.0 - hard_mask)

        pred_depth = tf.image.resize(pred_depth, (const.H, const.W))
        pred_mask = tf.image.resize(pred_mask, (const.H, const.W))
        return pred_depth, pred_mask


    def reproject(self): #this part is unrelated to the consistency...
        
        def rot_mat_for_angles_(invert_rot = False):
            return utils.voxel.get_transform_matrix_tf(
                theta = self.i2theta(0), 
                phi = self.i2phi(0), 
                invert_rot = invert_rot
            )

        world2cam_rot_mat = rot_mat_for_angles_()
        cam2world_rot_mat = rot_mat_for_angles_(invert_rot = True)
        
        #let's compute the oriented gt_voxel
        gt_voxel = tf.expand_dims(self.voxel, axis = 4)
        gt_voxel = utils.voxel.transformer_preprocess(gt_voxel)
        #simply rotate, but don't project!
        gt_voxel = utils.voxel.rotate_voxel(gt_voxel, world2cam_rot_mat)
        
        self.gt_voxel = gt_voxel

        #used later
        obj1 = tf.expand_dims(self.obj1, axis = 4)
        obj1 = utils.voxel.transformer_preprocess(obj1)
        self.obj1 = utils.voxel.rotate_voxel(obj1, world2cam_rot_mat)
        obj2 = tf.expand_dims(self.obj2, axis = 4)
        obj2 = utils.voxel.transformer_preprocess(obj2)
        self.obj2 = utils.voxel.rotate_voxel(obj2, world2cam_rot_mat)
        
        if not const.DEBUG_VOXPROJ:
            voxels_to_reproject = self.pred_voxel
        else:
            voxels_to_reproject = self.gt_voxel

        to_be_projected_and_postprocessed = [voxels_to_reproject]

        to_be_projected_and_postprocessed.extend(
            self.translate_views_multi(
                [0] * (const.NUM_VIEWS + const.NUM_PREDS - 1),
                list(range(1, const.NUM_VIEWS + const.NUM_PREDS)),
                tf.tile(
                    tf.expand_dims(voxels_to_reproject, axis = 0),
                    [const.NUM_VIEWS + const.NUM_PREDS - 1, 1, 1, 1, 1, 1]
                )
            )
        )
        projected_voxels = tf.map_fn(
            utils.voxel.project_and_postprocess,
            tf.stack(to_be_projected_and_postprocessed, axis = 0),
            parallel_iterations = 1
        )
        projected_voxels = tf.unstack(projected_voxels)


        self.unoriented = utils.voxel.rotate_voxel(voxels_to_reproject, cam2world_rot_mat)

        self.pred_depths, self.pred_masks = list(zip(*list(map(self.flatten, projected_voxels))))
        self.projected_voxels = projected_voxels

        
    def loss(self):

        if const.S == 64:
            self.gt_voxel = utils.tfutil.pool3d(self.gt_voxel)
        elif const.S == 32:
            self.gt_voxel = utils.tfutil.pool3d(utils.tfutil.pool3d(self.gt_voxel))
        
        loss = utils.losses.binary_ce_loss(self.pred_voxel, self.gt_voxel)
        if const.DEBUG_LOSSES:
            loss = utils.tfpy.print_val(loss, 'ce_loss')
        
        if const.DEBUG_VOXPROJ or const.DEBUG_UNPROJECT:
            z = tf.Variable(0.0)
            loss = z-z
        
        self.loss_ = loss

    def build_vis(self):
        self.vis = Munch(
            images = tf.concat(self.images, axis = 2),
            depths = tf.concat(self.depths, axis = 2),
            masks = tf.concat(self.masks, axis = 2),

            pred_masks = tf.concat(self.pred_masks, axis = 2),
            pred_depths = tf.concat(self.pred_depths, axis = 2),
            pred_vox = self.unoriented,
        )

        if hasattr(self, 'seg_obj1'):
            self.vis.seg_obj1 = self.seg_obj1
            self.vis.seg_obj2 = self.seg_obj2


class MultiViewQueryNet(MultiViewNet):

    resize_unproject = False

    def get_views_for_prediction(self):
        #no depth, mask, or seg
        return [self.images[:const.NUM_VIEWS]]
    
    def preunprojection(self, pred_inputs):
        with tf.compat.v1.variable_scope('2Dencoder'):
            return utils.tfutil.concat_apply_split(
                pred_inputs,
                utils.nets.encoder2D
            )
    
    def predict_(self, pred_main_inputs, pred_aux_input):
        pred_main_inputs_ = [
            self.translate_views_multi(
                list(range(0, const.NUM_VIEWS)),
                [0]*(const.NUM_VIEWS),
                x,
            )
            for x in pred_main_inputs
        ]

        pred_main_input = self.aggregate_inputs(pred_main_inputs_)

        # 8x 4x 4 x4 x 256, 8x8x8x8x128, 8x16x16x16x64, 8x32x32x32x32
        assert pred_aux_input is None
        self.feature_tensors = utils.nets.encoder_decoder3D(
            pred_main_input, aux = pred_aux_input,
        )

    def construct_pred_aux_input(self):
        return None

    def aggregate_inputs(self, inputs):
        assert const.AGGREGATION_METHOD == 'average'
        
        n = 1.0/float(len(inputs[0]))
        return [sum(input)*n for input in inputs]

    def reproject(self):
        # rotate scene to desired view point
        oriented_features = [
            self.translate_views_multi(
                [0] * const.NUM_PREDS,
                list(range(const.NUM_VIEWS, const.NUM_VIEWS + const.NUM_PREDS)),
                tf.tile(
                    tf.expand_dims(feature, axis = 0),
                    [const.NUM_PREDS, 1, 1, 1, 1, 1]
                )
            )
            for feature in self.feature_tensors
        ]
        # a NxM list of lists, where N is the number of feature scales and M is
        # the number of target views
        #concatenating before projection mysteriously fails???
        oriented_features = [
            utils.voxel.transformer_postprocess(
                tf.concat(
                    [
                        utils.voxel.project_voxel(feature)
                        for feature in features
                    ],
                    axis = 0
                )
            )
            for features in oriented_features
        ]

        with tf.compat.v1.variable_scope('2Ddecoder'):
            pred_views = utils.nets.decoder2D(oriented_features)
        self.pred_views = tf.split(pred_views, const.NUM_PREDS, axis = 0)
        
        
    def build_vis(self):
        self.vis = Munch(
            input_views = tf.concat(self.images[:const.NUM_VIEWS], axis = 2),
            query_views = tf.concat(self.queried_views, axis = 2),
            pred_views = tf.concat(self.pred_views, axis = 2),
            dump = {'occ': utils.nets.foo} if utils.nets.foo else {}
        )
        
    def loss(self):
        
        self.queried_views = self.images[const.NUM_VIEWS:]

        loss = sum(utils.losses.l2loss(pred, query, strict = True)
                   for (pred, query) in zip(self.pred_views, self.queried_views))
        loss /= const.NUM_PREDS

        #z = tf.Variable(0.0)
        #loss += z-z
        
        if const.DEBUG_LOSSES:
            loss = utils.tfpy.print_val(loss, 'l2_loss')
        self.loss_ = loss

class GQNBase(Net):

    def go_up_to_loss(self, index = None):
        self.setup_data(index)
        with tf.compat.v1.variable_scope('main'):
            self.predict()
            # self.add_weights('main_weights')
        self.loss()
        if const.LOSS_GAN:
            return [self.dis_loss,self.gen_loss]
        else:
            return self.loss_

    def get_tree(self,file_names):
        if const.customTrees:
            custom_trees = glob.glob("trees_67obj/*.p")
            index = random.randint(0,len(custom_trees)-1)

        np_filename = file_names.numpy()
        trees = []
        for i in range(len(file_names)):
            if ((const.mode=="test" and tf.math.equal(self.index, 0) or const.LOAD_VAL) and not const.COND_TREE_TO_IMG):
                tree = pickle.load(open("dummy.tree","rb"))
            else:
                if const.customTrees:
                    fileNameDecoded = custom_trees[index]
                    tree = pickle.load(open(fileNameDecoded,"rb"))
                    tree.parent = None
                else:   
                    fileNameDecoded = np_filename[i].decode()
                    tree = pickle.load(open(fileNameDecoded[3:],"rb"))
                    # remove later
                    # tree = pickle.load(open("try_trees/sphere_rubber_red_large.p","rb")) #cylinder 31, 32, 30
                    # # tree.bbox[3:] = 
                    # tree = pickle.load(open("try_trees/sphere_rubber_large_red.p","rb")) #sphere 28, 32, 29
                    # tree.bbox[3:] =[ 8,8,8]
                # st()
                if const.single_layer and not self.loading:
                    if tree.function == "describe":
                       # newtree = copy.deepcopy(tree)
                       # newtree2 = copy.deepcopy(tree)

                    #    size = random.choice(self.dictionary_colors)
                    #    newtree2.word = size
                    #    newtree2.function = "combine"
                    #    newtree2.num_children = 0
                    #    newtree2.children = []

                    #    size = random.choice(self.dictionary_sizes)
                    #    newtree.word = size
                    #    newtree.num_children = 1
                    #    newtree.function = "combine"
                       # newtree.children = []

                       tree.children = []
                       tree.num_children = len(tree.children)
                       tree.wordVal = tree.word
                   # tree.wordVal = tree.word + " " + newtree.word + " " + newtree2.word
                tree.word ="sphere"
                trees.append(tree)
        return trees

    def setup_data(self,index):

        # if const.mode == "train":
        #     data = self.inputNew.data(index=0)
        # elif const.mode == "valid":
        #     data = self.inputNew.data(index=1)
        # elif const.mode == "test":
        #     data = self.inputNew.data(index=2)
        # st()
        data = self.inputNew.data(index=index)

        trees = self.get_tree(data["treefile_path"])
        data["trees"] = trees
        # st()
        self.__dict__.update(data)
        phis, thetas = zip(*[
            tf.unstack(cam, axis = 1)
            for cam in self.query.context.cameras
        ])

        #convert to degrees
        self.thetas = list(map(utils.utils.degrees, thetas))
        self.phis = list(map(utils.utils.degrees, phis))

        query_phi, query_theta = tf.unstack(self.query.query_camera, axis = 1)
        self.query_theta = utils.utils.degrees(query_theta)
        self.query_phi = utils.utils.degrees(query_phi)

    def predict(self):
        raise NotImplementedError

    def loss(self):
        self.sentences = tf.convert_to_tensor("")
        if const.DIRECT_TREE_TO_IMG or (const.SAMPLE_VALUES and tf.math.equal(self.index,2)):

            if const.LOSS_FN == 'L1':
                loss = utils.losses.l1loss(self.pred_view_prior, self.target, self.target_mask)
            elif const.LOSS_FN == 'CE':
                loss = utils.losses.binary_ce_loss(self.pred_view_prior, self.target, self.target_mask)
            if not tf.math.equal(self.index,2):
                pos_loss = self.pos_loss/ self._total(self.pred_view_prior)
                self.loss_ = loss + (self.pos_beta * pos_loss)
                print('Total: {}\t Recon: {}\t Pos: {}'.format(self.loss_, loss, pos_loss))
            else:
                print('Total: {}\t '.format(loss))


        # elif const.FREEZE_ENC_DEC:
        #     loss = 0.0
        #     if const.TREE_LOSS:
        #         self.kld_loss = self.kl_coeff * self.bikld([self.latent_mean, self.latent_var], [self.prior_mean, self.prior_var]) + (1 - self.kl_coeff) * self.bikld([tf.stop_gradient(self.latent_mean), tf.stop_gradient(self.latent_var)], [self.prior_mean,self.prior_var])
        #         kld_loss, pos_loss = self.kld_loss / self._total(self.pred_view), self.pos_loss/ self._total(self.pred_view)
        #         kld_loss = tf.clip_by_value(kld_loss, 0.0, 500.0)

        #         self.loss_ = self.kl_beta * kld_loss + self.pos_beta * pos_loss

        #         if const.DIRECT_TREE_LOSS:

        #             if const.LOSS_FN == 'L1':
        #                 loss_prior = utils.losses.l1loss(self.pred_view_prior, self.target, self.target_mask)
        #             elif const.LOSS_FN == 'CE':
        #                 loss_prior = utils.losses.binary_ce_loss(self.pred_view_prior, self.target, self.target_mask)

        #             self.loss_ += loss_prior
        #             print('Total: {}\t Recon: {}\t Recon Prior: {}\t KL: {}\t Pos: {}'.format(self.loss_, loss, loss_prior, kld_loss, pos_loss))
        #         else:
        #             print('Total: {}\t Recon: {}\t KL: {}\t Pos: {}'.format(self.loss_, loss, kld_loss, pos_loss))
        elif (const.segmentation and not self.loading) or (const.segmentation and const.override_loading):
            if not const.classification:
                self.loss_ = utils.losses.binary_ce_loss(self.segmentation_masks,self.segmentation_target_masks)
            else:
                self.loss_ = utils.losses.binary_ce_loss(self.segmentation_masks,self.segmentation_target_masks)
                # self.loss_ = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.segmentation_target_masks,self.segmentation_masks))
                # print("predicted",self.segmentation_masks,"target",self.segmentation_target_masks)
            print('Total segmentation loss: {}\t'.format(self.loss_))

        elif const.RANK_TENSORS  and tf.math.equal(self.index,2):
            recalls = [1, 5, 10]
            pool_size = 9
            if self.double_pool_ricson.pool_size == self.double_pool_ricson.num:
                st()
                import pickle
                # sentence = [i.wordVal for i in self.trees]
                # pickle.dump(self.latent_mean,open("latent_mean_{}.p".format(self.num),"wb"))
                # pickle.dump(sentence,open("sentence_{}.p".format(self.num),"wb"))
                # self.num = self.num + 1
                # self.sentences = tf.convert_to_tensor(self.double_pool_tree.fetch_sentence())
                ranking,self.ranking_view,self.sentences= evaluate.compute_precision(self.double_pool_tree.fetch(),self.double_pool_ricson.fetch(),self.double_pool_tree.fetch_sentence(),recalls=recalls,pool_size=pool_size)
                # st()
                print(ranking,"ranking")

        elif const.COND_TREE_TO_IMG:
            if const.LOSS_FN == 'L1':
                loss = utils.losses.l1loss(self.pred_view, self.target, self.target_mask)
            elif const.LOSS_FN == 'CE':
                # st()
                if const.l2mask:
                    loss = utils.losses.binary_ce_loss(self.pred_view, self.target, self.target_mask) 
                else:
                    loss = utils.losses.binary_ce_loss(self.pred_view, self.target) 
            if const.BS_NORM:
                self.vis_kl_loss = self.vis_kl_loss/ const.BS
            else:
                self.vis_kl_loss = self.vis_kl_loss/self._total(self.pred_view)
            if self.pos_loss:
                pos_loss = self.pos_loss/ self._total(self.pred_view)
                self.loss_ = loss + (self.pos_beta * pos_loss) + self.vis_kl_loss
            else:
                if const.kl_vae_loss:
                    self.loss_ = loss + const.kl_loss_coeff*self.vis_kl_loss
                else:
                    self.loss_ = loss
                if const.L2_loss_z:
                    l2_loss = utils.losses.l2loss(self.z_vis, self.outputs3D,self.mask3d)
                    print('L2 Loss: {}\t'.format(l2_loss))
                    self.loss_ = self.loss_ + l2_loss

            if self.pos_loss:
                print('Total: {}\t Recon: {}\t Pos: {}\t Kl: {}'.format(self.loss_, loss, pos_loss, self.vis_kl_loss))
            else:
                print('Total: {}\t Recon: {}\t Kl: {}'.format(self.loss_, loss,self.vis_kl_loss))


        else:
            if not const.EMBEDDING_LOSS_3D:
                if const.LOSS_FN == 'L1':
                    loss = utils.losses.l1loss(self.pred_view, self.target, self.target_mask)
                elif const.LOSS_FN == 'CE':
                    loss = utils.losses.binary_ce_loss(self.pred_view, self.target, self.target_mask)

                if const.DEBUG_LOSSES:
                    loss = utils.tfpy.print_val(loss, 'recon-loss')
                    
                if const.GQN3D_CONVLSTM_STOCHASTIC:
                    loss += utils.tfpy.print_val(self.pred_view.loss, 'kl-loss')

                if const.EMBEDDING_LOSS:
                    # st()
                    emb_total_loss,rgb,emb,emb_pred = self.embed2d_loss(self.target, self.embed3d)
                    B, H, W, C = emb_pred.shape.as_list()
                    self.emb_pca, self.emb_pred_pca = self.embed2d_loss.emb_vis(rgb, emb, emb_pred, tf.ones([B,H,W,1]))
                    # st()
                    print('EMB Loss: {}\t'.format(emb_total_loss))
                    loss +=const.embed_loss_coeff * emb_total_loss
            else:
                # st()
                loss = self.embed3d_loss(self.emb3D_p[0],self.emb_3D_g)
                print('EMB Loss: {}\t'.format(loss))
            
            self.loss_ = loss

            if const.TREE_LOSS:
                pos_loss =  self.pos_loss/ self._total(self.pred_view)
                if const.KLD_LOSS:
                    self.kld_loss = self.kl_coeff * self.bikld([self.latent_mean, self.latent_var], [self.prior_mean, self.prior_var]) + (1 - self.kl_coeff) * self.bikld([tf.stop_gradient(self.latent_mean), tf.stop_gradient(self.latent_var)], [self.prior_mean,self.prior_var])
                    self.kld_loss = self.kld_loss / self._total(self.pred_view)
                    self.loss_ += self.kl_beta * self.kld_loss + self.pos_beta * pos_loss
                    # kld_loss = tf.clip_by_value(kld_loss, 0.0, 500.0)
                elif const.L2LOSS_DIST:
                    # st()
                    if const.l2mask3d:
                        l2_mask3d = (self.mask3d + const.MASKWEIGHT)/(1 + const.MASKWEIGHT)
                        meanL2 = utils.losses.l2loss(self.latent_mean,self.prior_mean,mask=l2_mask3d) 
                        varL2 = utils.losses.l2loss(self.latent_var,self.prior_var,mask = l2_mask3d)
                    else:
                        meanL2 = utils.losses.l2loss(self.latent_mean,self.prior_mean)
                        varL2 = utils.losses.l2loss(self.latent_var,self.prior_var)               
                    self.L2_loss = meanL2 + varL2
                    self.loss_ += self.L2_loss + self.pos_beta * pos_loss
                 
                elif const.L2_loss_z:
                    self.L2_loss_z = utils.losses.l2loss(self.z_latent,self.z_prior)
                    self.loss_ += self.L2_loss_z + self.pos_beta * pos_loss




                # rec_loss, kld_loss, pos_loss = tf.reduce_sum(self.loss_)/ self._total(self.pred_view) , tf.reduce_sum(self.kld_loss) / self._total(self.pred_view), tf.reduce_sum(self.pos_loss)/ self._total(self.pred_view)
                # rec_loss, kld_loss, pos_loss = self.loss_ , tf.reduce_sum(self.kld_loss) / self._total(self.pred_view), tf.reduce_sum(self.pos_loss)/ self._total(self.pred_view)
                # self.loss_ += self.pos_beta * pos_loss
                # st()
                if const.DIRECT_TREE_LOSS:

                    if const.LOSS_FN == 'L1':
                        loss_prior = utils.losses.l1loss(self.pred_view_prior, self.target, self.target_mask)
                    elif const.LOSS_FN == 'CE':
                        loss_prior = utils.losses.binary_ce_loss(self.pred_view_prior, self.target, self.target_mask)

                    self.loss_ += loss_prior
                    if const.KLD_LOSS:
                        print('Total: {}\t Recon: {}\t Recon Prior: {}\t KL: {}\t Pos: {}'.format(self.loss_, loss, loss_prior, self.kld_loss, pos_loss))
                    elif const.L2LOSS_DIST:
                        # st()
                        print('Total: {}\t Recon: {}\t Recon Prior: {}\t L2: {}\t Pos: {}'.format(self.loss_, loss, loss_prior, self.L2_loss, pos_loss))
                    elif const.L2_loss_z:
                        print('Total: {}\t Recon: {}\t Recon Prior: {}\t L2_z: {}\t Pos: {}'.format(self.loss_, loss, loss_prior, self.L2_loss_z, pos_loss))

                else:
                    if const.KLD_LOSS:
                        print('Total: {}\t Recon: {}\t KL: {}\t Pos: {}'.format(self.loss_, loss, self.kld_loss, pos_loss))
                    elif const.L2LOSS_DIST:
                        print('Total: {}\t Recon: {}\t L2: {}\t Pos: {}'.format(self.loss_, loss,  self.L2_loss, pos_loss))
                    elif const.L2_loss_z:
                        print('Total: {}\t Recon: {}\t Recon Prior: {}\t L2_z: {}\t Pos: {}'.format(self.loss_, loss, loss_prior, self.L2_loss_z, pos_loss))

            if const.LOSS_GAN:

                self.g_loss = tf.reduce_mean(input_tensor=self.disc_fake)
                self.d_loss = tf.reduce_mean(input_tensor=self.disc_real) - tf.reduce_mean(input_tensor=self.disc_fake)

                ddx = tf.gradients(ys=self.d_hat, xs=self.x_hat)[0]
                # print(ddx.get_shape().as_list())
                ddx = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(ddx), axis=1))
                ddx = tf.reduce_mean(input_tensor=tf.square(ddx - 1.0) * const.gan_scale)

                self.d_loss = self.d_loss + ddx

                g_loss = loss + 0.05*self.g_loss
                d_loss = self.d_loss

                if const.DEBUG_LOSSES:
                    g_loss = utils.tfpy.print_val(g_loss, 'gen-loss')
                    d_loss = utils.tfpy.print_val(d_loss, 'dis-loss')
                    
                self.dis_loss = d_loss
                self.gen_loss = g_loss



    def build_vis(self):
        # st()
        if (self.double_pool_tree.pool_size == self.double_pool_tree.num and const.RANK_TENSORS) or (not const.EMBEDDING_LOSS_3D and not const.RANK_TENSORS ):
            if const.DIRECT_TREE_TO_IMG or (const.SAMPLE_VALUES and tf.math.equal(self.index,2)):
                self.vis = Munch(
                    input_views = tf.concat(self.query.context.frames, axis = 2),
                    query_views = self.target,
                    file_name=self.file_name
                )
            elif const.RANK_TENSORS:
                self.vis = Munch(
                    input_views = tf.concat(self.query.context.frames, axis = 2),
                    query_views = self.target,
                    pred_views = self.pred_view,
                    file_name=self.file_name,
                    ranking=tf.expand_dims(tf.convert_to_tensor(self.ranking_view),axis=0),
                    # sentence = self.sentences
                )
            elif const.segmentation  and tf.math.equal(self.index,2):
                assert const.BS == 1
                self.vis=Munch(file_name=self.file_name,segmentation_masks= tf.expand_dims(tf.squeeze(self.segmentation_masks),0),sentence=tf.convert_to_tensor(self.trees[0].wordVal)\
                    ,cam_loc=tf.convert_to_tensor(str(self.query.query_camera.numpy())))

            elif const.segmentation:
                self.vis=Munch(file_name=self.file_name)


            else:
                self.vis = Munch(
                                input_views = tf.concat(self.query.context.frames, axis = 2),
                                query_views = self.target,
                                pred_views = self.pred_view,
                                file_name=self.file_name,
                            )                

            if (const.TREE_LOSS and const.DIRECT_TREE_LOSS) or const.DIRECT_TREE_TO_IMG:
                self.vis['pred_view_prior'] = self.pred_view_prior
            if const.SAMPLE_VALUES and tf.math.equal(self.index,2):
                assert const.BS == 1
                # st()
                # first_five = tf.concat(self.samples[:5],axis=2)
                # second_five = tf.concat(self.samples[5:],axis=2)
                # st()
                self.vis['cam_loc'] =  tf.convert_to_tensor(str(self.query.query_camera.numpy()))
                self.vis['samples'] = tf.expand_dims(tf.concat(self.samples,0),0)
                self.vis['sentence'] = tf.convert_to_tensor(self.trees[0].wordVal)
            if const.ARITH_MODE:
                #actually i want the views on a vertical axis
                #and batch on the horizontal axis!
                if False:
                    input_views = tf.concat(self.query.context.frames, axis = 2)
                    input_views = tf.concat(tf.unstack(input_views), axis = 0)
                    input_views = tf.expand_dims(input_views, axis = 0)
                    input_views = tf.tile(input_views, (const.BS, 1, 1, 1))

                else: #for figure making
                    input_views = self.query.context.frames[0]
                    #i don't actually want the last view
                    input_views = tf.unstack(input_views)
                    input_views = [input_views[2], input_views[0], input_views[1]]
                    input_views = tf.concat(input_views, axis = 1)
                    input_views = tf.expand_dims(input_views, axis = 0)
                    input_views = tf.tile(input_views, (const.BS, 1, 1, 1))

                query_views = tf.expand_dims(self.target[3], axis = 0) #view 3!!!
                query_views = tf.tile(query_views, (const.BS, 1, 1, 1))
                
                self.vis = Munch(
                    input_views = input_views,
                    query_views = query_views,
                    pred_views = self.pred_view)
        elif self.double_pool_tree.pool_size == self.double_pool_tree.num:
            self.vis = Munch(
                    input_views = tf.concat(self.query.context.frames, axis = 2),
                    query_views = self.target,
                    pred_views = self.pred_view,
                    file_name=self.file_name,
                    ranking=self.ranking_view,
                    # sentence=self.sentences,
                )
        else:
            self.vis = Munch(
                    file_name=self.file_name

                )



class GQN_with_2dencoder(GQNBase):
    '''shares some fns w/ gqn3d and gqn2d'''
    
    def get_inputs2Denc(self):
        return self.query.context.frames

    def get_outputs2Denc(self, inputs):
        with tf.name_scope('2Dencoder') as scope:
            return utils.tfutil.concat_apply_split(
                inputs,
                self.output_encoder2d_f1)



        
    def aggregate(self, features):
        n = 1.0/float(len(features[0]))
        return [sum(feature)*n for feature in features]
        
class GQN3D(GQN_with_2dencoder):


    def pos_criterion_unscaled(self,x,y):
        val = self.pos_criterion(x,y)
        # st()
        val = val*self._total(x)
        return val
    def pixelrecon_criterion_unscaled(self,x,y):
        val = self.pixelrecon_criterion(x,y)
        val = val*self._total(x)
        return val

    @staticmethod
    def _total(tensor):
        return tf.cast(tf.math.reduce_prod(tensor.shape),tf.float32)

    def compose_tree(self, treex, latent_canvas_size):
        for i in range(0, treex.num_children):
            treex.children[i] = self.compose_tree(treex.children[i], latent_canvas_size)

        # one hot embedding of a word
        ohe = self.get_code(self.dictionary, treex.word)
        # print("ohe",ohe.shape,ohe.dtype)

        if treex.function == 'combine':
            vis_dist = self.vis_dist(ohe)
            pos_dist = self.pos_dist(ohe)
            
            if treex.num_children > 0:
                # visual content
                vis_dist_child = treex.children[0].vis_dist
                vis_dist = self.combine_vis(vis_dist, vis_dist_child)
                # visual position
                pos_dist_child = treex.children[0].pos_dist
                pos_dist = self.combine_pos(pos_dist, pos_dist_child)
            
            treex.vis_dist = vis_dist
            treex.pos_dist = pos_dist

        elif treex.function == 'describe':
            # blend visual words
            # st()
            vis_dist = self.vis_dist(ohe)
            pos_dist = self.pos_dist(ohe)
            
            if treex.num_children > 0:
                # visual content
                vis_dist_child = treex.children[0].vis_dist
                vis_dist = self.describe_vis(vis_dist_child, vis_dist)
                # visual position
                pos_dist_child = treex.children[0].pos_dist
                pos_dist = self.describe_pos(pos_dist_child, pos_dist)
            treex.pos_dist = pos_dist

            # regress bbox
            treex.pos = np.maximum(treex.bbox[3:] // self.ds, [1, 1, 1])
            target_box = tf.convert_to_tensor(np.array(treex.bbox[3:])[np.newaxis, ...].astype(np.float32))
            regress_box, kl_box = self.box_vae(target_box, prior=treex.pos_dist)
            treex.pos_loss = self.pos_criterion_unscaled(target_box,regress_box) + kl_box
            # print(treex.pos,vis_dist)
            if treex.parent == None:
                ones = self.get_ones([1, 1, 1])
                # if not self.bg_bias:
                bg_vis_dist = [tf.zeros(latent_canvas_size),tf.zeros(latent_canvas_size)]
                # else:
                #     bg_vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
                #                    self.bias_var(ones).view(*latent_canvas_size)]
                b = np.maximum(treex.bbox // self.ds, [0, 0, 0, 1, 1, 1])

                bg_vis_dist = [self.update_util(bg_vis_dist[0], b, self.transform(vis_dist[0], tf.convert_to_tensor(treex.pos,tf.int32)),
                                                'assign'), \
                               self.update_util(bg_vis_dist[1], b,
                                                self.transform(vis_dist[1], tf.convert_to_tensor(treex.pos,tf.int32), variance=True),
                                                'assign')]
                vis_dist = bg_vis_dist
            else:
                try:
                    # resize vis_dist
                    # st()
                    vis_dist = [self.transform(vis_dist[0], tf.convert_to_tensor(treex.pos,tf.int32)), \
                                self.transform(vis_dist[1],tf.convert_to_tensor(treex.pos,tf.int32), variance=True)]
                except:
                    import IPython;
                    IPython.embed()
            # st()
            treex.vis_dist = vis_dist

        elif treex.function == 'layout':
            # get pos word as position prior
            # st()
            # st()
            treex.pos_dist = self.pos_dist(ohe)
            assert (treex.num_children > 0)

            # get offsets: use gt for training
            l_pos = treex.children[0].pos
            l_offset = np.maximum(treex.children[0].bbox[:3] // self.ds, [1, 1, 1])

            r_pos = treex.children[1].pos
            r_offset = np.maximum(treex.children[1].bbox[:3] // self.ds, [1, 1, 1])

            # regress offsets
            target_offset = np.append(l_offset * self.ds, r_offset * self.ds).astype(np.float32)
            target_offset = tf.convert_to_tensor(target_offset[np.newaxis, ...])
            regress_offset, kl_offset = self.offset_vae(target_offset, prior=treex.pos_dist)
            treex.pos_loss = self.pos_criterion_unscaled(regress_offset, target_offset) + kl_offset + treex.children[0].pos_loss + treex.children[1].pos_loss

            ######################### constructing latent map ###############################
            # bias filled mean&var
            ones = self.get_ones([1, 1, 1])
            # if not self.bg_bias:
            # st()
            vis_dist_variable = [tf.zeros(latent_canvas_size),tf.zeros(latent_canvas_size)]
            # else:
            #     vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
            #                 self.bias_var(ones).view(*latent_canvas_size)]

            # arrange the layout of two children
            # st()
            # vis_dist = [None,None]
            # st()
            vis_dist = [None,None]
            # print(treex.children[0].vis_dist[0].shape)
            # st()
            vis_dist[0] = self.update_util(vis_dist_variable[0], list(l_offset) + list(l_pos), treex.children[0].vis_dist[0],'assign')
            vis_dist[1] = self.update_util(vis_dist_variable[1], list(l_offset) + list(l_pos), treex.children[0].vis_dist[1],
                                           'assign')

            vis_dist_variable = [None,None]
            # del vis_dist_variable[0]
            # del vis_dist_variable[1]
                
            vis_dist[0] = self.update_util(vis_dist[0], list(r_offset) + list(r_pos), treex.children[1].vis_dist[0],
                                           'assign')
            vis_dist[1] = self.update_util(vis_dist[1], list(r_offset) + list(r_pos), treex.children[1].vis_dist[1],
                                           'assign')

            # continue layout
            if treex.parent != None:
                p = [min(l_offset[0], r_offset[0]), min(l_offset[1], r_offset[1]), min(l_offset[2], r_offset[2]), \
                     max(l_offset[0] + l_pos[0], r_offset[0] + r_pos[0]),
                     max(l_offset[1] + l_pos[1], r_offset[1] + r_pos[1]),
                     max(l_offset[2] + l_pos[2], r_offset[2] + r_pos[2])]
                treex.pos = [p[3] - p[0], p[4] - p[1], p[5] - p[2]]
                treex.vis_dist = [vis_dist[0][:, p[0]:p[3], p[1]:p[4], p[2]:p[5],:], \
                                  vis_dist[1][:,  p[0]:p[3], p[1]:p[4], p[2]:p[5],:]]
            else:
                treex.vis_dist = vis_dist


        return treex

    def check_valid(self, offsets, l_pos, r_pos, im_size):
        flag = True
        if offsets[0] + l_pos[0] > im_size:
            flag = False
            return flag
        if offsets[1] + l_pos[1] > im_size:
            flag = False
            return flag
        if offsets[2] + l_pos[2] > im_size:
            flag = False
            return flag
        if offsets[3] + r_pos[0] > im_size:
            flag = False
            return flag
        if offsets[4] + r_pos[1] > im_size:
            flag = False
            return flag
        if offsets[5] + r_pos[2] > im_size:
            flag = False
            return flag
        return flag

    def generate_compose_tree(self, treex, latent_canvas_size):
            # st()
            for i in range(0, treex.num_children):
                treex.children[i] = self.generate_compose_tree(treex.children[i], latent_canvas_size)

            # one hot embedding of a word
            ohe = self.get_code(self.dictionary, treex.word)

            if treex.function == 'combine':
                vis_dist = self.vis_dist(ohe)
                pos_dist = self.pos_dist(ohe)
                if treex.num_children > 0:
                    # visual content
                    vis_dist_child = treex.children[0].vis_dist
                    vis_dist = self.combine_vis(vis_dist, vis_dist_child)
                    # visual position
                    pos_dist_child = treex.children[0].pos_dist
                    pos_dist = self.combine_pos(pos_dist, pos_dist_child)

                treex.vis_dist = vis_dist
                treex.pos_dist = pos_dist

            elif treex.function == 'describe':
                # blend visual words
                # st()
                vis_dist = self.vis_dist(ohe)
                pos_dist = self.pos_dist(ohe)
                if treex.num_children > 0:
                    # visual content
                    vis_dist_child = treex.children[0].vis_dist
                    vis_dist = self.describe_vis(vis_dist_child, vis_dist)
                    # visual position
                    pos_dist_child = treex.children[0].pos_dist
                    pos_dist = self.describe_pos(pos_dist_child, pos_dist)

                treex.pos_dist = pos_dist

                # regress bbox
                target_box = tf.convert_to_tensor(np.array(treex.bbox[3:]).astype(np.float32))

                treex.pos = tf.cast(tf.reshape(tf.clip_by_value(self.box_vae.generate(prior=treex.pos_dist),self.ds,self.im_size),[-1]) // self.ds,tf.int32)
                print("pos value",treex.pos)
                # treex.pos = 

                if treex.parent == None:
                    ones = self.get_ones([1, 1,1])
                    # st()
                    # if not self.bg_bias:
                    #     bg_vis_dist = [Variable(torch.zeros(latent_canvas_size)).cuda(), \
                    #                    Variable(torch.zeros(latent_canvas_size)).cuda()]
                    # else:
                    #     bg_vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
                    #                    self.bias_var(ones).view(*latent_canvas_size)]
                    bg_vis_dist = [tf.zeros(latent_canvas_size),tf.zeros(latent_canvas_size)]

                    b = [int(latent_canvas_size[1]) // 2 - treex.pos[0] // 2,
                         int(latent_canvas_size[2]) // 2 - treex.pos[1] // 2,int(latent_canvas_size[3]) // 2 - treex.pos[2] // 2, treex.pos[0], treex.pos[1], treex.pos[2]]


                    bg_vis_dist = [self.update_util(bg_vis_dist[0], b, self.transform(vis_dist[0], tf.convert_to_tensor(treex.pos,tf.int32)),
                                                    'assign'), \
                                   self.update_util(bg_vis_dist[1], b,
                                                    self.transform(vis_dist[1], tf.convert_to_tensor(treex.pos,tf.int32), variance=True),
                                                    'assign')]

                    vis_dist = bg_vis_dist
                    treex.offsets = b
                else:
                    # resize vis_dist
                    vis_dist = [self.transform(vis_dist[0], treex.pos), \
                                self.transform(vis_dist[1], treex.pos, variance=True)]

                treex.vis_dist = vis_dist

            elif treex.function == 'layout':
                # st()
                # get pos word as position prior
                treex.pos_dist = self.pos_dist(ohe)
                assert (treex.num_children > 0)

                # get offsets: use gt for training
                l_pos = treex.children[0].pos
                r_pos = treex.children[1].pos
                # l_pos = treex.children[0].pos
                l_offset = np.maximum(treex.children[0].bbox[:3] // self.ds, [1, 1, 1])

                # r_pos = treex.children[1].pos
                r_offset = np.maximum(treex.children[1].bbox[:3] // self.ds, [1, 1, 1])

                # regress offsets
                # target_offset = np.append(l_offset * self.ds, r_offset * self.ds).astype(np.float32)
                # target_offset = tf.convert_to_tensor(target_offset)

                offsets = tf.cast(tf.reshape(tf.clip_by_value(tf.cast(self.offset_vae.generate(prior=treex.pos_dist),tf.int32), 0,self.im_size),[-1]) // self.ds,tf.int32)
                # st()
                # st()
                # offsets = tf.cast(target_offset[0],tf.int32)
                countdown = 0
                while self.check_valid(offsets, l_pos, r_pos, self.im_size // self.ds) == False:
                    offsets = tf.cast(tf.reshape(tf.clip_by_value(self.offset_vae.generate(prior=treex.pos_dist), 0,self.im_size),[-1]) // self.ds,tf.int32)
                    if countdown >= 100:
                        print('Tried proposing more than 100 times.')
                        import IPython;
                        IPython.embed()
                        print('Warning! Manually adapt offsets')
                        lat_size = self.im_size // self.ds
                        if offsets[0] + l_pos[0] > lat_size:
                            offsets[0] = lat_size - l_pos[0]
                        if offsets[1] + l_pos[1] > lat_size:
                            offsets[1] = lat_size - l_pos[1]
                        if offsets[2] + l_pos[2] > lat_size:
                            offsets[2] = lat_size - l_pos[2]
                        if offsets[3] + r_pos[0] > lat_size:
                            offsets[3] = lat_size - r_pos[0]
                        if offsets[4] + r_pos[1] > lat_size:
                            offsets[4] = lat_size - r_pos[1]
                        if offsets[5] + r_pos[2] > lat_size:
                            offsets[5] = lat_size - r_pos[2]


                    countdown += 1

                treex.offsets = offsets
                l_offset = offsets[:3]
                r_offset = offsets[3:]

                ######################### constructing latent map ###############################
                # bias filled mean&var
                # ones = self.get_ones([1, 1])
                vis_dist_variable = [tf.zeros(latent_canvas_size), \
                                    tf.zeros(latent_canvas_size)]
                # else:
                #     bg_vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
                #                    self.bias_var(ones).view(*latent_canvas_size)]
                vis_dist = [None,None]

                # vis_dist = bg_vis_dist
                try:
                    # arrange the layout of two children
                    # st()
                    vis_dist[0] = self.update_util(vis_dist_variable[0], list(l_offset) + list(l_pos), treex.children[0].vis_dist[0],
                                                   'assign')
                    vis_dist[1] = self.update_util(vis_dist_variable[1], list(l_offset) + list(l_pos), treex.children[0].vis_dist[1],
                                                   'assign')

                    vis_dist[0] = self.update_util(vis_dist[0], list(r_offset) + list(r_pos), treex.children[1].vis_dist[0],
                                                   'assign')
                    vis_dist[1] = self.update_util(vis_dist[1], list(r_offset) + list(r_pos), treex.children[1].vis_dist[1],
                                                   'assign')
                except:
                    print('latent distribution doesnt fit size.')
                    import IPython;
                    IPython.embed()

                if treex.parent != None:
                    p = [min(l_offset[0], r_offset[0]), min(l_offset[1], r_offset[1]), min(l_offset[2], r_offset[2]), \
                         max(l_offset[0] + l_pos[0], r_offset[0] + r_pos[0]),
                         max(l_offset[1] + l_pos[1], r_offset[1] + r_pos[1]),
                         max(l_offset[2] + l_pos[2], r_offset[2] + r_pos[2])]
                    treex.pos = [p[3] - p[0], p[4] - p[1], p[5] - p[2]]
                    treex.vis_dist = [vis_dist[0][:, p[0]:p[3], p[1]:p[4], p[2]:p[5],:], \
                                      vis_dist[1][:,  p[0]:p[3], p[1]:p[4], p[2]:p[5],:]]
                else:
                    treex.vis_dist = vis_dist

            return treex

    # def add_mask(self,tensor,bbox):
    #     maskVal = np.zeros_like(tensor,np.float32)
    #     padding =1
    #     bx, by, bz, bx_d, by_d, bz_d = bbox // self.ds
    #     maskVal[bx-padding:bx+bx_d+padding, by-padding:by+by_d+padding, bz-padding:bz+bz_d+padding, :] =  1
    #     return tensor*maskVal

    def crop(self,tensor,bbox):
        padding =1
        bx, by, bz, bx_d, by_d, bz_d = bbox // self.ds
        cropped_tensor = tf.strided_slice(tensor,[bx,by,bz],[bx+bx_d+padding,by+by_d+padding,bz+bz_d+padding],[1,1,1])
        return cropped_tensor

    def findType(self,word):
        if word in self.dictionary_textures:
            typeVal = "texture"
        elif word in self.dictionary_sizes:
            typeVal = "size"
        elif word in self.dictionary_colors:
            typeVal = "color"
        elif word in self.dictionary_shapes:
            typeVal = "shape"
        return typeVal

    def compose_tree_cond(self, treex,outputs3D_resized, latent_canvas_size,i,intermediate_dict={}):
        if not hasattr(treex, 'bbox'):
            treex.bbox = treex.parent.bbox
        for i in range(0, treex.num_children):
            treex.children[i],intermediate_dict = self.compose_tree_cond(treex.children[i],outputs3D_resized, latent_canvas_size,i,intermediate_dict)


        typeVal = self.findType(treex.word)

        if typeVal == "color":
            treex.word = "red"
        if typeVal == "shape":
            treex.word = "sphere"
        ohe = self.get_code(self.dictionary, treex.word)

        bx, by, bz, bx_d, by_d, bz_d = treex.bbox // self.ds

        # outputs3D_cropped = self.crop(output3d,treex.bbox)
        # outputs3D_resized = utils.tfutil.resize_voxel(outputs3D_cropped,16,16,16)
        # st()
        # st()
        if treex.function == 'combine':
            if not (const.SAMPLE_VALUES and tf.math.equal(self.index, 2)):
                ohe_tiled = tf.tile(tf.expand_dims(tf.expand_dims(ohe,0),0),[16,16,16,1])

                output3d_cond = tf.concat([outputs3D_resized,ohe_tiled],axis=-1)

                output3d_cond = self.inf_net(output3d_cond)
                # st()
                mean_vis = self.h_mean_cond_vis(output3d_cond)
                # 50 numbers
                var_vis = self.h_var_cond_vis(output3d_cond)
                # 50 number

                z_vis = self.sampler(mean_vis,var_vis)

                vis_kl_loss = self.kld(mean_vis,var_vis)
            else:
                z_vis = tf.keras.backend.random_normal(shape=[1,self.z_dim], mean=0., stddev=1.,seed=random.randint(1,100))
                z_vis = self.one_gaussian
                vis_kl_loss = 0

            if random.random() > self.THRESHOLD:
                z_merged = tf.concat([z_vis,ohe],1)
            else:
                z_merged = tf.concat([z_vis,tf.zeros_like(ohe)],1)
            
            if const.concat_3d:
                z_vis = self.vis_dist_cond(z_merged)
            else:
                z_vis = z_merged

            if treex.num_children > 0:
                if const.GoModules and hasattr(treex.children[0], 'vis_mean'):
                    vis_dist_child = treex.children[0].vis_mean
                    z_vis = self.combine_vis_z(z_vis ,vis_dist_child)
                vis_kl_loss = vis_kl_loss + treex.children[0].vis_kl_loss

            if not const.GoModules:
                typeVal = self.findType(treex.word)
                intermediate_dict[typeVal] = z_vis
            else:
                treex.vis_mean = z_vis

            treex.vis_kl_loss = vis_kl_loss


        elif treex.function == 'describe':
            st()
            if not (const.SAMPLE_VALUES and tf.math.equal(self.index, 2)):
                ohe_tiled = tf.tile(tf.expand_dims(tf.expand_dims(ohe,0),0),[16,16,16,1])
                
                output3d_cond = tf.concat([outputs3D_resized,ohe_tiled],axis=-1)

                output3d_cond = self.inf_net(output3d_cond)

                mean_vis = self.h_mean_cond_vis(output3d_cond)
                # 50 numbers
                var_vis = self.h_var_cond_vis(output3d_cond)
                # 50 number
                z_vis = self.sampler(mean_vis,var_vis)
                # st()
                vis_kl_loss =  self.kld(mean_vis,var_vis)
            else:
                z_vis = tf.keras.backend.random_normal(shape=[1,self.z_dim], mean=0., stddev=1.,seed=random.randint(1,100))
                z_vis = self.one_gaussian
                vis_kl_loss = 0
 
            
            z_merged = tf.concat([z_vis,ohe],1)

            if const.concat_3d:
                z_vis = self.vis_dist_cond(z_merged)
            else:
                z_vis = z_merged
    
            # st()
            if treex.num_children > 0:
                # visual content
                if const.GoModules and hasattr(treex.children[0], 'vis_mean'):
                    vis_dist_child = treex.children[0].vis_mean
                    z_vis = self.describe_vis_z(vis_dist_child,z_vis)
                else:
                    # st()
                    z_vis_append = z_vis
                    if "color" in intermediate_dict.keys():
                        z_vis_append =  tf.concat([z_vis_append,intermediate_dict["color"]],axis=-1)

                    if "size" in intermediate_dict.keys():
                        z_vis_append = tf.concat([z_vis_append,intermediate_dict["size"]],axis=-1)

                    if "texture" in intermediate_dict.keys():
                        z_vis_append = tf.concat([z_vis_append,intermediate_dict["texture"]],axis=-1)



                    # z_vis = tf.concat([z_vis,intermediate_dict["color"],intermediate_dict["size"],intermediate_dict["texture"]],axis=-1)
                # visual position
                vis_kl_loss = vis_kl_loss + treex.children[0].vis_kl_loss
            else:
                z_vis_append = z_vis
                if "color" in intermediate_dict.keys():
                    z_vis_append =  tf.concat([z_vis_append,intermediate_dict["color"]],axis=-1)

                if "size" in intermediate_dict.keys():
                    z_vis_append = tf.concat([z_vis_append,intermediate_dict["size"]],axis=-1)

                if "texture" in intermediate_dict.keys():
                    z_vis_append = tf.concat([z_vis_append,intermediate_dict["texture"]],axis=-1)

                vis_kl_loss = vis_kl_loss
            
            if not const.concat_3d:
                z_vis = self.vis_dist_cond(z_vis_append)
            z_vis = self.renderer_z(z_vis)
            treex.vis_kl_loss = vis_kl_loss 

            if (const.segmentation and not self.loading)  or (const.segmentation and const.override_loading):
                if const.segmentation_stop:
                    z_vis = tf.stop_gradient(z_vis)
                # st()
                if not const.classification:
                    segmentation_mask = self.fcn(z_vis)
                    segmentation_target_mask = self.segmentation_target_dict[treex.word]

                else:
                    segmentation_mask = self.classify(z_vis,self.loading)
                    if self.loading:
                        segmentation_target_mask = self.classification_target_dict[treex.word]
                    else:
                        segmentation_target_mask = self.segmentation_target_dict[treex.word]
                intermediate_dict["segmentation_mask"] = segmentation_mask
                intermediate_dict["segmentation_target_mask"] = segmentation_target_mask

            treex.pos = np.maximum(treex.bbox[3:] // self.ds, [1, 1, 1])
            target_box = tf.convert_to_tensor(np.array(treex.bbox[3:])[np.newaxis, ...].astype(np.float32))

            if treex.parent == None:
                ones = self.get_ones([1, 1, 1])

                bg_vis_dist = tf.zeros(latent_canvas_size)

                b = np.maximum(treex.bbox // self.ds, [0, 0, 0, 1, 1, 1])

                transformed_voxel = self.transform(z_vis, tf.convert_to_tensor(treex.pos,tf.int32))
                transformed_voxel = tf.expand_dims(transformed_voxel,0)

                bg_vis_dist = self.update_util(bg_vis_dist, b, transformed_voxel,
                                                'assign')
                z_vis = bg_vis_dist
            else:
                try:
                    z_vis = self.transform(z_vis, tf.convert_to_tensor(treex.pos,tf.int32))
                                
                except:
                    import IPython;
                    IPython.embed()
            treex.vis_mean = z_vis

        elif treex.function == 'layout':
            st()
            ohe_tiled = tf.tile(tf.expand_dims(tf.expand_dims(ohe,0),0),[32,32,32,1])

            # get pos word as position prior
            treex.pos_dist = self.pos_dist(ohe)

            assert (treex.num_children > 0)

            # get offsets: use gt for training
            l_pos = treex.children[0].pos
            l_offset = np.maximum(treex.children[0].bbox[:3] // self.ds, [1, 1, 1])

            r_pos = treex.children[1].pos
            r_offset = np.maximum(treex.children[1].bbox[:3] // self.ds, [1, 1, 1])

            # regress offsets
            target_offset = np.append(l_offset * self.ds, r_offset * self.ds).astype(np.float32)
            target_offset = tf.convert_to_tensor(target_offset[np.newaxis, ...])
            regress_offset, kl_offset = self.offset_vae(target_offset, prior=treex.pos_dist)
            treex.pos_loss = self.pos_criterion_unscaled(regress_offset, target_offset) + kl_offset + treex.children[0].pos_loss + treex.children[1].pos_loss

            treex.vis_kl_loss = treex.children[0].vis_kl_loss + treex.children[1].vis_kl_loss
            ######################### constructing latent map ###############################
            # bias filled mean&var
            ones = self.get_ones([1, 1, 1])
            # if not self.bg_bias:
            # st()
            vis_mean = tf.zeros(latent_canvas_size)
            # else:
            #     vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
            #                 self.bias_var(ones).view(*latent_canvas_size)]

            # arrange the layout of two children
            # st()
            # vis_dist = [None,None]
            # st()
            # vis_dist = [None,None]
            # print(treex.children[0].vis_dist[0].shape)
            # st()
            vis_mean = self.update_util(vis_mean, list(l_offset) + list(l_pos), treex.children[0].vis_mean,'assign')
            # vis_dist[1] = self.update_util(vis_dist_variable[1], list(l_offset) + list(l_pos), treex.children[0].vis_dist[1],
            #                                'assign')

            # vis_dist_variable = [None,None]
            # del vis_dist_variable[0]
            # del vis_dist_variable[1]
                
            vis_mean = self.update_util(vis_mean, list(r_offset) + list(r_pos), treex.children[1].vis_mean,'assign')
            # vis_dist[1] = self.update_util(vis_dist[1], list(r_offset) + list(r_pos), treex.children[1].vis_dist[1],
            #                                'assign')

            # continue layout
            if treex.parent != None:
                p = [min(l_offset[0], r_offset[0]), min(l_offset[1], r_offset[1]), min(l_offset[2], r_offset[2]), \
                     max(l_offset[0] + l_pos[0], r_offset[0] + r_pos[0]),
                     max(l_offset[1] + l_pos[1], r_offset[1] + r_pos[1]),
                     max(l_offset[2] + l_pos[2], r_offset[2] + r_pos[2])]
                treex.pos = [p[3] - p[0], p[4] - p[1], p[5] - p[2]]
                treex.vis_mean = vis_mean[:, p[0]:p[3], p[1]:p[4], p[2]:p[5],:]
            else:
                treex.vis_mean = vis_mean

        return treex,intermediate_dict

    def compose_tree_cond_custom(self, treex,outputs3D_resized, latent_canvas_size,i,intermediate_dict={}):
        # making sure all attributes have bbox
        # st()
        if not hasattr(treex, 'bbox'):
            treex.bbox = treex.parent.bbox
        for i in range(0, treex.num_children):
            treex.children[i],intermediate_dict = self.compose_tree_cond_custom(treex.children[i],outputs3D_resized, latent_canvas_size,i,intermediate_dict)

        # one hot embedding of a word
        # ohe = self.get_code_cond(self.dictionaries, treex.word)
        typeVal = self.findType(treex.word)
        if typeVal == "color":
            ohe = self.get_code(self.dictionary_colors, treex.word)
        elif typeVal == "size":
            ohe = self.get_code(self.dictionary_sizes,treex.word)
        elif typeVal == "texture":
            ohe = self.get_code(self.dictionary_textures,treex.word)
        elif typeVal == "shape":
            ohe = self.get_code(self.dictionary_shapes,treex.word)
        # output3d = self.outputs3D[i]
        

        bx, by, bz, bx_d, by_d, bz_d = treex.bbox // self.ds

        # st()
        # outputs3D_cropped = self.crop(output3d,treex.bbox)
        # outputs3D_resized = utils.tfutil.resize_voxel(outputs3D_cropped,16,16,16)
        # st()
        if treex.function == 'combine':

            if not (const.SAMPLE_VALUES and  tf.math.equal(self.index, 2)):
                ohe_tiled = tf.tile(tf.expand_dims(tf.expand_dims(ohe,0),0),[16,16,16,1])

                output3d_cond = tf.concat([outputs3D_resized,ohe_tiled],axis=-1)
                if typeVal=="color":
                    output3d_cond = self.inf_net_color(output3d_cond)
                elif typeVal=="size":
                    output3d_cond = self.inf_net_size(output3d_cond)
                elif typeVal=="texture":
                    output3d_cond = self.inf_net_text(output3d_cond)
                # st()
                mean_vis = self.h_mean_cond_vis(output3d_cond)
                # 50 numbers
                var_vis = self.h_var_cond_vis(output3d_cond)
                # 50 number


                z_vis = self.sampler(mean_vis,var_vis)

                vis_kl_loss = self.kld(mean_vis,var_vis)
            else:
                z_vis = tf.keras.backend.random_normal(shape=[1,self.z_dim], mean=0., stddev=1.)
                vis_kl_loss = 0
            if random.random() > self.THRESHOLD:
                z_merged = tf.concat([z_vis,ohe],1)
            else:
                z_merged = tf.concat([z_vis,tf.zeros_like(ohe)],1)

            if const.concat_3d:
                if typeVal=="color":
                    z_vis = self.vis_dist_cond_color(z_merged)
                elif typeVal=="size":
                    z_vis = self.vis_dist_cond_size(z_merged)
                elif typeVal=="texture":
                    z_vis = self.vis_dist_cond_text(z_merged)
            else:
                z_vis = z_merged

            if treex.num_children > 0:
                if const.GoModules and  hasattr(treex.children[0], 'vis_mean'):
                    vis_dist_child = treex.children[0].vis_mean
                    z_vis = self.combine_vis_z(z_vis,vis_dist_child)
                vis_kl_loss = vis_kl_loss + treex.children[0].vis_kl_loss

            if not const.GoModules:
                typeVal = self.findType(treex.word)
                intermediate_dict[typeVal] = z_vis
            else:
                treex.vis_mean = z_vis
            treex.vis_kl_loss = vis_kl_loss

        elif treex.function == 'describe':
            if not (const.SAMPLE_VALUES and  tf.math.equal(self.index, 2)):
                ohe_tiled = tf.tile(tf.expand_dims(tf.expand_dims(ohe,0),0),[16,16,16,1])
                
                output3d_cond = tf.concat([outputs3D_resized,ohe_tiled],axis=-1)

                output3d_cond = self.inf_net_shape(output3d_cond)

                mean_vis = self.h_mean_cond_vis(output3d_cond)
                # 50 numbers
                var_vis = self.h_var_cond_vis(output3d_cond)
                # 50 number
                z_vis = self.sampler(mean_vis,var_vis)
    
                vis_kl_loss =  self.kld(mean_vis,var_vis)

            else:
                z_vis = tf.keras.backend.random_normal(shape=[1,self.z_dim], mean=0., stddev=1.)
                vis_kl_loss = 0   
            # st()
            z_merged = tf.concat([z_vis,ohe],1)

            if const.concat_3d:
                z_vis = self.vis_dist_cond_shape(z_merged)
            else:
                z_vis = z_merged
    
            # st()
            if treex.num_children > 0:
                # visual content
                if const.GoModules  and  hasattr(treex.children[0], 'vis_mean'):
                    vis_dist_child = treex.children[0].vis_mean
                    z_vis = self.describe_vis_z(vis_dist_child,z_vis)
                else:
                    # st()
                    z_vis_append = z_vis
                    if "color" in intermediate_dict.keys():
                        z_vis_append =  tf.concat([z_vis_append,intermediate_dict["color"]],axis=-1)

                    if "size" in intermediate_dict.keys():
                        z_vis_append = tf.concat([z_vis_append,intermediate_dict["size"]],axis=-1)

                    if "texture" in intermediate_dict.keys():
                        z_vis_append = tf.concat([z_vis_append,intermediate_dict["texture"]],axis=-1)

                # visual position
                vis_kl_loss = vis_kl_loss + treex.children[0].vis_kl_loss
            else:
                z_vis_append = z_vis
                if "color" in intermediate_dict.keys():
                    z_vis_append =  tf.concat([z_vis_append,intermediate_dict["color"]],axis=-1)

                if "size" in intermediate_dict.keys():
                    z_vis_append = tf.concat([z_vis_append,intermediate_dict["size"]],axis=-1)

                if "texture" in intermediate_dict.keys():
                    z_vis_append = tf.concat([z_vis_append,intermediate_dict["texture"]],axis=-1)

                vis_kl_loss = vis_kl_loss
            
            if not const.concat_3d:
                z_vis = self.vis_dist_cond(z_vis_append)
            z_vis = self.renderer_z(z_vis)
            treex.vis_kl_loss = vis_kl_loss 

            treex.pos = np.maximum(treex.bbox[3:] // self.ds, [1, 1, 1])
            target_box = tf.convert_to_tensor(np.array(treex.bbox[3:])[np.newaxis, ...].astype(np.float32))

            if treex.parent == None:
                ones = self.get_ones([1, 1, 1])

                bg_vis_dist = tf.zeros(latent_canvas_size)

                b = np.maximum(treex.bbox // self.ds, [0, 0, 0, 1, 1, 1])

                transformed_voxel = self.transform(z_vis, tf.convert_to_tensor(treex.pos,tf.int32))
                transformed_voxel = tf.expand_dims(transformed_voxel,0)

                bg_vis_dist = self.update_util(bg_vis_dist, b, transformed_voxel,
                                                'assign')
                z_vis = bg_vis_dist
            else:
                try:
                    z_vis = self.transform(z_vis, tf.convert_to_tensor(treex.pos,tf.int32))
                                
                except:
                    import IPython;
                    IPython.embed()
            treex.vis_mean = z_vis

        elif treex.function == 'layout':
            st()
            ohe_tiled = tf.tile(tf.expand_dims(tf.expand_dims(ohe,0),0),[32,32,32,1])

            # get pos word as position prior
            treex.pos_dist = self.pos_dist(ohe)

            assert (treex.num_children > 0)

            # get offsets: use gt for training
            l_pos = treex.children[0].pos
            l_offset = np.maximum(treex.children[0].bbox[:3] // self.ds, [1, 1, 1])

            r_pos = treex.children[1].pos
            r_offset = np.maximum(treex.children[1].bbox[:3] // self.ds, [1, 1, 1])

            # regress offsets
            target_offset = np.append(l_offset * self.ds, r_offset * self.ds).astype(np.float32)
            target_offset = tf.convert_to_tensor(target_offset[np.newaxis, ...])
            regress_offset, kl_offset = self.offset_vae(target_offset, prior=treex.pos_dist)
            treex.pos_loss = self.pos_criterion_unscaled(regress_offset, target_offset) + kl_offset + treex.children[0].pos_loss + treex.children[1].pos_loss

            treex.vis_kl_loss = treex.children[0].vis_kl_loss + treex.children[1].vis_kl_loss
            ######################### constructing latent map ###############################
            # bias filled mean&var
            ones = self.get_ones([1, 1, 1])
            # if not self.bg_bias:
            # st()
            vis_mean = tf.zeros(latent_canvas_size)
            # else:
            #     vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
            #                 self.bias_var(ones).view(*latent_canvas_size)]

            # arrange the layout of two children
            # st()
            # vis_dist = [None,None]
            # st()
            # vis_dist = [None,None]
            # print(treex.children[0].vis_dist[0].shape)
            # st()
            vis_mean = self.update_util(vis_mean, list(l_offset) + list(l_pos), treex.children[0].vis_mean,'assign')
            # vis_dist[1] = self.update_util(vis_dist_variable[1], list(l_offset) + list(l_pos), treex.children[0].vis_dist[1],
            #                                'assign')

            # vis_dist_variable = [None,None]
            # del vis_dist_variable[0]
            # del vis_dist_variable[1]
                
            vis_mean = self.update_util(vis_mean, list(r_offset) + list(r_pos), treex.children[1].vis_mean,'assign')
            # vis_dist[1] = self.update_util(vis_dist[1], list(r_offset) + list(r_pos), treex.children[1].vis_dist[1],
            #                                'assign')

            # continue layout
            if treex.parent != None:
                p = [min(l_offset[0], r_offset[0]), min(l_offset[1], r_offset[1]), min(l_offset[2], r_offset[2]), \
                     max(l_offset[0] + l_pos[0], r_offset[0] + r_pos[0]),
                     max(l_offset[1] + l_pos[1], r_offset[1] + r_pos[1]),
                     max(l_offset[2] + l_pos[2], r_offset[2] + r_pos[2])]
                treex.pos = [p[3] - p[0], p[4] - p[1], p[5] - p[2]]
                treex.vis_mean = vis_mean[:, p[0]:p[3], p[1]:p[4], p[2]:p[5],:]
            else:
                treex.vis_mean = vis_mean


        return treex,intermediate_dict

    def compose_tree_cond_color(self, treex,outputs3D_resized, latent_canvas_size,i,intermediate_dict={}):
        # making sure all attributes have bbox
        # st()
        if not hasattr(treex, 'bbox'):
            treex.bbox = treex.parent.bbox
        for i in range(0, treex.num_children):
            treex.children[i],intermediate_dict = self.compose_tree_cond_color(treex.children[i],outputs3D_resized, latent_canvas_size,i,intermediate_dict)
        

        if treex.function == "combine":
            if treex.word not in self.dictionary_colors:
                return treex,intermediate_dict
            else:
                ohe = self.get_code_cond(self.dictionaries, treex.word)

                bx, by, bz, bx_d, by_d, bz_d = treex.bbox // self.ds

                ohe_tiled = tf.tile(tf.expand_dims(tf.expand_dims(ohe,0),0),[16,16,16,1])
                output3d_cond = tf.concat([outputs3D_resized,ohe_tiled],axis=-1)
                # st()
                if const.inf_net:
                    output3d_cond = self.inf_net(output3d_cond)
                mean_vis = self.h_mean_cond_vis(output3d_cond)
                var_vis = self.h_var_cond_vis(output3d_cond)
                st()
                if const.visualize_inf and not const.LOAD_VAL:
                    val = self.sampler(mean_vis,var_vis,stoch=False)
                    self.record.writerow(val.numpy().reshape([-1]))
                    self.recordfile.flush()
                    self.metadata.writerow([treex.word])
                    self.metadatafile.flush()

                z_vis = self.sampler(mean_vis,var_vis)
                vis_kl_loss = self.kld(mean_vis,var_vis)
                z_merged = tf.concat([z_vis,ohe],1)
                z_vis = z_merged
                typeVal = self.findType(treex.word)
                intermediate_dict[typeVal] = [z_vis,vis_kl_loss]
        
        elif treex.function == "describe":
            z_vis,vis_kl_loss =  intermediate_dict["color"]
            z_vis = self.vis_dist_cond(z_vis)
            z_vis = self.renderer_z(z_vis)

            treex.pos = np.maximum(treex.bbox[3:] // self.ds, [1, 1, 1])
            # target_box = tf.convert_to_tensor(np.array(treex.bbox[3:])[np.newaxis, ...].astype(np.float32))
            # st()
            if treex.parent == None:
                ones = self.get_ones([1, 1, 1])

                bg_vis_dist = tf.zeros(latent_canvas_size)

                b = np.maximum(treex.bbox // self.ds, [0, 0, 0, 1, 1, 1])
                transformed_voxel = self.transform(z_vis, tf.convert_to_tensor(treex.pos,tf.int32))
                transformed_voxel = tf.expand_dims(transformed_voxel,0)
                # st()
                bg_vis_dist = self.update_util(bg_vis_dist, b,transformed_voxel ,
                                                'assign')
                z_vis = bg_vis_dist
            else:
                st()
            treex.vis_mean = z_vis
            treex.vis_kl_loss = vis_kl_loss
        return treex,intermediate_dict

    def get_code(self,dictionary, word):
        code = np.zeros((1, len(dictionary)))
        print(word)
        code[0, dictionary.index(word)] = 1
        # code = code
        code = tf.cast(tf.convert_to_tensor(code),"float32")
        return code

    def get_code_cond(self,dictionary, word):
        wordType = [key for key,val in dictionary.items() if word in val]
        assert len(wordType)==1
        wordType = wordType[0]
        curr_dictionary = dictionary[wordType]
        code = np.zeros((1, len(curr_dictionary)))
        # color shape location size textures
        if wordType == "color":
            code[0, curr_dictionary.index(word)] = 1
            code21 = self.colors2dict(code)
        
        elif wordType == "shape":
            code[0, curr_dictionary.index(word)] = 1
            code21 = self.shapes2dict(code)
        
        elif wordType == "location":
            code[0, curr_dictionary.index(word)] = 1
            code21 = self.locations2dict(code)
        
        elif wordType == "size":
            code[0, curr_dictionary.index(word)] = 1
            code21 = self.sizes2dict(code)        
        
        elif wordType == "textures":
            code[0, curr_dictionary.index(word)] = 1
            code21 = self.textures2dict(code)

        else:
            raise Exception("Impossible!")
                
        # code = code
        code21 = tf.cast(tf.convert_to_tensor(code21),"float32")
        return code21

    def get_ones(self, size):
        return tf.ones(size)

    # def assign_util(self, canvas_size, bx, update, mode):
    #     if mode == 'assign':
    #         # st()
    #         indices = self.gen_indices(bx[0],bx[1],bx[2],bx[3],bx[4],bx[5])
    #         a= tf.scatter_nd(indices,update,canvas_size)
    #         # a[:,bx[0]:bx[0] + bx[2], bx[1]:bx[1] + bx[3],:].assign(b)
    #     # elif mode == 'add':
    #     #     a[:, :, bx[0]:bx[0] + bx[2], bx[1]:bx[1] + bx[3]] = \
    #     #         a[:, :, bx[0]:bx[0] + bx[2], bx[1]:bx[1] + bx[3]] + b
    #     # elif mode == 'slice':
    #     #     a = a[:, :, bx[0]:bx[0] + bx[2], bx[1]:bx[1] + bx[3]].clone()
    #     # else:
    #     #     raise ValueError('Please specify the correct mode.')
    #     return a

    def update_util(self, canvas, bx, val, mode):
        if mode == 'assign':
            indices = self.gen_indices(bx[0],bx[1],bx[2],bx[3],bx[4],bx[5])
            a= tf.tensor_scatter_nd_update(canvas,indices,val)
        return a

    # def gen_indices(self,m,n,o,h,w,d):
    #     h_val =np.arange(m,m+h)
    #     w_val =np.arange(n,n+w)
    #     d_val =np.arange(o,o+d)
    #     indices = np.zeros((1,h,w,d,4))
    #     for i_h in range(h):
    #         for i_w in range(w):
    #             for i_d in range(d):
    #                 indices[0,i_h,i_w,i_d,:] = np.array([0,h_val[i_h],w_val[i_w],d_val[i_d]])
    #     indices = np.array(indices,np.int64)
    #     return indices

    def gen_indices(self,m,n,o,h,w,d):
        indm = tf.tile(tf.reshape(tf.range(m,m+h),[1,h,1,1,1]),[1,1,w,d,1])
        indn = tf.tile(tf.reshape(tf.range(n,n+w),[1,1,w,1,1]),[1,h,1,d,1])
        indo = tf.tile(tf.reshape(tf.range(o,o+d),[1,1,1,d,1]),[1,h,w,1,1])
        indz = tf.zeros([1,h,w,d,1],tf.int64)
        indices = tf.concat([indz,indm,indn,indo],-1)
        return indices

    def gentree(self,treex):
        prior_mean_all = []
        prior_var_all = []
        trees = []
        bboxes = []
        mask3d = []
        pos_loss = 0
        padding = 1
        if tf.math.not_equal(self.index, 2) or const.RANK_TENSORS or const.FREEZE_ENC_DEC:
            for i in range(0, len(treex)):  # iterate through every tree of the batch
                trees.append(self.compose_tree(treex[i], self.latent_canvas_size))
                prior_mean_all += [trees[i].vis_dist[0]]
                prior_var_all += [trees[i].vis_dist[1]]
                pos_loss += trees[i].pos_loss
                if treex[i].function == "describe":
                    bx, by, bz, bx_d, by_d, bz_d = self.trees[i].bbox // self.ds
                    zeros = np.zeros([32,32,32,32],np.float32)
                    ones =  np.ones([bx_d+padding,by_d+padding, bz_d+padding,32])
                    zeros[bx:bx+bx_d+padding, by:by+by_d+padding, bz:bz+bz_d+padding, :] =  ones
                else:
                    # considering that max objects is 2
                    bx, by, bz, bx_d, by_d, bz_d = self.trees[i].children[0].bbox // self.ds
                    zeros = np.zeros([32,32,32,32],np.float32)
                    ones =  np.ones([bx_d+padding,by_d+padding, bz_d+padding,32])
                    zeros[bx:bx+bx_d+padding, by:by+by_d+padding, bz:bz+bz_d+padding, :] =  ones

                    bx, by, bz, bx_d, by_d, bz_d = self.trees[i].children[1].bbox // self.ds
                    ones =  np.ones([bx_d+padding,by_d+padding, bz_d+padding,32])
                    zeros[bx:bx+bx_d+padding, by:by+by_d+padding, bz:bz+bz_d+padding, :] =  ones
                mask3d.append(zeros)
        else:
            for i in range(0, len(treex)):  # iterate through every tree of the batch
                trees.append(self.generate_compose_tree(treex[i], self.latent_canvas_size))
                # if treex[i].function == "describe":
                #     bx, by, bz, bx_d, by_d, bz_d = self.trees[i].bbox // self.ds
                #     zeros = np.zeros([32,32,32,32],np.float32)
                #     ones =  np.ones([bx_d+2*padding,by_d+2*padding, bz_d+2*padding,32])
                #     zeros[bx-padding:bx+bx_d+padding, by-padding:by+by_d+padding, bz-padding:bz+bz_d+padding, :] =  ones


                prior_mean_all += [trees[i].vis_dist[0]]
                prior_var_all += [trees[i].vis_dist[1]]
        
        prior_mean = tf.concat(prior_mean_all, 0)
        prior_var = tf.concat(prior_var_all, 0)
        # st()
        if const.renderer:
            # st()
            prior_mean, prior_var = self.renderer([prior_mean, prior_var])
        mask3d = tf.stack(mask3d)
        return  prior_mean,prior_var,pos_loss,mask3d

    def gentree_cond(self,treex):
        trees = []
        z_vis_all = []
        pos_loss = 0
        vis_kl_loss = 0
        # st()
        padding = 1
        mask3d = []
        segmentation_masks = []
        segmentation_target_masks = []
        # st()
        for i in range(0, len(treex)):  
            # intermediate_dict = {}
            if not (const.SAMPLE_VALUES and tf.math.equal(self.index,2)):
                output3d = self.outputs3D[i]
                outputs3D_cropped = self.crop(output3d,treex[i].bbox)
                outputs3D_resized = utils.tfutil.resize_voxel(outputs3D_cropped,tf.convert_to_tensor([16,16,16]))
            else:
                outputs3D_resized = None
            if const.onlycolor:
                tree , _ = self.compose_tree_cond_color(treex[i],outputs3D_resized, self.latent_canvas_size,i)
            elif const.custominf:
                tree , intermediate_dict = self.compose_tree_cond_custom(treex[i],outputs3D_resized, self.latent_canvas_size,i)                
            else:
                tree , intermediate_dict = self.compose_tree_cond(treex[i],outputs3D_resized, self.latent_canvas_size,i)
            trees.append(tree)
            if treex[i].function == "describe":
                bx, by, bz, bx_d, by_d, bz_d = self.trees[i].bbox // self.ds
                zeros = np.zeros([32,32,32,32],np.float32)
                ones =  np.ones([bx_d+padding,by_d+padding, bz_d+padding,32])
                zeros[bx:bx+bx_d+padding, by:by+by_d+padding, bz:bz+bz_d+padding, :] =  ones
            
            if (const.segmentation and not self.loading)  or (const.segmentation and const.override_loading):
                segmentation_mask =  intermediate_dict["segmentation_mask"]
                segmentation_target_mask =  intermediate_dict["segmentation_target_mask"]
                segmentation_masks.append(segmentation_mask)
                segmentation_target_masks.append(segmentation_target_mask)

            z_vis_all += [trees[i].vis_mean]
            vis_kl_loss += trees[i].vis_kl_loss
            mask3d.append(zeros)
        z_vis = tf.concat(z_vis_all, 0)
        # st()
        if (const.segmentation and not self.loading)  or (const.segmentation and const.override_loading):
            self.segmentation_masks = tf.concat(segmentation_masks,0)
            self.segmentation_target_masks = tf.concat(segmentation_target_masks,0)
        self.clean_tree(treex)
        mask3d = tf.stack(mask3d)
        pos_loss = None
        return  z_vis,pos_loss,vis_kl_loss,mask3d
    

    def getMask3d(self,treex):
        padding = 1
        mask3d = []
        for i in range(0, len(treex)):  
            # intermediate_dict = {}
            zeros = np.zeros([32,32,32,32],np.float32)
            if treex[i].function == "describe":
                bx, by, bz, bx_d, by_d, bz_d = self.trees[i].bbox // self.ds
                ones =  np.ones([bx_d+padding,by_d+padding, bz_d+padding,32])
                zeros[bx:bx+bx_d+padding, by:by+by_d+padding, bz:bz+bz_d+padding, :] =  ones
            mask3d.append(zeros)
        mask3d = tf.stack(mask3d)
        return  mask3d

    def rpn(self,tensor):
        cropped_tensor =  self.crop(tensor)


    def predict(self):
        if const.RANK_TENSORS:
            tmp = []
            for i in self.trees:
                val = i.wordVal
                tmp.append(val)


        if const.DIRECT_TREE_TO_IMG:
            self.prior_mean, self.prior_var, self.pos_loss, self.mask3d = self.gentree(self.trees)
            z_prior = self.sampler(self.prior_mean, self.prior_var)
            inputs2DdecPrior = self.get_inputs2Ddec([z_prior])
            outputs2DdecPrior = self.get_outputs2Ddec(inputs2DdecPrior)
            self.pred_view_prior = outputs2DdecPrior.pred_view

        elif (const.segmentation and not self.loading) or (const.segmentation and const.override_loading):
            inputs2Denc = self.get_inputs2Denc()

            # f1
            outputs2Denc = self.get_outputs2Denc(inputs2Denc)


            inputs3D = self.get_inputs3D(outputs2Denc)
            
            # f2
            outputs3D = self.get_outputs3D(inputs3D)

            if const.ARITH_MODE:
                outputs3D = self.do_arithmetic(outputs3D)
            # st()
            if const.segmentation_stop:
                self.outputs3D = tf.stop_gradient(outputs3D[-1])
            self.z_vis, self.pos_loss, self.vis_kl_loss,self.mask3d = self.gentree_cond(self.trees)

            inputs2Ddec = self.get_inputs2Ddec([self.z_vis])
            outputs2Ddec = self.get_outputs2Ddec(inputs2Ddec)
            self.pred_view = outputs2Ddec.pred_view


        elif const.SAMPLE_VALUES and tf.math.equal(self.index,2) and const.COND_TREE_TO_IMG:
            self.samples= []
            # st()
            if const.SAMPLE_ANGLES:
                z_prior,pos_loss,vis_kl_loss,mask3d = self.gentree_cond(self.trees)
                for phi_idx in range(const.VV):
                    self.query_phi = tf.convert_to_tensor([phi_idx*const.VDELTA + const.MINV],tf.float32)
                    for theta_idx in range(const.HV):
                        self.query_theta = tf.convert_to_tensor([theta_idx*const.HDELTA + const.MINH],tf.float32)
                        st()
                        inputs2DdecPrior = self.get_inputs2Ddec([z_prior])
                        outputs2DdecPrior = self.get_outputs2Ddec(inputs2DdecPrior)
                        self.samples.append(outputs2DdecPrior.pred_view)
                        self.pred_view_prior = outputs2DdecPrior.pred_view

            else:
                for i in range(2):
                    self.query_theta = np.array([30],np.float32)
                    self.query_phi = np.array([20],np.float32)
                    z_prior,pos_loss,vis_kl_loss,mask3d = self.gentree_cond(self.trees)
                    # st()
                    inputs2DdecPrior = self.get_inputs2Ddec([z_prior])
                    outputs2DdecPrior = self.get_outputs2Ddec(inputs2DdecPrior)
                    self.samples.append(outputs2DdecPrior.pred_view)
            self.pred_view_prior = outputs2DdecPrior.pred_view

        elif const.SAMPLE_VALUES and tf.math.equal(self.index,2):
            self.samples= []
            for i in range(10):
                self.prior_mean, self.prior_var, self.pos_loss,mask3d = self.gentree(self.trees)
                z_prior = self.sampler(self.prior_mean, self.prior_var)
                inputs2DdecPrior = self.get_inputs2Ddec([z_prior])
                outputs2DdecPrior = self.get_outputs2Ddec(inputs2DdecPrior)
                self.samples.append(outputs2DdecPrior.pred_view)
            self.pred_view_prior = outputs2DdecPrior.pred_view


        elif const.RANK_TENSORS and tf.math.equal(self.index,2):
            # st()
            self.samples= []
            inputs2Denc = self.get_inputs2Denc()
            # f1
            outputs2Denc = self.get_outputs2Denc(inputs2Denc)
            inputs3D = self.get_inputs3D(outputs2Denc)
            outputs3D = self.get_outputs3D(inputs3D)
            # output3d from ricson
            outputs3D = [outputs3D[-1]]
            self.prior_mean, self.prior_var, self.pos_loss,mask3d = self.gentree(self.trees)
            z_prior = self.sampler(self.prior_mean, self.prior_var)
            z_prior = self.prior_mean

            self.latent_mean = self.h_mean(outputs3D[0])
            # self.h_var = nn.Conv2d(hiddim, latentdim, 3, 1, 1)
            self.latent_var = self.h_var(outputs3D[0])
            if True:
                self.latent_mean = self.latent_mean * mask3d
                self.latent_var = self.latent_var * mask3d
            # self.z_latent = self.sampler(self.latent_mean, self.latent_var)
            self.z_latent = self.latent_mean
            # reverse_batch = tf.reverse(self.z_latent,[0])
            # z_prior = reverse_batch+z_prior
            
            inputs2DdecPrior = self.get_inputs2Ddec([z_prior])
            outputs2DdecPrior = self.get_outputs2Ddec(inputs2DdecPrior)
            self.pred_view_prior = outputs2DdecPrior.pred_view


            import pickle
            # st()
            inputs2Ddec = self.get_inputs2Ddec([self.z_latent])

            # f4
            outputs2Ddec = self.get_outputs2Ddec(inputs2Ddec)

            self.pred_view = outputs2Ddec.pred_view



            # st()

            self.double_pool_ricson.update(self.latent_mean,self.target,sentences=tmp)

            self.double_pool_tree.update(self.prior_mean,np.ones_like(self.target),sentences=tmp)


        elif const.COND_TREE_TO_IMG:
            # st()
            inputs2Denc = self.get_inputs2Denc()

            # f1
            outputs2Denc = self.get_outputs2Denc(inputs2Denc)


            inputs3D = self.get_inputs3D(outputs2Denc)

            # f2
            outputs3D = self.get_outputs3D(inputs3D)

            if const.ARITH_MODE:
                outputs3D = self.do_arithmetic(outputs3D)
            # st()
            if const.stop:
                self.outputs3D = tf.stop_gradient(outputs3D[-1])
            else:
                self.outputs3D = outputs3D[-1]
            # if const.USE_MEAN:
            #     self.outputs3D = self.h_mean(self.outputs3D)
            # st()
            self.z_vis, self.pos_loss, self.vis_kl_loss,self.mask3d = self.gentree_cond(self.trees)
            
            # self.vis_kl_loss = 0 
            # self.pos_loss = None
            # mask3d = self.getMask3d(self.trees)
            self.outputs3D = self.outputs3D * self.mask3d
            inputs2Ddec = self.get_inputs2Ddec([self.z_vis])
            outputs2Ddec = self.get_outputs2Ddec(inputs2Ddec)
            self.pred_view = outputs2Ddec.pred_view

        else:
            inputs2Denc = self.get_inputs2Denc()

            # f1
            outputs2Denc = self.get_outputs2Denc(inputs2Denc)

            inputs3D = self.get_inputs3D(outputs2Denc)

            # f2
            outputs3D = self.get_outputs3D(inputs3D)

            if const.ARITH_MODE:
                outputs3D = self.do_arithmetic(outputs3D)
            # st()
            outputs3D = [outputs3D[-1]]
            if const.mask3d:
                mask3d = self.getMask3d(self.trees)
                outputs3D = outputs3D*mask3d
            if const.TREE_LOSS:
                self.prior_mean, self.prior_var, self.pos_loss, self.mask3d = self.gentree(self.trees)
                self.latent_mean = self.h_mean(outputs3D[0])
                # self.h_var = nn.Conv2d(hiddim, latentdim, 3, 1, 1)
                self.latent_var = self.h_var(outputs3D[0])

                if const.mask3d:
                    # st()
                    self.latent_mean = self.latent_mean * self.mask3d
                    self.latent_var = self.latent_var * self.mask3d

                self.z_latent = self.sampler(self.latent_mean, self.latent_var)
                self.z_prior = self.sampler(self.prior_mean, self.prior_var,stoch=const.stochasticity)

                if const.DIRECT_TREE_LOSS:
                    inputs2DdecPrior = self.get_inputs2Ddec([self.z_prior])
                    outputs2DdecPrior = self.get_outputs2Ddec(inputs2DdecPrior)
                    self.pred_view_prior = outputs2DdecPrior.pred_view

            # st()
            if not const.EMBEDDING_LOSS_3D:
            # f3
                if const.FREEZE_ENC_DEC:
                    inputs2Ddec = self.get_inputs2Ddec(outputs3D)
                elif const.TREE_LOSS:
                    inputs2Ddec = self.get_inputs2Ddec([self.z_latent])
                else:
                    inputs2Ddec = self.get_inputs2Ddec(outputs3D)

                # f4
                outputs2Ddec = self.get_outputs2Ddec(inputs2Ddec)

                self.pred_view = outputs2Ddec.pred_view




    
    def clean_tree(self, treex):
        for i in range(0, len(treex)):
            self._clean_tree(treex[i])

    def _clean_tree(self, treex):
        for i in range(0, treex.num_children):
            self._clean_tree(treex.children[i])

        if treex.function == 'combine':
            treex.vis_mean = None
            treex.pos_dist = None
        elif treex.function == 'describe':
            treex.vis_mean = None
            treex.pos_dist = None

        elif treex.function == 'layout':
            treex.vis_mean = None
            treex.pos_dist = None


    def do_arithmetic(self, features):
        #0 contains [0,-,-]
        #1 contains [-,-,2]
        #2 contains [0,1,-]
        #3 contains [-,1,2]

        #testing #2 - #0 + #1 = #3
        
        def arith(feature):
            assert const.BS == 4
            feature =  tf.expand_dims(feature[2] - feature[0] + feature[1], axis = 0)
            return tf.tile(feature, (const.BS, 1, 1, 1, 1))
        
        return [arith(feature) for feature in features]
                           
    def get_inputs3D(self, inputs):
        unprojected_features = self.unproject_inputs(inputs)
        aligned_features = self.align_to_first(unprojected_features) # 4 scales, each in 3 views
        # st()
        return self.aggregate(aligned_features)
        
    def get_outputs3D(self, inputs):
        # with tf.compat.v1.variable_scope('3DED'):
        #     return utils.nets.encoder_decoder3D(inputs)
        return self.output3d_f2(inputs)
    
    def get_inputs2Ddec(self, inputs):
        aligned_inputs = self.align_to_query(inputs) #4 scales
        projected_inputs = self.project_inputs(aligned_inputs)

        self.__todump = projected_inputs #should also be postprocessed
        
        # with tf.compat.v1.variable_scope('depthchannel_net'):
        #     return [utils.nets.depth_channel_net_v2(feat)
        #             for feat in projected_inputs]
        return [self.input_decoder2d_f3(feat) for feat in projected_inputs]

    def get_inputs_aligned(self, inputs):
        aligned_inputs = self.align_to_query(inputs) #4 scales
        # projected_inputs = self.project_inputs(aligned_inputs)

        # self.__todump = projected_inputs #should also be postprocessed
        
        # with tf.compat.v1.variable_scope('depthchannel_net'):
        #     return [utils.nets.depth_channel_net_v2(feat)
        #             for feat in projected_inputs]
        return aligned_inputs

    def get_outputs2Ddec(self, inputs):
        if const.GQN3D_CONVLSTM:
            return self.conv_lstm_decoder_f4(inputs, None, self.target)
        else:
            raise Exception('need to update this with pred_view and embed attributes')
            return utils.nets.decoder2D(inputs, False)

    def convlstm_decoder(self, inputs):
        #we get feature maps of different resolution as input
        #downscale last and concat with second last

        inputs = [utils.tfutil.poolorunpool(x, 16) for x in inputs]
        net = tf.concat(inputs, axis = -1)
        # net = slim.conv2d(net, 256, [3, 3])
        net = tf.keras.layers.Conv2D(256, 3)(net)

        dims = 3+const.EMBEDDING_LOSS * const.embedding_size
        out, extra = utils.fish_network.make_lstmConv(
            net,
            None,
            self.target,
            [['convLSTM', const.CONVLSTM_DIM, dims, const.CONVLSTM_STEPS, const.CONVLSTM_DIM]], 
            stochastic = const.GQN3D_CONVLSTM_STOCHASTIC,
            weight_decay = 1E-5,
            is_training = const.mode == 'train',
            reuse = False,
            output_debug = False,
        )

        out_img = utils.tfutil.tanh01(out[:,:,:,:3])
        embedding = out[:,:,:,3:] if const.EMBEDDING_LOSS else tf.constant(0.0, dtype = tf.float32)

        return Munch(pred_view = out_img, embedding = embedding, kl = extra['kl_loss'])

    def unproject_inputs(self, inputs):
        
        def stack_unproject_unstack(_inputs):
            _inputs = tf.stack(_inputs, axis = 0)
            _inputs = tf.map_fn(
                lambda x: utils.nets.unproject(x, False),
                _inputs, parallel_iterations = 1
            )
            _inputs = tf.unstack(_inputs, axis = 0)
            return _inputs
        
        return [stack_unproject_unstack(inp) for inp in inputs]

    def project_inputs(self, inputs):
        return [
            utils.voxel.transformer_postprocess(
                utils.voxel.project_voxel(feature)
            )
            for feature in inputs
        ]

    def translate_multiple(self, dthetas, phi1s, phi2s, voxs):
        dthetas = tf.stack(dthetas, axis = 0)
        phi1s = tf.stack(phi1s, 0)
        phi2s = tf.stack(phi2s, 0)
        voxs = tf.stack(voxs, 0)

        f = lambda x: utils.voxel.translate_given_angles(*x)
        out = tf.map_fn(f, [dthetas, phi1s, phi2s, voxs], dtype = tf.float32)
        return tf.unstack(out, axis = 0)
    
    def align_to_first(self, features):
        return [self.align_to_first_single(feature) for feature in features]

    def align_to_query(self, features):
        return [self.align_to_query_single(feature) for feature in features]
    
    def align_to_first_single(self, feature):
        #3 features from different views
        # dthetas = [self.thetas[0] - theta for theta in self.thetas]
        # phi1s = self.phis
        # phi2s = [self.phis[0] for _ in self.phis]

        dthetas = [-theta for theta in self.thetas]
        phi1s = self.phis
        phi2s = [[0.0]*const.BS for _ in self.phis]
        return self.translate_multiple(dthetas, phi1s, phi2s, feature)
    
    def align_to_query_single(self, feature):
        #a single feature from view 0
        # dthetas = [self.query_theta - self.thetas[0]]
        # phi1s = [self.phis[0]]
        # phi2s = [self.query_phi]

        dthetas = [self.query_theta]
        phi1s = [[0.0]*const.BS]
        phi2s = [self.query_phi]
        return self.translate_multiple(dthetas, phi1s, phi2s, [feature])[0]

    def build_vis(self):
        super().build_vis()
        # self.vis.dump = self.tensors_to_dump

        if const.EMBEDDING_LOSS:
            self.vis.embed = tf.concat([self.emb_pca,self.emb_pred_pca], axis = 2)
        
class GQN2D(GQN_with_2dencoder):
    
    def predict(self):
        self.tensors_to_dump = {}
        
        inputs2Denc = self.get_inputs2Denc()
        outputs2Denc = self.get_outputs2Denc(inputs2Denc)
        encoded = self.aggregate(outputs2Denc)
        encoded = self.add_query(encoded)
        decoded = utils.nets.decoder2D(encoded[::-1], False)
        self.pred_view = decoded

    def add_query(self, encoded):
        pose_info = tf.concat(self.poses + [self.query_pose], axis = 1)
        encoded = [utils.tfutil.add_feat_to_img(feat, pose_info) for feat in encoded]

        encoded = [
            tf.keras.layers.Conv2D(dims, 1)(feat)
            for (feat, dims) in zip(encoded, [64, 128, 256, 512])
        ]

        return encoded

    def setup_data(self):
        super().setup_data()
        
        thetas_r = list(map(utils.utils.radians, self.thetas))
        phis_r = list(map(utils.utils.radians, self.phis))
        query_theta_r = utils.utils.radians(self.query_theta)
        query_phi_r = utils.utils.radians(self.query_phi)

        foo = lambda theta, phi: [tf.cos(theta), tf.sin(theta), tf.cos(phi), tf.sin(phi)]
        bar = lambda theta, phi: tf.stack(foo(theta, phi), axis = 1)
        self.poses = [bar(*x) for x in zip(thetas_r, phis_r)]
        self.query_pose = bar(query_theta_r, query_phi_r)

class GQN2Dtower(GQNBase):
    '''wraps around fish_network.MultiCameraGQN'''

    def predict(self):

        #encoder
        images = list(self.query.context.frames) + [self.target]
        inputs = {
            'images': images,
            'undistorted_images': images,
            'cam_posrot': list(self.query.context.cameras) + [self.query.query_camera],
            'label_names': ['view0', 'view1', 'view2', 'view3'],
            'target_shapes': [(const.BS, const.H, const.W, 3)]*4,
        }
        
        encoded = utils.fish_network.MultiCameraGQN(
            inputs, output_dim = 3, is_training = const.mode == 'train'
        ).camera_cnns
        encoded = sum(encoded)

        if const.ARITH_MODE:
            encoded = self.do_arithmetic(encoded)        
        
        #decoder
        camera = tf.reshape(self.query.query_camera, (const.BS, 1, 1, 2))
        
        out, extra = utils.fish_network.make_lstmConv(
            encoded,
            camera,
            self.target,
            [['convLSTM', const.CONVLSTM_DIM, 3, const.CONVLSTM_STEPS, const.CONVLSTM_DIM]], 
            stochastic = False,
            weight_decay = 1E-5,
            is_training = const.mode == 'train',
            reuse = False,
            output_debug = False,
        )

        out = utils.tfutil.tanh01(out)
        out.loss = extra['kl_loss']

        self.pred_view = out
        

    def do_arithmetic(self, features):
        assert const.BS == 4
        feature = tf.expand_dims(features[2] - features[0] + features[1], axis = 0)
        return tf.tile(feature, (const.BS, 1, 1, 1))
