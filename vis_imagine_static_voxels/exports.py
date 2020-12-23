import numpy as np
import constants as const
from scipy.misc import imsave
from os.path import join, exists
import os
import tensorflow as tf
import utils
from threading import Thread
from pprint import pprint
import pickle
import ipdb
import math
from utils import binvox_rw


st = ipdb.set_trace

# if const.eager:
#     # import tensorflow.contrib.summary as tcs
#     # summary_constructor = tcs.create_file_writer
#     summary_constructor = tf.summary.FileWriter
# else:
#     summary_constructor = tf.summary.FileWriter


class Export(object):
    def __init__(self, location = None):
        self.location = location
        self.summary_constructor = tf.summary.create_file_writer(const.tb_dir)

    def process(self, items, mode, step):
        if const.DEBUG_EXPORTS:
            self.process_(items, mode, step)
        else:
            t = Thread(target=lambda: self.process_(items, mode, step))
            t.start()

    def process_(self, items, mode, step):
        nyi()

    def finish(self):
        pass


class Evaluator(Export):
    def __init__(self, names):
        self.names = names
        self.c = 0
        self.sums = {name: 0.0 for name in self.names}
        self.sumsqs = {name: 0.0 for name in self.names}
        
        super(Evaluator, self).__init__()
    
    def process_(self, items, _, __):
        for name in self.names:
            assert name in items, 'metric %s not in items' % name
            self.sums[name] += items[name]
            self.sumsqs[name] += items[name]**2
            
        self.c += 1

    def finish(self):
        self.averages = {name: self.sums[name]/float(self.c) for name in self.names}

        #variance = e[x^2]-e[x]^2
        self.stds = {
            name: math.sqrt(self.sumsqs[name]/float(self.c) - self.averages[name]**2)
            for name in self.names
        }
        
        for metric in self.averages:
            print('METRIC: %s' % metric)
            print(self.averages[metric], '+/-', self.stds[metric])


class TB(Export):
    def __init__(self, valmodename='val', 
                 trainmodename='train', 
                 testmodename = 'test', 
                 ignoremodes=[]):

        tb_dir = join(const.tb_dir, const.exp_name)
        utils.utils.ensure(tb_dir)
        super(TB, self).__init__(tb_dir)

        self.valmodename = valmodename
        self.trainmodename = trainmodename
        self.testmodename = testmodename
        self.ignoremodes = ignoremodes

        # self.train_writer = summary_constructor(join(self.location, 'train'))
        # self.test_writer = summary_constructor(join(self.location, 'test'))
        # self.val_writer = summary_constructor(join(self.location, 'val'))

    def process_(self, items, mode, step):
        if const.eager:
            print('skipping tensorboard')
            with self.summary_constructor.as_default():
                if mode == self.valmodename:
                    tf.summary.scalar('val', items, step=step)
                elif mode == self.trainmodename:
                    tf.summary.scalar('train', items, step=step)
                elif mode == self.testmodename:
                    tf.summary.scalar('test', items, step=step)
                elif mode in self.ignoremodes:
                    pass
                else:
                    raise Exception('unrecognized mode')


class Vis(Export):
    def __init__(self):
        vis_dir = join(const.vis_dir, const.exp_name)
        utils.utils.ensure(vis_dir)
        super(Vis, self).__init__(vis_dir)
        self.suffix = None
        self.visfn = None


    def preprocess(self, items):
        return items

    def process_(self, items, mode, step):
        items = self.preprocess(items)
        step_str = str(step).zfill(6)
        self.visfn["segmentation_masks"] = self.save_voxel

        # st()
        if mode != "test":
            filename = items.pop("file_name",None)
            for name in items:
                obj = items[name]
                if isinstance(obj, np.ndarray):
                    index = int(const.BS > 1)
                    #take just one item from our batch, but not the first, to catch edge cases!
                    obj = np.copy(obj[0])
                suffix = self.suffix[name]
                path_head = join(self.location, mode)
                utils.utils.ensure(path_head)
                save_path = join(path_head, '%s_%s.png' % (step_str, suffix))
                fn = self.visfn[name]
                fn(save_path, obj)
            self.postprocess(mode, step)
        else:
            # BATCH SIZE SHOULD ALWAYS BE ONE FOR NOW
            filename = items.pop("file_name",None)
            step_str = str(step).zfill(6)
            path_head = join(self.location, mode)
            if const.SAMPLE_VALUES:
                camloc = items.pop("cam_loc",None).decode()
                folderName = items.pop("sentence",None).decode()
                path_head = join(path_head,folderName)
            assert filename != None
            assert const.BS == 1
            for name in items:
                obj = items[name]
                if isinstance(obj, np.ndarray):
                    index = int(const.BS > 1)
                    #take just one item from our batch, but not the first, to catch edge cases!
                    obj = np.copy(obj[index])
                
                suffix = self.suffix[name]
                utils.utils.ensure(path_head)
                if name== "samples":
                    # st()
                    for i,obj_i in enumerate(obj):
                        save_path = join(path_head, 'samples_%s_%s_%s.png' % (camloc,i, suffix))
                        fn = self.visfn[name]
                        fn(save_path, obj_i)
                else:
                    save_path = join(path_head, '%s_%s.png' % (step_str, suffix))
                    fn = self.visfn[name]
                    fn(save_path, obj)
            self.postprocess(mode, step)

    def postprocess(self, mode, step):
        pass

    def __add__(self, spec):
        newsuffix = self.suffix.copy()
        newsuffix.update(spec.suffix)
        newvisfn = self.visfn.copy()
        newvisfn.update(spec.visfn)
        v = Vis()
        v.suffix = newsuffix
        v.visfn = visfn
        return v

class AEVis(Vis):
    def __init__(self):
        super().__init__()

        self.g = lambda pth, x: utils.img.imsave01(pth, utils.img.flatimg(x))
        
        self.suffix = {k: k for k in ['in', 'out']}
        self.visfn = {
            'in': self.g, 
            'out': self.g,
        }

class MultiViewVis(Vis):

    def g(self, pth, obj):
        utils.img.imsavegrid01(pth, obj, 3.0, 5.0)

    def save_voxel(self,filename,voxel_, THRESHOLD=0.95):
        pickle.dump(voxel_,open(filename,"wb"))
        # S1 = voxel_.shape[2]
        # S2 = voxel_.shape[1]
        # S3 = voxel_.shape[0]
        # # st()
        # filename = filename.replace('png', 'binvox')
        # binvox_obj = binvox_rw.Voxels(
        #     np.transpose(voxel_, [2, 1, 0]) >= THRESHOLD,
        #     dims = [S1, S2, S3],
        #     translate = [0.0, 0.0, 0.0],
        #     scale = 1.0,
        #     axis_order = 'xyz'
        # )   

        # with open(filename, "wb") as f:
        #     binvox_obj.write(f)
    
    def save_vox(self, pth, vox):

        S = vox.shape[1]

        THRESHOLD = 0.5

        if False:
            print('vox stats')
            print(np.min(vox))
            print(np.mean(vox))
            print(np.max(vox))
            print(np.std(vox))

        if len(vox.shape) > 3:
            vox = np.squeeze(vox, axis = 3)
        
        #recall that x and z are messed up so
        vox = np.transpose(vox, (2, 1, 0))
        
        utils.utils.check_numerics(vox)
        binvox_obj = utils.binvox_rw.Voxels(
            vox > THRESHOLD,
            dims = [S, S, S],
            translate = [0.0, 0.0, 0.0],
            scale = 1.0,
            axis_order = 'xyz'
        )
        pth = pth.replace('png', 'binvox')
        with open(pth, 'wb') as f:
            binvox_obj.write(f)

    def save_seg_mask(self, pth, vox):

        S = vox.shape[1]

        THRESHOLD = 0.5

        if False:
            print('vox stats')
            print(np.min(vox))
            print(np.mean(vox))
            print(np.max(vox))
            print(np.std(vox))

        if len(vox.shape) > 3:
            vox = np.squeeze(vox, axis = 3)
        
        #recall that x and z are messed up so
        vox = np.transpose(vox, (2, 1, 0))
        
        utils.utils.check_numerics(vox)
        binvox_obj = utils.binvox_rw.Voxels(
            vox > THRESHOLD,
            dims = [S, S, S],
            translate = [0.0, 0.0, 0.0],
            scale = 1.0,
            axis_order = 'xyz'
        )
        pth = pth.replace('png', 'binvox')
        with open(pth, 'wb') as f:
            binvox_obj.write(f)

    def save_feats(self, pth, feats):
        np.save(pth.replace('png', 'npy'), feats)


class MultiViewReconstructionVis(MultiViewVis):
    def __init__(self):
        super().__init__()

        self.suffix = {
            k: k for k in
            ['images', 'depths', 'pred_depths', 'pred_masks', 'masks', 'pred_vox',
             'pred_feats', 'seg_obj1', 'seg_obj2',]
        }

        self.visfn = {
            'images': utils.img.imsave01,
            'depths': lambda pth, obj: utils.img.imsave01(pth, utils.img.flatimg(obj), 3.0, 5.0),
            'pred_depths': lambda pth, obj: utils.img.imsave01(pth, utils.img.flatimg(obj), 3.0, 5.0),
            'masks': lambda pth, obj: utils.img.imsave01(pth, utils.img.flatimg(obj)),
            'pred_masks': lambda pth, obj: utils.img.imsave01(pth, utils.img.flatimg(obj)),
            'pred_vox': self.save_vox,
            'seg_obj1': self.save_vox,
            'seg_obj2': self.save_vox,            
            'pred_feats': self.save_feats,
            "segmentation_masks": self.save_voxel
        }


class MultiViewQueryVis(MultiViewVis):
    def __init__(self):
        super().__init__()

        if const.DIRECT_TREE_TO_IMG:
            self.suffix = {
                k: k for k in
                ['input_views', 'query_views', 'pred_view_prior', 'dump', 'gen_views', 'embed',"segmentation_masks"]
            }            
        elif const.TREE_LOSS and const.DIRECT_TREE_LOSS or const.SAMPLE_VALUES or const.RANK_TENSORS:
            self.suffix = {
                k: k for k in
                ['input_views', 'query_views', 'pred_view_prior', 'pred_views', 'dump', 'gen_views', 'embed', 'samples','ranking',"segmentation_masks"]
            }
        else:
            self.suffix = {
                k: k for k in
                ['input_views', 'query_views', 'pred_views', 'dump', 'gen_views', 'embed',"segmentation_masks"]
            }

        self.visfn = {k : utils.img.imsave01 for k in self.suffix}
        self.visfn['dump'] = self.dump_tensor
        self.visfn['gen_views'] = self.save_gen_views
        self.visfn["segmentation_masks"] = self.save_voxel
        # self.visfn['embed'] = self.vis_embed

    def vis_embed(self, pth, x):
        from sklearn.decomposition import PCA
        def pcaembed(img, clip = True):
            H, W, K = np.shape(img)
            pixelskd = np.reshape(img, (H*W, K))
            P = PCA(3)
            #only fit a small subset for efficiency
            P.fit(np.random.permutation(pixelskd)[:65536]) 
            pixels3d = P.transform(pixelskd)
            out_img = np.reshape(pixels3d, (H, W, 3))

            if clip:
                std = np.std(out_img)
                mu = np.mean(out_img)
                out_img = np.clip(out_img, mu-std*2, mu+std*2)

            out_img -= np.min(out_img)
            out_img /= np.max(out_img)
            out_img *= 255
            return out_img

        pcavis = pcaembed(x)
        H, W, _ = pcavis.shape
        pcavis = np.reshape(pcavis, (H, W, -1))
        imsave(pth, pcavis)
        
        
    def dump_tensor(self, pth, x):
        if not const.DUMP_TENSOR:
            return
        # st()
        part_names = pth.split("/")
        folder_name = part_names[:-1]
        filename = part_names[-1]
        updated_foldername = "/".join(folder_name + ["dump"])
        utils.utils.ensure(updated_foldername)
        pth = os.path.join(updated_foldername,filename)
        pth = pth.replace('png', 'npy')
        new = pth[:-9] + pth[-3:]
        # st()
        np.save(new,x["o3"][0][0])
        # with open(pth, 'wb') as f:
        #     pickle.dump(x, f)

    def save_gen_views(self, pth, x):
        folder = pth.replace('.png', '')
        
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        for i, img in enumerate(x):
            pth = os.path.join(folder, '%d.png' % i)
            utils.img.imsave01(pth, img)

        params = (const.GEN_FRAMERATE, folder, "%d", folder)
        os.system("ffmpeg -r %d -i %s/%s.png -vcodec mpeg4 -y %s.mp4" % params)
        os.system("ffmpeg -r %d -i %s/%s.png -y %s.gif" % params)
