import inputs
import nets
import constants as const
import os.path as path
from fig import Config
import utils
import tensorflow as tf
from pprint import pprint
import glob
import os
import sys
from ipdb import set_trace as st
import IPython
ip = IPython.embed
from tensorflow.python.keras import backend
import pickle

old_names = ['out_enc_2d/sequential_26/conv2d_15/kernel:0', 'out_enc_2d/sequential_26/conv2d_15/bias:0', 'out_enc_2d/sequential_26/batch_normalization_v2_8/gamma:0', 'out_enc_2d/sequential_26/batch_normalization_v2_8/beta:0', 'out_enc_2d/sequential_26/conv2d_16/kernel:0', 'out_enc_2d/sequential_26/conv2d_16/bias:0', 'out_enc_2d/sequential_26/batch_normalization_v2_9/gamma:0', 'out_enc_2d/sequential_26/batch_normalization_v2_9/beta:0', 'out_enc_2d/sequential_27/conv2d_17/kernel:0', 'out_enc_2d/sequential_27/conv2d_17/bias:0', 'out_enc_2d/sequential_27/batch_normalization_v2_10/gamma:0', 'out_enc_2d/sequential_27/batch_normalization_v2_10/beta:0', 'out_enc_2d/sequential_27/conv2d_18/kernel:0', 'out_enc_2d/sequential_27/conv2d_18/bias:0', 'out_enc_2d/sequential_27/batch_normalization_v2_11/gamma:0', 'out_enc_2d/sequential_27/batch_normalization_v2_11/beta:0', 'out_enc_2d/sequential_28/conv2d_19/kernel:0', 'out_enc_2d/sequential_28/conv2d_19/bias:0', 'out_enc_2d/sequential_28/batch_normalization_v2_12/gamma:0', 'out_enc_2d/sequential_28/batch_normalization_v2_12/beta:0', 'out_enc_2d/sequential_28/conv2d_20/kernel:0', 'out_enc_2d/sequential_28/conv2d_20/bias:0', 'out_enc_2d/sequential_28/batch_normalization_v2_13/gamma:0', 'out_enc_2d/sequential_28/batch_normalization_v2_13/beta:0', 'out_enc_2d/sequential_29/conv2d_21/kernel:0', 'out_enc_2d/sequential_29/conv2d_21/bias:0', 'out_enc_2d/sequential_29/batch_normalization_v2_14/gamma:0', 'out_enc_2d/sequential_29/batch_normalization_v2_14/beta:0', 'out_enc_2d/sequential_29/conv2d_22/kernel:0', 'out_enc_2d/sequential_29/conv2d_22/bias:0', 'out_enc_2d/sequential_29/batch_normalization_v2_15/gamma:0', 'out_enc_2d/sequential_29/batch_normalization_v2_15/beta:0', 'out_3d/sequential_30/conv3d_28/kernel:0', 'out_3d/sequential_30/conv3d_28/bias:0', 'out_3d/sequential_31/conv3d_29/kernel:0', 'out_3d/sequential_31/conv3d_29/bias:0', 'out_3d/sequential_32/conv3d_30/kernel:0', 'out_3d/sequential_32/conv3d_30/bias:0', 'out_3d/sequential_33/conv3d_31/kernel:0', 'out_3d/sequential_33/conv3d_31/bias:0', 'out_3d/sequential_34/conv3d_transpose_4/kernel:0', 'out_3d/sequential_34/conv3d_transpose_4/bias:0', 'out_3d/sequential_35/conv3d_transpose_5/kernel:0', 'out_3d/sequential_35/conv3d_transpose_5/bias:0', 'out_3d/sequential_36/conv3d_transpose_6/kernel:0', 'out_3d/sequential_36/conv3d_transpose_6/bias:0', 'out_3d/sequential_37/conv3d_transpose_7/kernel:0', 'out_3d/sequential_37/conv3d_transpose_7/bias:0', 'in_dec_2d/conv2d_23/kernel:0', 'in_dec_2d/conv2d_23/bias:0', 'conv_lstm_dec/conv2d_24/kernel:0', 'conv_lstm_dec/conv2d_24/bias:0', 'conv_lstm_dec/conv_lstm_1/conv_lstm_full/convLSTM_input_c/kernel:0', 'conv_lstm_dec/conv_lstm_1/conv_lstm_full/convLSTM_input_c/bias:0', 'conv_lstm_dec/conv_lstm_1/conv_lstm_full/convLSTM_o_/kernel:0', 'conv_lstm_dec/conv_lstm_1/conv_lstm_full/convLSTM_o_/bias:0', 'conv_lstm_dec/conv_lstm_1/conv_lstm_full/convLSTM_final_/kernel:0', 'conv_lstm_dec/conv_lstm_1/conv_lstm_full/convLSTM_final_/bias:0']
new_names = ['out_enc_2d/sequential/conv2d/kernel:0', 'out_enc_2d/sequential/conv2d/bias:0', 'out_enc_2d/sequential/batch_normalization_v2/gamma:0', 'out_enc_2d/sequential/batch_normalization_v2/beta:0', 'out_enc_2d/sequential/conv2d_1/kernel:0', 'out_enc_2d/sequential/conv2d_1/bias:0', 'out_enc_2d/sequential/batch_normalization_v2_1/gamma:0', 'out_enc_2d/sequential/batch_normalization_v2_1/beta:0', 'out_enc_2d/sequential_1/conv2d_2/kernel:0', 'out_enc_2d/sequential_1/conv2d_2/bias:0', 'out_enc_2d/sequential_1/batch_normalization_v2_2/gamma:0', 'out_enc_2d/sequential_1/batch_normalization_v2_2/beta:0', 'out_enc_2d/sequential_1/conv2d_3/kernel:0', 'out_enc_2d/sequential_1/conv2d_3/bias:0', 'out_enc_2d/sequential_1/batch_normalization_v2_3/gamma:0', 'out_enc_2d/sequential_1/batch_normalization_v2_3/beta:0', 'out_enc_2d/sequential_2/conv2d_4/kernel:0', 'out_enc_2d/sequential_2/conv2d_4/bias:0', 'out_enc_2d/sequential_2/batch_normalization_v2_4/gamma:0', 'out_enc_2d/sequential_2/batch_normalization_v2_4/beta:0', 'out_enc_2d/sequential_2/conv2d_5/kernel:0', 'out_enc_2d/sequential_2/conv2d_5/bias:0', 'out_enc_2d/sequential_2/batch_normalization_v2_5/gamma:0', 'out_enc_2d/sequential_2/batch_normalization_v2_5/beta:0', 'out_enc_2d/sequential_3/conv2d_6/kernel:0', 'out_enc_2d/sequential_3/conv2d_6/bias:0', 'out_enc_2d/sequential_3/batch_normalization_v2_6/gamma:0', 'out_enc_2d/sequential_3/batch_normalization_v2_6/beta:0', 'out_enc_2d/sequential_3/conv2d_7/kernel:0', 'out_enc_2d/sequential_3/conv2d_7/bias:0', 'out_enc_2d/sequential_3/batch_normalization_v2_7/gamma:0', 'out_enc_2d/sequential_3/batch_normalization_v2_7/beta:0', 'out_3d/sequential_4/conv3d/kernel:0', 'out_3d/sequential_4/conv3d/bias:0', 'out_3d/sequential_5/conv3d_1/kernel:0', 'out_3d/sequential_5/conv3d_1/bias:0', 'out_3d/sequential_6/conv3d_2/kernel:0', 'out_3d/sequential_6/conv3d_2/bias:0', 'out_3d/sequential_7/conv3d_3/kernel:0', 'out_3d/sequential_7/conv3d_3/bias:0', 'out_3d/sequential_8/conv3d_transpose/kernel:0', 'out_3d/sequential_8/conv3d_transpose/bias:0', 'out_3d/sequential_9/conv3d_transpose_1/kernel:0', 'out_3d/sequential_9/conv3d_transpose_1/bias:0', 'out_3d/sequential_10/conv3d_transpose_2/kernel:0', 'out_3d/sequential_10/conv3d_transpose_2/bias:0', 'out_3d/sequential_11/conv3d_transpose_3/kernel:0', 'out_3d/sequential_11/conv3d_transpose_3/bias:0', 'in_dec_2d/conv2d_8/kernel:0', 'in_dec_2d/conv2d_8/bias:0', 'conv_lstm_dec/conv2d_9/kernel:0', 'conv_lstm_dec/conv2d_9/bias:0', 'conv_lstm_dec/conv_lstm/conv_lstm_full/convLSTM_input_c/kernel:0', 'conv_lstm_dec/conv_lstm/conv_lstm_full/convLSTM_input_c/bias:0', 'conv_lstm_dec/conv_lstm/conv_lstm_full/convLSTM_o_/kernel:0', 'conv_lstm_dec/conv_lstm/conv_lstm_full/convLSTM_o_/bias:0', 'conv_lstm_dec/conv_lstm/conv_lstm_full/convLSTM_final_/kernel:0', 'conv_lstm_dec/conv_lstm/conv_lstm_full/convLSTM_final_/bias:0']
old_new_mapping = {old: new for (old, new) in zip(old_names, new_names)}


class Mode:
    def __init__(self, data_name, ops_name):
        self.data_name = data_name
        self.ops_name = ops_name

        
class Model:
    def __init__(self, net, modes = None):
        self.net = net
        self.data_selector = self.net.data_selector
        self.non_trainable_weight_dict = {}
        
        self.modes = modes
        if modes is None:
            self.modes = {
                'train': Mode('train', 'train_run'),
                'valt': Mode('train', 'val_run'),
                # 'valv': Mode('val', 'val_run'),
                'test': Mode('test', 'test_run'),
            }
            

        if not const.eager:
            self.net.go() #build the graph here
        
    def get_ops(self, mode,kl_coeff):
        if const.eager:
            self.net(self.index_for_mode(mode),kl_coeff, self.non_trainable_weight_dict)
        return getattr(self.net, self.modes[mode].ops_name)
    
    def get_data_name(self, mode):
        return self.modes[mode].data_name

    def index_for_mode(self, mode):
        input_collection_to_number = {'train': 0, 'val': 1, 'test': 2}
        data_name = self.get_data_name(mode)
        return input_collection_to_number[data_name]
    
    def fd_for_mode(self, mode):
        assert not const.eager
        if self.data_selector is None:
            return None
        else:
            return {self.data_selector: self.index_for_mode(mode)}

    def run(self, mode, sess = None,kl_coeff=None):
        ops = self.get_ops(mode,kl_coeff)
        if const.eager:
            return utils.utils.apply_recursively(ops, utils.tfutil.read_eager_tensor)
        else:

            # sess.run(self.net.input, feed_dict = self.fd_for_mode(mode))
            # import ipdb; ipdb.set_trace()
            # st()
            fd = self.fd_for_mode(mode)
            if const.LOSS_GAN:
                if mode == "train":
                    # st()
                    dis_adam,gen_adam = ops["opt"]
                    for _ in range(0, const.gan_d_iters):
                        sess.run(dis_adam, feed_dict =fd )
                    sess.run(gen_adam, feed_dict = fd)
                    return sess.run({"summary":ops["summary"]}, feed_dict = fd)
            return sess.run(ops, feed_dict = fd)



class PersistentModel(Model):
    def __init__(self, model_, ckpt_dir, modes = None):
        self.ckpt_dir = ckpt_dir
        utils.utils.ensure(ckpt_dir)
        super(PersistentModel, self).__init__(model_, modes)

        # self.initsaver()

    def initsaver(self):
        self.savers = {}
        parts = self.net.weights
        for partname in parts:
            partweights = parts[partname][1]
            if partweights:
                self.savers[partname] = tf.compat.v1.train.Saver(partweights)

    def save(self, sess, name, step):
        config = Config(name, step)
        # parts = self.net.weights
        savepath = path.join(self.ckpt_dir, name + '_' +  str(step))
        utils.utils.ensure(savepath)
        # st()
        self.net.save_weights(os.path.join(savepath, 'weights.h5'))
        # for partname in parts:
        #     partpath = path.join(savepath, partname)
        #     utils.utils.ensure(partpath)
        #     partscope, weights = parts[partname]

        #     if not weights: #nothing to do
        #         continue
            
        #     partsavepath = path.join(partpath, 'X')

        #     saver = self.savers[partname]
        #     saver.save(sess, partsavepath, global_step=step)

        #     config.add(partname, partscope, partpath)
        config.save()

    def load(self, sess, name):
        old_weights_dict = {}
        if const.load_name:
            name = const.load_name
        config = Config(name)
        config.load()
        step = config.step
        
        # if you want to transfer a custom part of the network to the other network then run that to exp to transfer
        # with save_custom as true and select the network and save the parts as pickle

        # while loading just follow the same format with st

        # will make a better way of doing it later....


        if const.load_name and const.FREEZE_ENC_DEC:
            # st()
            if const.save_custom:
                exp_name = const.exp_name
                load_name = const.load_name
                const.set_experiment("unset_everything")
                const.set_experiment(load_name)
                const.LOAD_VAL=True
                # # st()
                old_model_obj = GQN3D()
                old_model_obj.net(self.index_for_mode('train'), 0,loading=True)
                loadpath = os.path.join('ckpt', const.load_name + '_' + str(step), 'weights.h5')
                # st()
                old_model_obj.net.load_weights(loadpath)
                st()
                val = {i.name:i.get_weights()  for i in old_model_obj.net.layers[:4]}
                pickle.dump(val,open("encoder_weights.p","wb"))
            else:
                encoder_weights = pickle.load(open("encoder_weights.p","rb"))
                self.net(self.index_for_mode('test'),0,loading=True)
                step = 7777
                st()
                for i,layer in enumerate(self.net.layers[:4]):
                    layer.set_weights(encoder_weights[layer.name+"_1"])

                st()
                print("loaded")
            # self.net(self.index_for_mode('test'), 0, self.non_trainable_weight_dict)
            # tuples = []
            # for model_weights in self.net.weights:
            #     if model_weights.name in old_weights_dict:
            #         tuples.append((model_weights, old_weights_dict[model_weights.name]))
            # backend.batch_set_value(tuples)



        else:
            loadpath = os.path.join('ckpt', const.load_name + '_' + str(step), 'weights.h5')
            # if const.segmentation and const.loading:
            #     loadpath = os.path.join('ckpt', const.load_name + '_' + str(125000), 'weights.h5')
            # const.LOAD_VAL = True
            # st(
            for i in range(1):
                self.net(self.index_for_mode('train'), 0, loading=const.loading)
            # const.LOAD_VAL = False

            self.net.load_weights(loadpath)

            # tuples = []
            # for model_weights in self.net.weights:
            #     if model_weights.name in old_weights_dict:
            #         print(model_weights.numpy().shape)
            #         print(old_weights_dict[model_weights.name].numpy().shape)
            #         tuples.append((model_weights, old_weights_dict[model_weights.name]))
            #         counter += 1
            #         st()
            #     else:
            #         tuples.append((model_weights, model_weights))
                
            # backend.batch_set_value(tuples)

        # for partname in config.dct:
        #     partscope, partpath = config.dct[partname]

        #     if partname not in parts:
        #         raise Exception("cannot load, part %s not in model" % partpath)

        #     ckpt = tf.train.get_checkpoint_state(partpath)
        #     if not ckpt:
        #         raise Exception("checkpoint not found? (1)")
        #     elif not ckpt.model_checkpoint_path:
        #         raise Exception("checkpoint not found? (2)")
        #     loadpath = ckpt.model_checkpoint_path

        #     scope, weights = parts[partname]

        #     if not weights: #nothing to do
        #         continue

        #     weights = {utils.utils.exchange_scope(weight.op.name, scope, partscope): weight
        #                for weight in weights}

        #     saver = tf.compat.v1.train.Saver(weights)
        #     # st()
        #     saver.restore(sess, loadpath)
        return step

    
class MultiViewReconstruction(PersistentModel):
    def __init__(self):
        input_ = inputs.MultiViewReconstructionInput()
        net = nets.MultiViewReconstructionNet(input_)
        super().__init__(net, const.ckpt_dir)

        
class MultiViewQuery(PersistentModel):
    def __init__(self):
        input_ = inputs.MultiViewReconstructionInput()
        net = nets.MultiViewQueryNet(input_)
        
        super().__init__(net, const.ckpt_dir)

class GQNBaseModel(PersistentModel):
    def __init__(self, net_cls):
        if const.DEEPMIND_DATA:
            input_ = inputs.GQNInput()
        else:
            input_ = inputs.GQNShapenet()
        net = net_cls(input_)
        super().__init__(net, const.ckpt_dir)

        if const.generate_views:
            self.modes['gen'] = Mode('test', 'gen_run')
        
class GQN2D(GQNBaseModel):
    def __init__(self):
        return super().__init__(nets.GQN2D)

class GQN2Dtower(GQNBaseModel):
    def __init__(self):
        return super().__init__(nets.GQN2Dtower)
        
class GQN3D(GQNBaseModel):
    def __init__(self):
        return super().__init__(nets.GQN3D)

class TestGQN(GQNBaseModel):
    def __init__(self):
        return super().__init__(nets.TestInput)

class MnistAE(PersistentModel):
    def __init__(self):
        input_ = inputs.MNISTInput()
        net = nets.MnistAE(input_)
        super().__init__(net, const.ckpt_dir)

