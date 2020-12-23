
from lib_classes.modules.utils_basic import *
from lib_classes.modules import utils_improc
import constants as const
import ipdb
st = ipdb.set_trace
from sklearn.decomposition import PCA


class SimpleNetBlock(tf.keras.Model):
    def __init__(self,out_chans, blk_num,istrain):
        super(SimpleNetBlock, self).__init__()

        self.out_chans = out_chans
        self.istrain = istrain
        self.blk_num = blk_num

        
        self.conv2d = tf.keras.layers.Conv2D(out_chans*(2**self.blk_num) ,kernel_size=3, strides=2, activation=tf.nn.leaky_relu,\
         padding='VALID',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3))
        self.batchnorm = tf.keras.layers.BatchNormalization()
        
        self.conv2d_1 = tf.keras.layers.Conv2D(out_chans*(2**self.blk_num) ,kernel_size=3, dilation_rate=2, activation=tf.nn.leaky_relu,\
         padding='VALID',kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3))
        self.batchnorm_1 = tf.keras.layers.BatchNormalization()

        self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(out_chans, kernel_size=[4,4], strides=2,padding='SAME',\
            activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3))
        self.batchnorm_2 = tf.keras.layers.BatchNormalization()

    def call(self,feat,blk_num):
        feat = tf.pad(tensor=feat, paddings=[[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
        feat = self.conv2d(feat)
        print_shape(feat)
        feat = self.batchnorm(feat, self.istrain)
        
        feat = tf.pad(tensor=feat, paddings=[[0,0],[2,2],[2,2],[0,0]], mode='SYMMETRIC')
        feat = self.conv2d_1(feat)
        print_shape(feat)
        feat = self.batchnorm_1(feat, self.istrain)
        if blk_num > 0:
            upfeat = self.conv2d_transpose(feat)
            print_shape(upfeat)
            upfeat = self.batchnorm_2(upfeat, self.istrain)
        else:
            upfeat = feat
        return feat, upfeat

class SimpleNet(tf.keras.Model):
    # slim = tf.contrib.slim
    def __init__(self,out_chans,istrain):
        super(SimpleNet, self).__init__()
        nblocks = 2
        
        self.out_chans = out_chans
        self.nblocks  =  nblocks
        self.SimpleNetBlocks = []
        self.istrain = istrain
        self.conv2d = tf.keras.layers.Conv2D( out_chans ,kernel_size=5, activation=None)
        for blk_num in range(self.nblocks):
            self.SimpleNetBlocks.append(SimpleNetBlock(out_chans,blk_num, self.istrain))

    def call(self,input):
        print("rgb")
        print_shape(input)
        B, H, W, C = input.shape.as_list()
        normalizer_fn = None
        weights_initializer = tf.compat.v1.initializers.truncated_normal(stddev=1e-3)

        upfeats = list()
        feat = input
        # tf.compat.v1.summary.histogram(feat.name, feat)
        for blk_num in range(self.nblocks):
            feat, upfeat = self.SimpleNetBlocks[blk_num](feat, blk_num)
            upfeats.append(upfeat)
        upfeat = tf.concat(upfeats, axis = 3)
        # st()
        upfeat = tf.pad(tensor=upfeat, paddings=[[0,0],[2,2],[2,2],[0,0]], mode='SYMMETRIC')
        emb = self.conv2d(upfeat)
        # emb = slim.conv2d(upfeat, out_chans, kernel_size=1, activation_fn=None,
        #                   normalizer_fn=None, scope='conv_final')
        print_shape(emb)
        print("rgb_trans")
        return emb


class embnet2(tf.keras.Model):
    def __init__(self,istrain):
        super(embnet2, self).__init__()
        self.simpleNet = SimpleNet(const.emb_dim,istrain=istrain)
        self.beta = tf.Variable(1.2, dtype=tf.float32, name='margin_beta')

    def batch_norm(x, istrain):
        # return tf.identity(x)
        # decay of 0.99 can take ~1k steps to learn (according to my plots)
        return self.batchnorm(x, decay=0.9, 
                                            is_training=istrain,
                                            # updates_collections=None,
                                            center=True,
                                            scale=True,
                                            reuse=False)
    def get_distance(self,x):
        n = x.shape.as_list()[0]
        square = tf.reduce_sum(input_tensor=x**2, axis=1, keepdims=True)
        dis_square = square + tf.transpose(a=square) - 2.0 * tf.matmul(x, tf.transpose(a=x)) + EPS 
        # st()
        return tf.sqrt(dis_square + tf.eye(n))

    def reduce_emb(self,emb, inbound=None, together=False):
        ## emb -- [S,H/2,W/2,C], inbound -- [S,H/2,W/2,1]
        ## Reduce number of chans to 3 with PCA. For vis.
        S,H,W,C = emb.shape.as_list()
        keep = 3
        if together:
            # emb = tf.py_function(self.pca_embed_together, [emb,keep], tf.float32)
            emb = tf.convert_to_tensor(self.pca_embed_together(emb,keep))

        else:
            emb = tf.py_function(self.pca_embed, [emb,keep], tf.float32)
        emb.set_shape([S,H,W,keep])
        emb = normalize(emb) - 0.5
        if inbound is not None:
            emb_inbound = emb*inbound
        else:
            emb_inbound = None
        return emb, emb_inbound

    def pca_embed_together(self,emb, keep):
        ## emb -- [S,H/2,W/2,C]
        ## keep is the number of principal components to keep
        ## Helper function for reduce_emb.
        S, H, W, K = np.shape(emb)
        if np.isnan(emb).any():
            out_img = np.zeros([S,H,W,keep], dtype=emb.dtype)
        pixelskd = np.reshape(emb, (S*H*W, K))
        P = PCA(keep)
        P.fit(pixelskd)
        pixels3d = P.transform(pixelskd)
        out_img = np.reshape(pixels3d, [S,H,W,keep]).astype(np.float32)
        if np.isnan(out_img).any():
            out_img = np.zeros([S,H,W,keep], dtype=np.float32)
        return out_img
    def distance_sampling(self,x, cutoff, nonzero_loss_cutoff, n_split):
        n, d = x.shape.as_list()
        split = n/n_split
        # st()
        distance = tf.maximum(self.get_distance(x), cutoff)
        log_weights = ((2.0 - float(d)) * tf.math.log(distance)
                      - (float(d-3)/2) * tf.math.log(1.0 - 0.25*(distance**2)))
        # st()
        weights = tf.exp(log_weights - tf.reduce_max(input_tensor=log_weights))

        mask = np.ones(weights.shape)
        for i in range(0, n):
            for idx_split in range(n_split):
                #mask[i,i] = 0
                # st()
                mask[i,int((i+split*idx_split)%n)] = 0
        # st()
        mask = tf.constant(mask, tf.float32)
        weights = weights * mask * tf.cast((distance < nonzero_loss_cutoff), tf.float32)
        weights = weights / tf.reduce_sum(input_tensor=weights, axis=1, keepdims=True)
        #a_indices = tf.random.uniform([n, 1], maxval=n, dtype=tf.int32)
        a_indices = tf.random.shuffle(tf.range(start=0, limit=n, delta=1, dtype=tf.int32))
        a_indices = tf.reshape(a_indices, [n, 1])
        #positive samples: interval equals to split
        # st()
        split_indices =int(split)*tf.random.uniform([n,1], minval=1, maxval=n_split, dtype=tf.int32)
        p_indices = tf.floormod((a_indices + split_indices), tf.constant(n, dtype=tf.int32))
        weights_sampled = tf.gather_nd(weights, a_indices)
        n_indices = tf.random.categorical(tf.math.log(weights_sampled), 1)
        n_indices = tf.reshape(n_indices, [n, 1])
        #print(a_indices.shape.as_list(), p_indices.shape.as_list(), n_indices.shape.as_list())
        return a_indices, p_indices, n_indices #shape: [n, 1]


    # def SimpleNetBlock(feat, blk_num, out_chans, istrain):
    #     from tensorflow.contrib.slim import conv2d, conv2d_transpose

    #     with tf.compat.v1.variable_scope('Block%d' % blk_num):
    #         feat = tf.pad(tensor=feat, paddings=[[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
    #         feat = conv2d(feat, out_chans*(2**blk_num), stride=2, scope='conv')
    #         print_shape(feat)
    #         feat = batch_norm(feat, istrain)
            
    #         feat = tf.pad(tensor=feat, paddings=[[0,0],[2,2],[2,2],[0,0]], mode='SYMMETRIC')
    #         feat = conv2d(feat, out_chans*(2**blk_num), rate=2, scope='dilconv')
    #         print_shape(feat)
    #         feat = batch_norm(feat, istrain)
    #         if blk_num > 0:
    #             upfeat = conv2d_transpose(feat, out_chans, kernel_size=[4,4], stride=2,
    #                                       padding='SAME', scope='deconv')
    #             print_shape(upfeat)
    #             upfeat = batch_norm(upfeat, istrain)
    #         else:
    #             upfeat = feat
    #         return feat, upfeat
    def margin_loss(self,emb, n_sampling, n_split):
        alpha = 0.2
        cutoff = 0.5
        nonzero_loss_cutoff = 1.4
        a_indices, p_indices, n_indices = self.distance_sampling(emb, cutoff, nonzero_loss_cutoff, n_split)
        emb_a = tf.gather_nd(emb, a_indices)
        emb_p = tf.gather_nd(emb, p_indices)
        emb_n = tf.gather_nd(emb, n_indices)
        d_ap = tf.sqrt(tf.reduce_sum(input_tensor=(emb_p - emb_a)**2, axis=1) + 1e-8)
        d_an = tf.sqrt(tf.reduce_sum(input_tensor=(emb_n - emb_a)**2, axis=1) + 1e-8)

        loss_p = tf.maximum(d_ap - self.beta + alpha, 0.0)
        loss_n = tf.maximum(self.beta - d_an + alpha, 0.0)

        pair_num = tf.reduce_sum(input_tensor=tf.cast(loss_p > 0.0, tf.float32)+tf.cast(loss_n > 0.0, tf.float32))
        loss = tf.reduce_sum(input_tensor=loss_p + loss_n)/pair_num

        return loss
    def emb_vis(self,rgb, emb, emb_pred, inbound):
        ## emb,emb_pred -- [S,H/2,W/2,C] where C is length of emb vector per pixel.
        ## rgb -- [S,H/2,W/2,3], inbound -- [S,H/2,W/2,1]
        S,H,W,C = emb.shape.as_list()
        embs = tf.concat([emb, emb_pred], axis=0)
        inbounds = tf.concat([inbound, inbound], axis=0)
        # emb, emb_inbound = reduce_emb(emb, inbound)
        # emb_pred, emb_pred_inbound = reduce_emb(emb_pred, inbound)
        
        embs, embs_inbound = self.reduce_emb(embs, inbounds, together=True)
        # emb_inbound, emb_pred_inbound = tf.split(embs_inbound, 2, axis=0)
        emb, emb_pred = tf.split(embs, 2, axis=0)
        rgb_emb_vis = tf.concat([rgb, emb, emb_pred], axis=2)
        # utils_improc.summ_rgb('rgb_emb_embpred', rgb_emb_vis)
        # return emb_inbound, emb_pred_inbound
        return emb, emb_pred


    # def EmbNet3D(emb_pred, emb, istrain):
    #     total_loss = 0.0

    #     with tf.variable_scope('emb3D'):
    #         print 'EmbNet3D...'

    #         B, H, W, D, C = emb_pred.shape.as_list()
    #         # assert(C==hyp.emb_dim)
            
    #         loss = margin_loss_3D(emb, emb_pred)
    #         emb_pca, emb_pred_pca = emb_vis(rgb, emb, emb_pred)
    #         total_loss = utils_misc.add_loss(total_loss, loss,
    #                                          hyp.emb_coeff, 'margin_3D')

    #         # smooth_loss = edge_aware_smooth_loss(emb, rgb)
    #         # smooth_loss += edge_aware_smooth_loss(emb_pred, rgb)
    #         # total_loss = utils_misc.add_loss(total_loss, smooth_loss,
    #         #                                 hyp.emb_smooth_coeff, 'smooth')

    #         # l1_loss = l1_on_axis(emb-emb_pred)
    #         # utils_improc.summ_oned('l1_loss', l1_loss)
    #         # # l1_loss = reduce_masked_mean(l1_loss, inbound)
    #         # total_loss = utils_misc.add_loss(total_loss, l1_loss,
    #         #                                  hyp.emb_l1_coeff, 'l1')

    #         # # emb = emb / l2_on_axis(emb, axis=3)
    #         # # emb_pred = emb_pred / l2_on_axis(emb_pred, axis=3)
    #         # return total_loss, emb, emb_pred, emb_pca, emb_pred_pca
    #         # # return total_loss
    #         return total_loss


    # def margin_loss_3D(emb0, emb1):
    #     # emb0 and emb1 are B x H x W x D x C
    #     B,H,W,D,C = emb0.shape.as_list()
    #     loss = 0.0
    #     emb0_all = []
    #     emb1_all = []
    #     for s in range(B):
    #         n_sampling = 960
    #         sample_indicies = tf.random.uniform([n_sampling, 1], maxval=H*W*D, dtype=tf.int32)
    #         emb0_s_ = tf.reshape(emb0[s], [H*W*D, C])
    #         emb1_s_ = tf.reshape(emb1[s], [H*W*D, C])
    #         emb0_s_ = tf.gather_nd(emb0_s_, sample_indicies)
    #         emb1_s_ = tf.gather_nd(emb1_s_, sample_indicies)
    #         # these are N x D
    #         emb0_all.append(emb0_s_)
    #         emb1_all.append(emb1_s_)
    #     emb0_all = tf.concat(emb0_all, axis=0)
    #     emb1_all = tf.concat(emb1_all, axis=0)
    #     emb_all = tf.concat([emb0_all, emb1_all], axis=0)
    #     n_split = 2
    #     loss = margin_loss(emb_all, n_sampling, n_split) / float(B)
    #     return loss
    def margin_loss_2D(self,emb, emb_pred):
        ## emb,emb_pred,emb_aug -- [S,H/2,W/2,C]
        ## Use lifted_struct_loss between emb,emb_pred,emb_aug treating
        ## every s in S as a separate loss.

        # losstype = hyp.emb_loss
        # assert losstype in {'lifted', 'npairs'}
        # losstype = 'lifted'
        B,H,W,C = emb.shape.as_list()
        losstype = 'margin'
        # S,H,W,C = emb.shape.as_list()
        loss = 0.0
        emb_all = []
        emb_pred_all = []
        for s in range(B):
            n_sampling = 960
            sample_indicies = tf.random.uniform([n_sampling, 1], maxval=H*W, dtype=tf.int32)
            emb_s_ = tf.reshape(emb[s], [H*W, C])
            emb_s_ = tf.gather_nd(emb_s_, sample_indicies)
            emb_pred_s_ = tf.reshape(emb_pred[s], [H*W, C])
            emb_pred_s_ = tf.gather_nd(emb_pred_s_, sample_indicies)
            emb_all.append(emb_s_)
            emb_pred_all.append(emb_pred_s_)

        emb_all = tf.concat(emb_all, axis=0)
        emb_pred_all = tf.concat(emb_pred_all, axis=0)
        emb_all = tf.concat([emb_all, emb_pred_all], axis=0)
        n_split = 2
        loss = self.margin_loss(emb_all, n_sampling, n_split) / float(B)
        return loss



    @tf.function
    def call(self,rgb, emb_pred):
        # rgb is [S,H,W,3]
        # inbound is [S,H,W,1]
        # emb_pred -- [S,H/2,W/2,C] where C is length of emb vector per pixel.

        ## Compute embs for `rgb` using EmbNet(SimpleNet) and
        ## compare/loss against `emb_pred`. Use loss only within
        ## the mask `inbound`.

        total_loss = 0.0
        # st()
        with tf.compat.v1.name_scope('emb'):
            # print 'EmbNet...'

            B, H, W, C = emb_pred.shape.as_list()
            assert(C==const.emb_dim)
            
            # inbound = tf.image.resize_nearest_neighbor(inbound, [H, W])
            inbound = tf.ones([B,H,W,1])

            # if hyp.emb_use_aug:
            #     # ignore/replace emb_pred
            #     rgb_aug = random_color_augs(rgb)
            #     rgb_all = tf.concat([rgb, rgb_aug], axis=0)
            #     emb_all = SimpleNet(rgb_all, istrain, C)
            #     emb, emb_pred = tf.split(emb_all, 2, axis=0)
            #     inbound = tf.ones_like(inbound)
            #     emb_aug = None # support old code that used BOTH aug and pred
            # else:
            emb = self.simpleNet(rgb)
            
            emb = emb / (EPS + l2_on_axis(emb, axis=3))
            emb_pred = emb_pred / (EPS + l2_on_axis(emb_pred, axis=3))
            # st()
            emb_aug = None # support old code that used BOTH aug and pred
            
            rgb = tf.image.resize(rgb, [H, W], method=tf.image.ResizeMethod.BILINEAR)

            loss = self.margin_loss_2D(emb, emb_pred)
            # emb_pca, emb_pred_pca = self.emb_vis(rgb, emb, emb_pred, inbound)

            total_loss = add_loss(total_loss, loss,
                                             const.emb_coeff, 'metric')

            # loss = metric_loss(rgb, emb, emb_pred, emb_aug, inbound)
            # emb_pca, emb_pred_pca = emb_vis(rgb, emb, emb_pred, inbound)
            # total_loss = utils_misc.add_loss(total_loss, loss,
            #                                  hyp.emb_coeff, 'metric')

            # smooth_loss = edge_aware_smooth_loss(emb, rgb)
            # smooth_loss += edge_aware_smooth_loss(emb_pred, rgb)
            # total_loss = utils_misc.add_loss(total_loss, smooth_loss,
            #                                  hyp.emb_smooth_coeff, 'smooth')

            l1_loss_im = l1_on_chans(emb-emb_pred)
            # utils_improc.summ_oned('l1_loss', l1_loss_im*inbound)
            l1_loss = reduce_masked_mean(l1_loss_im, inbound)
            total_loss = add_loss(total_loss, l1_loss,
                                             const.emb_l1_coeff, 'l1')

            # loss_3D = margin_loss_3D(emb3D_g, emb3D_e)
            # total_loss = utils_misc.add_loss(total_loss, loss_3D,
            #                                  hyp.emb_3D_coeff, '3D')

            # dx, dy, dz = gradient3D(emb3D_e, absolute=True)
            # smooth_vox = tf.reduce_mean(dx+dy+dx, axis=4, keepdims=True)
            # smooth_loss = tf.reduce_mean(smooth_vox)
            # total_loss = utils_misc.add_loss(total_loss, smooth_loss, hyp.emb_smooth3D_coeff, 'smooth3D')
            # total_loss, emb, emb_pred, inbound, emb_pca, emb_pred_pca
            # emb = emb / l2_on_axis(emb, axis=3)
            # emb_pred = emb_pred / l2_on_axis(emb_pred, axis=3)
            return total_loss,rgb,emb,emb_pred
        # return total_loss