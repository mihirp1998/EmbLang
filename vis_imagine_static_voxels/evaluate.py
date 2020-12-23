import numpy as np
import random
import ipdb 
st = ipdb.set_trace
import numpy as np
import utils
import tensorflow as tf
def subsample_embs_2D(emb_e, emb_g, rgb, samps=10):
    # emb_e and _g are images
    B, H, W, C = emb_e.shape
    
    F = 10 # yields a 21x21 patch
    patches_e, patches_g, patches_r = [], [], []
    for e, g, r in zip(emb_e, emb_g, rgb):
        for samp in range(samps):
            row = np.random.randint(F, H-F)
            col = np.random.randint(F, W-F)
            patch_e = e[row-F:row+F+1,col-F:col+F+1]
            patch_g = g[row-F:row+F+1,col-F:col+F+1]
            patch_r = r[row-F:row+F+1,col-F:col+F+1]
            # print patch_e.shape
            patches_e.append(patch_e)
            patches_g.append(patch_g)
            patches_r.append(patch_r)

    patches_e = np.stack(patches_e, axis=0)
    patches_g = np.stack(patches_g, axis=0)
    patches_r = np.stack(patches_r, axis=0)
    
    return patches_e, patches_g, patches_r
    # these are B*samp x F x F x C
def l2loss_numpy(x, y, mask=None, verify_shape = True, strict = False):
    if verify_shape:
        assert len(x.shape) == len(y.shape)
    if strict:
        xshp = x.shape
        yshp = y.shape
        assert (None not in xshp) and (None not in yshp) and xshp == yshp
    if mask is None:
        return np.mean(np.square(x - y))
    else:
        # where the mask is 0, the loss is 0
        return tf.reduce_mean(input_tensor=tf.square(mask * (x - y))) / (tf.reduce_mean(input_tensor=mask) + const.eps)
def make_border_black(vis):
    vis = np.copy(vis)
    vis[0,:,:] = 0
    vis[-1,:,:] = 0
    vis[:,0,:] = 0
    vis[:,-1,:] = 0
    return vis

def findLossMatrix(emb_e,emb_g):
    l2_loss_per_e= []
    for i in emb_e:
        l2_loss_per_g = []
        for j in emb_g:
            l2_loss_per_g.append(l2loss_numpy(i,j))
        l2_loss_per_e.append(l2_loss_per_g)
    return l2_loss_per_e
def make_border_green(vis):
    vis = np.copy(vis)
    vis[0,:,0] = 0
    vis[0,:,1] = 255
    vis[0,:,2] = 0
    
    vis[-1,:,0] = 0
    vis[-1,:,1] = 255
    vis[-1,:,2] = 0

    vis[:,0,0] = 0
    vis[:,0,1] = 255
    vis[:,0,2] = 0
    
    vis[:,-1,0] = 0
    vis[:,-1,1] = 255
    vis[:,-1,2] = 0
    return vis

def compute_precision(estim, ground, sentence, recalls=[1,3,5], pool_size=4):
    # inputs are lists
    # list elements are H x W x C
    emb_e, vis_e = estim
    emb_g, vis_g = ground
    # st()
    assert(len(emb_e)==len(emb_g))
    B = len(emb_e)
    precision = np.zeros(len(recalls), np.float32)
    # print 'precision B = %d' % B

    if len(vis_e[0].shape)==4:
        # H x W x D x C
        # squish the height dim, and look at the birdview
        vis_e = [np.mean(vis, axis=0) for vis in vis_e]
        vis_g = [np.mean(vis, axis=0) for vis in vis_g]
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    elif len(vis_e[0].shape)==3:
        # H x W x C
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    else:
        assert(False) # vis_e shape is weird

    perm = np.random.permutation(B)
    vis_inds = perm[:10]
    
    if B >= pool_size: # otherwise it's not going to be accurate
        # st()
        emb_e = np.stack(emb_e, axis=0)
        emb_g = np.stack(emb_g, axis=0)
        # emb_e = np.concatenate(emb_e, axis=0)
        # emb_g = np.concatenate(emb_g, axis=0)
        l2_loss_per_e= []
        for i in emb_e:
            l2_loss_per_g = []
            for j in emb_g:
                l2_loss_per_g.append(l2loss_numpy(i,j))
            l2_loss_per_e.append(l2_loss_per_g)
        
        # st()
        scores = np.array(l2_loss_per_e)
        # vect_e = np.reshape(emb_e, [B, -1])
        # vect_g = np.reshape(emb_g, [B, -1])
        # scores = np.dot(vect_e, np.transpose(vect_g))
        ranks = np.argsort(scores)
        sentences = []
        vis = []
        for i in vis_inds:
            minivis = []
            sentences.append(sentence[i])
            # first col: query
            # minivis.append(vis_e[i])
            minivis.append(make_border_black(vis_e[i]))
            
            # # second col: true answer
            # minivis.append(vis_g[i])
            
            # remaining cols: ranked answers
            for j in range(10):
                v = vis_g[ranks[i, j]]
                if ranks[i, j]==i:
                    minivis.append(make_border_green(v))
                else:
                    minivis.append(v)
            # concat retrievals along width
            minivis = np.concatenate(minivis, axis=1)
            # print 'got this minivis:', 
            # print minivis.shape
            
            vis.append(minivis)
        # concat examples along height
        vis = np.concatenate(vis, axis=0)
        # print 'got this vis:', 
        # print vis.shape
            
        for recall_id, recall in enumerate(recalls):
            for emb_id in range(B):
                if emb_id in ranks[emb_id, :recall]:
                    precision[recall_id] += 1
            # print("precision@", recall, float(precision[recall_id])/float(B))
        precision = precision/float(B)
    else:
        precision = np.nan*precision
        vis = np.zeros((H*10, W*11, 3), np.uint8)
    
    return precision, vis, sentence


if __name__ == "__main__":
    import numpy as np
    prec,vis = compute_precision(([np.random.randn(32,32,32,32)]*16,\
        [np.random.randn(64,64,3)]*16),\
    ([np.random.randn(32,32,32,32)]*16,[np.random.randn(64,64,3)]*16))
    print(prec,vis.shape) 
