import _init_paths
import os
import sys
import argparse
import os.path as osp
import random
import pickle
import glob
import numpy as np
import ipdb 
st = ipdb.set_trace
# from lib.tree import Tree
# from modules import Layout, Combine, Describe

######### hyperparameters ##########



def refine_tree_info(tree):
    tree = _set_bbox(tree)
    # tree = _set_layout_bbox(tree)
    return tree


def string(tree):
    # function_obj = tree.function_obj
    # set the bbox for the tree node
    wordVal = ""
    # st()
    if len(tree.children) >0:
        for child in tree.children:
            wordVal = string(child)
            if hasattr(tree, 'wordVal'):
                tree.wordVal =  wordVal + " " +tree.wordVal 
            else:
                tree.wordVal = tree.word + " " + wordVal 
        return tree.wordVal
        

    else:
        tree.wordVal = tree.word
        return  tree.wordVal

if __name__ == '__main__':
    files = glob.glob("CLEVR_64_36_AFTER_CORRECTION_NO_DEPTH/trees/train/*.tree")
    trees = [pickle.load(open(i,"rb")) for i in files]
    [string(i) for i in trees] 
    [pickle.dump(j,open(files[i],"wb")) for i,j in enumerate(trees)]


    # random.seed(12113)
    #
    # # tree = Tree()
    # # tree = expand_tree(tree, 0, None, [], 0)
    # # allign_tree(tree)
    #
    # num_sample = 1
    # trees = []
    # for i in range(num_sample):
    #     treei = Tree()
    #     treei = expand_tree(treei, 0, None, [], 0, max_level=2)
    #     allign_tree(treei, 0)
    #     objects = extract_objects(treei)
    #     trees += [treei]
    #     print(objects)
    #
    # visualize_tree(trees)

    # for i in range(1):
    #     print('normal sample tree')
    #     tree = sample_tree(max_layout_level=2, add_layout_prob=0.6, zero_shot=True, train=True)
    #     visualize_trees([tree])
    #     print('max sample tree')
    #     tree = sample_tree_flexible(max_layout_level=3, add_layout_prob=0.6, zero_shot=False, train=True,
    #                                 arguments={'max_num_objs': 3})
    #     visualize_trees([tree])
    #     print('fix sample tree')
    #     tree = sample_tree_flexible(max_layout_level=3, add_layout_prob=0.6, zero_shot=False, train=True,
    #                                 arguments={'fix_num_objs': 8})
    #     visualize_trees([tree])