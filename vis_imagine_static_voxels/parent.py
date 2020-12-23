
import _init_paths
import pickle
from lib.tree import Tree
import os.path as osp
import os

def add_parent(tree):
  tree = _add_parent(tree, None)

  return tree

def _add_parent(tree, parent):
  tree.parent = parent
  for i in range(0, tree.num_children):
    tree.children[i] = _add_parent(tree.children[i], tree) 

  return tree

res = 64

path = 'CLEVR_64_36_AFTER_CORRECTION_NO_DEPTH_TEST/trees_noparent'
outpath = 'CLEVR_64_36_AFTER_CORRECTION_NO_DEPTH_TEST/trees'

split = ['train']
# split = ['train']

os.rename(outpath, path)
for s in split:
  treepath = osp.join(path, s)
  files = os.listdir(treepath)
  try:
    os.makedirs(osp.join(outpath, s))
  except:
    pass

  for fi in files:
    if fi.endswith('tree'):
      with open(osp.join(path, s, fi), 'rb') as f:
        treei = pickle.load(f)
      treei = add_parent(treei)
      pickle.dump(treei, open(osp.join(outpath, s, fi), 'wb'))
