import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from nbtschematic import SchematicFile
from mpl_toolkits.mplot3d import Axes3D

res = 64

# vpath = 'output/blendfiles/train/CLEVR_new_000001.schematic'
vpath = 'CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_00000%d.schematic'

for img in range(10):
	sf = SchematicFile.load(vpath%img)
	blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
	data = np.frombuffer(sf.data, dtype=sf.data.dtype)
	blocks = blocks.reshape((res,res,res))
	# np.save('voxel.npy',blocks)
	vals = np.unique(blocks)
	print(vals)
	colors = np.empty(blocks.shape, dtype=object)
	colorname = ['red','blue','green','black','yellow','cyan','magenta']
	for i,c in zip(vals, colorname):
		colors[blocks == i] = c

	# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
	# box = [108,  93,   0,  19,  20,  19]
	# Z = np.array([[108,93,0],[108,93,19],[108,113,0],[127,93,0],[108,113,19],[127,113,0],[127,93,19],[127,113,19]])
	# verts = [[Z[0],Z[1],Z[2],Z[3]],
	#  [Z[4],Z[5],Z[6],Z[7]], 
	#  [Z[0],Z[1],Z[5],Z[4]], 
	#  [Z[2],Z[3],Z[7],Z[6]], 
	#  [Z[1],Z[2],Z[6],Z[5]],
	#  [Z[4],Z[7],Z[3],Z[0]]]


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.voxels(blocks, facecolors=colors)
	# ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
	# ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
	plt.show()
	plt.close()