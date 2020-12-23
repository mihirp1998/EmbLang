import tensorflow as tf

def sum():
	return tf.ones([2,2,2])
def resize_by_axis(image, dim_1, dim_2, ax):
	resized_list = []
	unstack_img_depth_list = tf.unstack(image, axis = ax)
	for i in unstack_img_depth_list:
		resized_list.append(tf.image.resize(i, [dim_1, dim_2]))
	stack_img = tf.stack(resized_list, axis=ax)  
	return stack_img

def resize_voxel(vox,dims):
	dim_1,dim_2,dim_3 = dims
	resized_along_depth = resize_by_axis(vox,dim_1,dim_2,3) 
	resized_along_width = resize_by_axis(resized_along_depth,dim_1,dim_3,2)
	return resized_along_width
# resized_along_depth = resize_by_axis(x,50,60,2, True) 
# resized_along_width = resize_by_axis(resized_along_depth,50,70,1,True)