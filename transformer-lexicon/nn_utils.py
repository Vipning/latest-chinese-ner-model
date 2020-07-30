import tensorflow as tf

def shape(tensor,dim):
    return tensor.get_shape().as_list()[dim] or tf.shape(tensor)[dim]