import tensorflow as tf
import math
def shape(tensor,dim):
    return tensor.get_shape().as_list()[dim] or tf.shape(tensor)[dim]

def get_local_bias(inputs, wins, inf=-1e9, name=None):
    # def smaller():
    #     return wins
    # def larger():
    #     return 4

    with tf.name_scope(name, default_name="attention_bias", values=[inputs]):
        length = inputs
        # wins=tf.cond(tf.greater(wins,length),larger,smaller,strict=True)
        wins = tf.to_int64(wins)
        mask=tf.ones([length, length])#length*length
        
        mask=tf.matrix_band_part(mask, wins, wins)

        ret = inf * (1.0 - mask)
    return tf.reshape(ret, [1, 1, length, length])

def gate_sum(x, y, q,num_heads,scope='gate_sum'):
    with tf.variable_scope(scope, default_name="gate_sum", values=[x, y]):
        shape = x.get_shape().as_list()[-1]
        #heads = x.get_shape().as_list()[2]
        gate_w = tf.get_variable('gate_weight',shape=[num_heads*shape,num_heads],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        # gate_U = tf.get_variable('gate_u', shape=[shape, heads])
        # gate_b = tf.get_variable('gate_bias', shape=[shape],
        #                          initializer=tf.initializers.zeros)
        gate = tf.sigmoid(tf.transpose(tf.reshape(tf.matmul(tf.reshape(tf.transpose(q,[0,2,1,3]), [-1, num_heads*shape]), gate_w),
                                     [tf.shape(q)[0], tf.shape(q)[2], tf.shape(q)[1], 1]),[0,2,1,3]))

        out = x * gate + y * (1 - gate)

        return tf.reshape(out, tf.shape(x))
