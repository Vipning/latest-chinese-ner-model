import tensorflow as tf
import math
from nn_utils import shape


def attention_bias(inputs,inf=-1e9, name="attention_bias"):
    with tf.name_scope(name, values=[inputs]):
        mask = inputs
        ret = (1.0 - tf.cast(mask, tf.float32)) * inf
        return tf.expand_dims(tf.expand_dims(ret, 1), 1)

def _split_heads(x, num_heads):
    n = num_heads
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]#[batch_size,seq_length,num_heads,head_size]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])#[batch_size,num_heads,seq_length,head_size]


def _combine_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))#[batch_size,seq_length,num_heads*head_size]
    x.set_shape(new_shape)
    return x

def dot_product_attention(query, key, value, bias, keep_prob,name=None):
    """ dot-product attention.
    Args:
        query: a Tensor with shape [batch, heads, length_q, depth_k]
        key: a Tensor with shape [batch, heads, length_kv, depth_k]
        value: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        keep_prob: a floating point number
        name: an optional string

    Returns:
        A Tensor.
    """
    with tf.name_scope(name, default_name="dot_product_attention",
                       values=[query, key, value]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(query, key, transpose_b=True)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, keep_prob)

        return tf.matmul(weights, value)

def _shift(BD):#选元素
    """
    convert:
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
    to:
        0   1  2
        -1  0  1
        -2 -1  0
    """
    bsz=shape(BD,0)
    n_head=shape(BD,1)
    max_len=shape(BD,2)
    zero_pad = tf.zeros(shape=(bsz, n_head, max_len, 1))
    BD = tf.reshape(tf.concat([BD, zero_pad], axis=-1), shape=(bsz, n_head, -1, max_len))
    BD = tf.reshape(BD[:, :, :-1], shape=(bsz, n_head, max_len, -1))
    BD = BD[:, :, :, max_len:]
    return BD


def relative_multi_head_attention(x,mask,attention_size,num_heads,drop_keep_rate=1.0,reuse=None):
    # borrowed from: https://github.com/Kyubyong/transformer/blob/master/modules.py
    with tf.variable_scope("relative_multi_head_attention",reuse=reuse):
        # attention size must consistent with queries（keys）'s -1 dim
        batch_size = shape(x,0)
        # attention_size = x.get_shape().as_list()[-1]
        max_time = shape(x,1)

        pos_embed = relative_positional_encoding(max_time,attention_size//num_heads,True)

        # linear projections, shape=(batch_size, max_time, attention_size)
        query = tf.layers.dense(x, attention_size,use_bias=False,name="query_project",
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        # query = tf.layers.dense(x, attention_size, activation=tf.nn.relu, name="query_project",
        #                         kernel_initializer=tf.contrib.layers.xavier_initializer())
        # key do not dense in this model
        key = x
        # value = tf.layers.dense(x, attention_size, activation=tf.nn.relu, name="value_project",
        #                         kernel_initializer=tf.contrib.layers.xavier_initializer())
        value = tf.layers.dense(x, attention_size,use_bias=False,name="value_project",
                                kernel_initializer=tf.contrib.layers.xavier_initializer())

        # split and concatenation, shape=(batch_size, num_heads, max_time, attention_size / num_heads)
        query_ = tf.stack(tf.split(query, num_heads, axis=2), axis=1)
        key_ = tf.stack(tf.split(key, num_heads, axis=2), axis=1)
        # value_ = tf.concat(tf.split(value, num_heads, axis=2), axis=0)
        value_ = tf.stack(tf.split(value, num_heads, axis=2), axis=1)
        # shape =(num_heads, attention_size / num_heads)
        u = tf.get_variable('var_u', shape=[num_heads, attention_size // num_heads],
                            initializer=tf.glorot_normal_initializer())
        v = tf.get_variable('var_v', shape=[num_heads, attention_size // num_heads],
                            initializer=tf.glorot_normal_initializer())

        Qu = query_ + u[:, None]
        QKuK = tf.einsum('bnqd,bnkd->bnqk', Qu, key_)

        vR = tf.einsum('nd,ld->nl', v, pos_embed)[None, :, None]
        QR = tf.einsum('bnqd,ld->bnql',query_, pos_embed)
        QRvR = QR + vR
        QRvR = _shift(QRvR)#

        attn_outs = QKuK + QRvR#[batch_size,num_heads,max_time,max_time]
        # attn_outs = tf.reshape(attn_outs, shape=(batch_size*num_heads, max_time, max_time))
        # attn_outs = tf.concat(tf.unstack(attn_outs, axis=1), axis=0)
        # activation
        #apply talking heads before softmax
        # pre_softmax_weight=tf.get_variable('pre_softmax_weight',shape=[num_heads,num_heads],initializer=tf.glorot_normal_initializer())
        # attn_outs=tf.einsum("BNFT,NL->BLFT", attn_outs,pre_softmax_weight)
        #print(mask)
        ret=(1.0-tf.cast(mask,tf.float32))*-1e9
        bias=tf.expand_dims(tf.expand_dims(ret,1),1)
        attn_outs+=bias
        attn_outs = tf.nn.softmax(attn_outs)
        #apply talking heads after softmax
        # post_softmax_weight=tf.get_variable('post_softmax_weight',shape=[num_heads,num_heads],initializer=tf.glorot_normal_initializer())
        # attn_outs=tf.einsum("BNFT,NL->BLFT", attn_outs,post_softmax_weight)
        # dropout
        attn_outs = tf.nn.dropout(attn_outs,drop_keep_rate)
        # attn_outs = tf.concat(tf.unstack(attn_outs, axis=1), axis=0)
        # weighted sum
        outputs = tf.matmul(attn_outs,value_)
        # restore shape
        # outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs=_combine_heads(outputs)
        # outputs = tf.layers.dense(outputs,attention_size,use_bias=False,name="output_project",
        #                         kernel_initializer=tf.contrib.layers.xavier_initializer())
        # residual connection
        outputs += x
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32")(outputs)

    return outputs


def relative_positional_encoding(max_seq_len,pos_dim,zero_pad=False,reuse=None):
    with tf.variable_scope("positional_encoding", reuse=reuse):
        position_ind = tf.range(0,max_seq_len * 2)
        position=tf.cast(tf.range(-max_seq_len,max_seq_len),tf.float32)
        half_dim=pos_dim//2
        log_timescale_increment=math.log(10000) /(tf.cast(half_dim,tf.float32) - 1)
        inv_timescales = tf.exp(tf.to_float(tf.range(half_dim)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.reshape(signal,[max_seq_len*2,-1])
        
        if zero_pad:
          tf.concat((tf.zeros(shape=[1,pos_dim]),signal[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(signal, position_ind)

        return outputs

def feedforward(inputs, num_units, drop_keep_rate=1.0, reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope("multi_head_attention", reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True,'kernel_initializer':tf.contrib.layers.xavier_initializer()}
        outputs = tf.layers.conv1d(**params)

        # dropout
        outputs = tf.nn.dropout(outputs, drop_keep_rate)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True,'kernel_initializer':tf.contrib.layers.xavier_initializer()}
        outputs = tf.layers.conv1d(**params)

        # dropout
        outputs = tf.nn.dropout(outputs, drop_keep_rate)

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = layer_normalize(outputs)
        outputs=tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32")(outputs)

    return outputs
