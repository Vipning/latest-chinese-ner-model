import math
import tensorflow as tf
from nn_utils import linear


def _split_heads(x, num_heads):
    n = num_heads
    old_shape = x.get_shape().dims

    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])


def _combine_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    x.set_shape(new_shape)

    return x


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a
    different frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can
    be experessed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.
    """
    with tf.name_scope("add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) *
                       tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def attention_bias(inputs,inf=-1e9, name="attention_bias"):
    with tf.name_scope(name, values=[inputs]):
        mask = inputs
        ret = (1.0 - mask) * inf
    return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def dot_product_attention(query, key, value, bias,keep_prob,name=None):
    """ dot-product attention.

    Args:
        query: a Tensor with shape [batch, heads, length_q, depth_k]
        key: a Tensor with shape [batch, heads, length_kv, depth_k]
        value: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        keep_prob: a floating point number
        summaries: a boolean
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
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


def multihead_attention(query,memory, bias, key_size, value_size, output_size,
                        num_heads, keep_prob=None, data_format="NHWC",
                        attention_function="dot_product",dtype=None, scope=None):
    """ Multihead scaled-dot-product attention with input/output
        transformations.

    Args:
        query: a Tensor with shape [batch, length_q, channels] if
            data_format is `NHWC`, [batch, channels, length_q] if
            data_format is `NCHW`
        memory: a Tensor with shape [batch, length_m, channels] if
            data_format is `NHWC`, [batch, channels, length_q] if
            data_format is `NCHW`
        bias: bias Tensor (see attention_bias())
        key_size: an integer
        value_size: an integer
        output_size: an integer
        num_heads: an integer dividing total_key_depth and total_value_depth
        keep_prob: a floating point number
        summaries: a boolean
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        data_format: "NHWC" or "NCHW"
        attention_function: "dot_product" or "additive"
        dtype: an optional instance of tf.DType
        scope: an optional string

    Returns:
        A Tensor.
    """
    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_attention",
                           values=[query, memory], dtype=dtype):

        axis = 2

        if memory is None:
            # self attention
            size = key_size * 2 + value_size
            combined = linear(query, size, True, True, data_format=data_format,
                              scope="qkv_transform")
            q, k, v = tf.split(combined, [key_size, key_size, value_size],
                               axis=axis)
        else:
            q = linear(query, key_size, True, data_format=data_format,
                       scope="q_transform")
            combined = linear(memory, key_size + value_size, True,
                              data_format=data_format, scope="kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=axis)

        # split heads
        q = _split_heads(q, num_heads)
        k = _split_heads(k, num_heads)
        v = _split_heads(v, num_heads)

        # scale query
        if attention_function == "dot_product":
            key_depth_per_head = key_size // num_heads
            q *= key_depth_per_head ** -0.5

            # attention
            x = dot_product_attention(q, k, v, bias, keep_prob)

        # combine heads
        x = _combine_heads(x)

        x = linear(x, output_size, True, data_format=data_format,
                   scope="output_transform")

        return x
