import tensorflow as tf
import numpy as np



def _linear_2d(inputs, output_size, bias, concat=True):
  input_size = [item.get_shape()[1].value for item in inputs]
  outputs = []

  if concat:
    total_input_size = 0
    # input_size = sum(input_size)
    for size in input_size:
      total_input_size += size

    inputs = tf.concat(inputs, 1)

    shape = [total_input_size, output_size]
    matrix = tf.get_variable("matrix", shape)
    outputs.append(tf.matmul(inputs, matrix))
  else:
    for i in range(len(input_size)):
      shape = [input_size[i], output_size]
      name = "matrix_%d" % i
      matrix = tf.get_variable(name, shape)
      outputs.append(tf.matmul(inputs[i], matrix))

  output = tf.add_n(outputs)

  if bias is not None:
    shape = [output_size]
    bias = tf.get_variable("bias", shape)
    output = tf.nn.bias_add(output, bias)

  return output


def _linear_3d(inputs, output_size, bias, concat=True,data_format="NHWC"):#NHWC
  channel_axis = -1
  space_axis = 1

  input_size = [item.get_shape()[channel_axis].value for item in inputs]

  outputs = []

  if concat:
    input_size = sum(input_size)
    inputs = tf.concat(inputs, channel_axis)
    inputs = tf.expand_dims(inputs, space_axis)

    shape = [input_size, output_size]
    matrix = tf.get_variable("matrix", shape)
    matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
    output = tf.nn.convolution(inputs, matrix, "VALID",
                               data_format=data_format)
    outputs.append(output)
  else:
    for i in range(len(input_size)):
      inputs = tf.expand_dims(inputs, space_axis)

      shape = [input_size[i], output_size]
      name = "matrix_%d" % i
      matrix = tf.get_variable(name, shape)
      matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
      output = tf.nn.convolution(inputs, matrix, "VALID",
                                 data_format=data_format)
      outputs.append(output)

  output = tf.add_n(outputs)

  if bias is not None:
    bias = tf.get_variable("bias", [output_size])
    output = tf.nn.bias_add(output, bias, data_format=data_format)

  output = tf.squeeze(output, space_axis)

  return output


def linear(inputs, output_size, bias, concat=True, data_format="NHWC",
           dtype=None, scope=None):
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  ndims = [ip.get_shape().ndims for ip in inputs]

  if any([dim - ndims[0] for dim in ndims]):
    raise ValueError("inputs do not agree on dimensions: %s" % ndims)

  rank = ndims[0]
  with tf.variable_scope(scope, default_name="linear", values=[inputs],
                         dtype=dtype):
    if rank == 2:
      output = _linear_2d(inputs, output_size, bias, concat)
    elif rank == 3:
      output = _linear_3d(inputs, output_size, bias, concat, data_format)
    else:
      raise ValueError("Input rank must be 2, 3 or 4, found %d" % rank)

    return output
    
def shape(tensor,dim):
    return tensor.get_shape().as_list()[dim] or tf.shape(tensor)[dim]

def projection(inputs, output_size, initializer=None):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
  if len(inputs.get_shape()) > 3:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

  if len(inputs.get_shape()) == 3:
    batch_size = shape(inputs, 0)
    seqlen = shape(inputs, 1)
    emb_size = shape(inputs, 2)
    current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
  else:
    current_inputs = inputs

  for i in range(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
    hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
    current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
  output_bias = tf.get_variable("output_bias", [output_size])
  outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
  return outputs

class CustomLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, batch_size, initializer,dropout=1.):
    self._num_units = num_units
    self._dropout = dropout
    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
    initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
    initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
      c, h = state
      h *= self._dropout_mask
      concat = projection(tf.concat([inputs, h], 1), 3 * self.output_size, initializer=self._initializer)
      i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
      i = tf.sigmoid(i)
      new_c = (1 - i) * c  + i * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer