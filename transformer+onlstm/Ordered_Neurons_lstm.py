import tensorflow as tf
from tensorflow.nn.rnn_cell import RNNCell,LSTMStateTuple
from tensorflow.python.ops import array_ops,nn_ops,init_ops,math_ops

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term

class ON_LSTM(RNNCell):
    def __init__(self, num_units,chunk_size, forget_bias=1.0,
                   state_is_tuple=True, reuse=None):

        super(ON_LSTM, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                       "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self.chunk_size = chunk_size
        self.n_chunk = num_units // chunk_size

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def cumsum(self,x,direction='right'):

        if direction == 'right':
            output = tf.cumsum(tf.nn.softmax(x,-1),-1)
            return output
        elif direction == 'left':
            output = 1-tf.cumsum(tf.nn.softmax(x,-1),-1)
            return output
        # else :
        #     return output

    def call(self, inputs, state):#state
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        concat = _linear([inputs, h], 4 * self._num_units + 2 * self.n_chunk, True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        f_master_t = concat[:,:self.n_chunk]
        f_master_t = self.cumsum(tf.nn.softmax(f_master_t,axis=-1))
        f_master_t = tf.expand_dims(f_master_t,2)

        i_master_t = concat[:,self.n_chunk:2*self.n_chunk]
        i_master_t = self.cumsum(tf.nn.softmax(i_master_t,axis=-1),'left')
        i_master_t = tf.expand_dims(i_master_t,2)
        concat = concat[:, 2*self.n_chunk:]
        #reshape
        concat = tf.reshape(concat,[-1,self.n_chunk*4,self.chunk_size])

        f_t = tf.nn.sigmoid(concat[:, :self.n_chunk])
        i_t = tf.nn.sigmoid(concat[:, self.n_chunk : 2*self.n_chunk])
        o_t = tf.nn.sigmoid(concat[:, 2*self.n_chunk : 3*self.n_chunk])
        c_t_hat = tf.tanh(concat[:, 3*self.n_chunk:])

        w_t = f_master_t * i_master_t

        new_c = w_t*(f_t*tf.reshape(c,[-1,self.n_chunk,self.chunk_size]) + i_t*c_t_hat) + \
         (i_master_t-w_t)*c_t_hat + \
         (f_master_t-w_t)*tf.reshape(c,[-1,self.n_chunk,self.chunk_size])
        new_h = tf.tanh(new_c)*o_t
        new_c = tf.reshape(new_c,[-1,self._num_units])
        new_h = tf.reshape(new_h,[-1,self._num_units])

#         i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

#         new_c = (
#             c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
#         new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state


if __name__=='__main__':
    inputs=tf.ones([128,28,28])
    batch_size = tf.shape(inputs)[0]
    cell_fw=ON_LSTM(60,3)
    # cell_fw = tf.nn.rnn_cell.BasicLSTMCell(50)
    init_state = cell_fw.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell_fw,inputs,initial_state=init_state, time_major=False,dtype=tf.float32)
    print(outputs)