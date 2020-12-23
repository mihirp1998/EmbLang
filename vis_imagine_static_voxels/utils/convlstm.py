# import tensorflow as tf
# import tensorflow_addons as tfa

# class ConvLSTMCell(tf.compat.v1.nn.rnn_cell.RNNCell):
#     """A LSTM cell with convolutions instead of multiplications.

#     Reference:
#       Xingjian, S. H. I., et al. "Convolutional LSTM network: 
#       A machine learning approach for precipitation nowcasting.
#       " Advances in Neural Information Processing Systems. 2015.
#     """

#     def __init__(self,
#                  shape,
#                  filters,
#                  kernel,
#                  initializer=None,
#                  forget_bias=1.0,
#                  activation=tf.tanh,
#                  normalize=True):
#         self._kernel = kernel
#         self._filters = filters
#         self._initializer = initializer
#         self._forget_bias = forget_bias
#         self._activation = activation
#         self._size = tf.TensorShape(shape + [self._filters])
#         self._normalize = normalize
#         self._feature_axis = self._size.ndims

#     @property
#     def state_size(self):
#         return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

#     @property
#     def output_size(self):
#         return self._size

#     def __call__(self, x, h, scope=None):
#         with tf.compat.v1.variable_scope(scope or self.__class__.__name__):
#             previous_memory, previous_output = h

#             channels = x.shape[-1].value
#             filters = self._filters
#             gates = 4 * filters if filters > 1 else 4
#             x = tf.concat([x, previous_output], axis=self._feature_axis)
#             n = channels + filters
#             m = gates
#             W = tf.compat.v1.get_variable(
#                 'kernel', self._kernel + [n, m], initializer=self._initializer, use_resource=False)
#             y = tf.nn.convolution(input=x, filters=W, padding='SAME')
#             if not self._normalize:
#                 y += tf.compat.v1.get_variable(
#                     'bias', [m], initializer=tf.compat.v1.initializers.constant(0.0), use_resource=False)
#             input_contribution, input_gate, forget_gate, output_gate = tf.split(
#                 y, 4, axis=self._feature_axis)

#             if self._normalize:
#                 input_contribution = tfa.layers.LayerNormalization(
#                     input_contribution)
#                 input_gate = tfa.layers.LayerNormalization(input_gate)
#                 forget_gate = tfa.layers.LayerNormalization(forget_gate)
#                 output_gate = tfa.layers.LayerNormalization(output_gate)

#             memory = (
#                 previous_memory * tf.sigmoid(forget_gate + self._forget_bias) +
#                 tf.sigmoid(input_gate) * self._activation(input_contribution))

#             if self._normalize:
#                 memory = tfa.layers.LayerNormalization(memory)

#             output = self._activation(memory) * tf.sigmoid(output_gate)

#             return output, tf.nn.rnn_cell.LSTMStateTuple(memory, output)


# class ConvGRUCell(tf.compat.v1.nn.rnn_cell.RNNCell):
#     """A GRU cell with convolutions instead of multiplications."""

#     def __init__(self,
#                  shape,
#                  filters,
#                  kernel,
#                  initializer=None,
#                  activation=tf.tanh,
#                  normalize=True):
#         self._filters = filters
#         self._kernel = kernel
#         self._initializer = initializer
#         self._activation = activation
#         self._size = tf.TensorShape(shape + [self._filters])
#         self._normalize = normalize
#         self._feature_axis = self._size.ndims

#     @property
#     def state_size(self):
#         return self._size

#     @property
#     def output_size(self):
#         return self._size

#     def __call__(self, x, h, scope=None):
#         with tf.compat.v1.variable_scope(scope or self.__class__.__name__):

#             with tf.compat.v1.variable_scope('Gates'):
#                 channels = x.shape[-1].value
#                 inputs = tf.concat([x, h], axis=self._feature_axis)
#                 n = channels + self._filters
#                 m = 2 * self._filters if self._filters > 1 else 2
#                 W = tf.compat.v1.get_variable(
#                     'kernel',
#                     self._kernel + [n, m],
#                     initializer=self._initializer, use_resource=False)
#                 y = tf.nn.convolution(input=inputs, filters=W, padding='SAME')
#                 if self._normalize:
#                     reset_gate, update_gate = tf.split(
#                         y, 2, axis=self._feature_axis)
#                     reset_gate = tfa.layers.LayerNormalization(reset_gate)
#                     update_gate = tfa.layers.LayerNormalization(update_gate)
#                 else:
#                     y += tf.compat.v1.get_variable(
#                         'bias', [m], initializer=tf.compat.v1.initializers.constant(1.0), use_resource=False)
#                     reset_gate, update_gate = tf.split(
#                         y, 2, axis=self._feature_axis)
#                 reset_gate, update_gate = tf.sigmoid(reset_gate), tf.sigmoid(
#                     update_gate)

#             with tf.compat.v1.variable_scope('Output'):
#                 inputs = tf.concat(
#                     [x, reset_gate * h], axis=self._feature_axis)
#                 n = channels + self._filters
#                 m = self._filters
#                 W = tf.compat.v1.get_variable(
#                     'kernel',
#                     self._kernel + [n, m],
#                     initializer=self._initializer, use_resource=False)
#                 y = tf.nn.convolution(input=inputs, filters=W, padding='SAME')
#                 if self._normalize:
#                     y = tfa.layers.LayerNormalization(y)
#                 else:
#                     y += tf.compat.v1.get_variable(
#                         'bias', [m], initializer=tf.compat.v1.initializers.constant(0.0), use_resource=False)
#                 y = self._activation(y)
#                 output = update_gate * h + (1 - update_gate) * y

#             return output, output
