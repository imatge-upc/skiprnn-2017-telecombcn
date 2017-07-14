"""
Extended version of the BasicLSTMCell and BasicGRUCell in TensorFlow that allows to easily add custom inits,
normalization, etc.
"""

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

from rnn_cells import rnn_ops


class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    """
    Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, layer_norm=False):
        """
        Initialize the basic LSTM cell
        :param num_units: int, the number of units in the LSTM cell
        :param forget_bias: float, the bias added to forget gates
        :param activation: activation function of the inner states
        :param layer_norm: bool, whether to use layer normalization
        """
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._layer_norm = layer_norm

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # Parameters of gates are concatenated into one multiply for efficiency.
            concat = rnn_ops.linear([inputs, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

            if self._layer_norm:
                i = rnn_ops.layer_norm(i, name="i")
                j = rnn_ops.layer_norm(j, name="j")
                f = rnn_ops.layer_norm(f, name="f")
                o = rnn_ops.layer_norm(o, name="o")

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

    def trainable_initial_state(self, batch_size):
        """
        Create a trainable initial state for the BasicLSTMCell
        :param batch_size: number of samples per batch
        :return: LSTMStateTuple
        """
        def _create_initial_state(batch_size, state_size, trainable=True, initializer=tf.random_normal_initializer()):
            with tf.device('/cpu:0'):
                s = tf.get_variable('initial_state', shape=[1, state_size], dtype=tf.float32, trainable=trainable,
                                    initializer=initializer)
                state = tf.tile(s, tf.stack([batch_size] + [1]))
            return state

        with tf.variable_scope('initial_c'):
            initial_c = _create_initial_state(batch_size, self._num_units)
        with tf.variable_scope('initial_h'):
            initial_h = _create_initial_state(batch_size, self._num_units)
        return tf.contrib.rnn.LSTMStateTuple(initial_c, initial_h)


class BasicGRUCell(tf.contrib.rnn.RNNCell):
    """
    Gated Recurrent Unit cell.
    The implementation is based on http://arxiv.org/abs/1406.1078.
    """

    def __init__(self, num_units, activation=tf.tanh, layer_norm=False):
        """
        Initialize the basic GRU cell
        :param num_units: int, the number of units in the LSTM cell
        :param activation: activation function of the inner states
        :param layer_norm: bool, whether to use layer normalization
        """
        self._num_units = num_units
        self._activation = activation
        self._layer_norm = layer_norm

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with num_units cells."""
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                concat = rnn_ops.linear([inputs, state], 2 * self._num_units, True, bias_start=1.0)
                r, u = tf.split(value=concat, num_or_size_splits=2, axis=1)

                if self._layer_norm:
                    r = rnn_ops.layer_norm(r, name="r")
                    u = rnn_ops.layer_norm(u, name="u")

                # Apply non-linearity after layer normalization
                r = tf.sigmoid(r)
                u = tf.sigmoid(u)

            with tf.variable_scope("candidate"):
                c = self._activation(rnn_ops.linear([inputs, r * state], self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    def trainable_initial_state(self, batch_size):
        """
        Create a trainable initial state for the BasicGRUCell
        :param batch_size: number of samples per batch
        :return: tensor with shape [batch_size, self.state_size]
        """
        with tf.variable_scope('initial_h'):
            initial_h = rnn_ops.create_initial_state(batch_size, self._num_units)
        return initial_h
