"""
Skip RNN cells that decide which timesteps should be attended.
"""

from __future__ import absolute_import
from __future__ import print_function

import collections
import tensorflow as tf

from rnn_cells import rnn_ops

from tensorflow.python.framework import ops


SkipLSTMStateTuple = collections.namedtuple("SkipLSTMStateTuple", ("c", "h", "update_prob", "cum_update_prob"))
SkipLSTMOutputTuple = collections.namedtuple("SkipLSTMOutputTuple", ("h", "state_gate"))
LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple

SkipGRUStateTuple = collections.namedtuple("SkipGRUStateTuple", ("h", "update_prob", "cum_update_prob"))
SkipGRUOutputTuple = collections.namedtuple("SkipGRUOutputTuple", ("h", "state_gate"))


def _binary_round(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    
    Based on http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html

    :param x: input tensor
    :return: y=round(x) with gradients defined by the identity mapping (y=x)
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)


class SkipLSTMCell(tf.contrib.rnn.RNNCell):
    """
    Single Skip LSTM cell. Augments the basic LSTM cell with a binary output that decides whether to
    update or copy the cell state. The binary neuron is optimized using the Straight Through Estimator.
    """
    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, layer_norm=False, update_bias=1.0):
        """
        Initialize the Skip LSTM cell
        :param num_units: int, the number of units in the LSTM cell
        :param forget_bias: float, the bias added to forget gates
        :param activation: activation function of the inner states
        :param layer_norm: bool, whether to use layer normalization
        :param update_bias: float, initial value for the bias added to the update state gate
        """
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._layer_norm = layer_norm
        self._update_bias = update_bias

    @property
    def state_size(self):
        return SkipLSTMStateTuple(self._num_units, self._num_units, 1, 1)

    @property
    def output_size(self):
        return SkipLSTMOutputTuple(self._num_units, 1)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c_prev, h_prev, update_prob_prev, cum_update_prob_prev = state

            # Parameters of gates are concatenated into one multiply for efficiency.
            concat = rnn_ops.linear([inputs, h_prev], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

            if self._layer_norm:
                i = rnn_ops.layer_norm(i, name="i")
                j = rnn_ops.layer_norm(j, name="j")
                f = rnn_ops.layer_norm(f, name="f")
                o = rnn_ops.layer_norm(o, name="o")

            new_c_tilde = (c_prev * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
            new_h_tilde = self._activation(new_c_tilde) * tf.sigmoid(o)

            # Compute value for the update prob
            with tf.variable_scope('state_update_prob'):
                new_update_prob_tilde = rnn_ops.linear(new_c_tilde, 1, True, bias_start=self._update_bias)
                new_update_prob_tilde = tf.sigmoid(new_update_prob_tilde)

            # Compute value for the update gate
            cum_update_prob = cum_update_prob_prev + tf.minimum(update_prob_prev, 1. - cum_update_prob_prev)
            update_gate = _binary_round(cum_update_prob)

            # Apply update gate
            new_c = update_gate * new_c_tilde + (1. - update_gate) * c_prev
            new_h = update_gate * new_h_tilde + (1. - update_gate) * h_prev
            new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
            new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob

            new_state = SkipLSTMStateTuple(new_c, new_h, new_update_prob, new_cum_update_prob)
            new_output = SkipLSTMOutputTuple(new_h, update_gate)

            return new_output, new_state

    def trainable_initial_state(self, batch_size):
        """
        Create a trainable initial state for the SkipLSTMCell
        :param batch_size: number of samples per batch
        :return: SkipLSTMStateTuple
        """
        with tf.variable_scope('initial_c'):
            initial_c = rnn_ops.create_initial_state(batch_size, self._num_units)
        with tf.variable_scope('initial_h'):
            initial_h = rnn_ops.create_initial_state(batch_size, self._num_units)
        with tf.variable_scope('initial_update_prob'):
            initial_update_prob = rnn_ops.create_initial_state(batch_size, 1, trainable=False,
                                                               initializer=tf.ones_initializer())
        with tf.variable_scope('initial_cum_update_prob'):
            initial_cum_update_prob = rnn_ops.create_initial_state(batch_size, 1, trainable=False,
                                                                   initializer=tf.zeros_initializer())
        return SkipLSTMStateTuple(initial_c, initial_h, initial_update_prob, initial_cum_update_prob)


class MultiSkipLSTMCell(tf.contrib.rnn.RNNCell):
    """
    Stack of Skip LSTM cells. The selection binary output is computed from the state of the cell on top of
    the stack.
    """
    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, layer_norm=False, update_bias=1.0):
        """
        Initialize the stack of Skip LSTM cells
        :param num_units: list of int, the number of units in each LSTM cell
        :param forget_bias: float, the bias added to forget gates
        :param activation: activation function of the inner states
        :param layer_norm: bool, whether to use layer normalization
        :param update_bias: float, initial value for the bias added to the update state gate
        """
        if not isinstance(num_units, list):
            num_units = [num_units]
        self._num_units = num_units
        self._num_layers = len(self._num_units)
        self._forget_bias = forget_bias
        self._activation = activation
        self._layer_norm = layer_norm
        self._update_bias = update_bias

    @property
    def state_size(self):
        return [LSTMStateTuple(num_units, num_units) for num_units in self._num_units[:-1]] + \
               [SkipLSTMStateTuple(self._num_units[-1], self._num_units[:-1], 1, 1)]

    @property
    def output_size(self):
        return SkipLSTMOutputTuple(self._num_units[-1], 1)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            update_prob_prev, cum_update_prob_prev = state[-1].update_prob, state[-1].cum_update_prob
            cell_input = inputs
            state_candidates = []

            # Compute update candidates for all layers
            for idx in range(self._num_layers):
                with tf.variable_scope('layer_%d' % (idx + 1)):
                    c_prev, h_prev = state[idx].c, state[idx].h

                    # Parameters of gates are concatenated into one multiply for efficiency.
                    concat = rnn_ops.linear([cell_input, h_prev], 4 * self._num_units[idx], True)

                    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
                    i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

                    if self._layer_norm:
                        i = rnn_ops.layer_norm(i, name="i")
                        j = rnn_ops.layer_norm(j, name="j")
                        f = rnn_ops.layer_norm(f, name="f")
                        o = rnn_ops.layer_norm(o, name="o")

                    new_c_tilde = (c_prev * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
                    new_h_tilde = self._activation(new_c_tilde) * tf.sigmoid(o)

                    state_candidates.append(LSTMStateTuple(new_c_tilde, new_h_tilde))
                    cell_input = new_h_tilde

            # Compute value for the update prob
            with tf.variable_scope('state_update_prob'):
                new_update_prob_tilde = rnn_ops.linear(state_candidates[-1].c, 1, True, bias_start=self._update_bias)
                new_update_prob_tilde = tf.sigmoid(new_update_prob_tilde)

            # Compute value for the update gate
            cum_update_prob = cum_update_prob_prev + tf.minimum(update_prob_prev, 1. - cum_update_prob_prev)
            update_gate = _binary_round(cum_update_prob)

            # Apply update gate
            new_states = []
            for idx in range(self._num_layers - 1):
                new_c = update_gate * state_candidates[idx].c + (1. - update_gate) * state[idx].c
                new_h = update_gate * state_candidates[idx].h + (1. - update_gate) * state[idx].h
                new_states.append(LSTMStateTuple(new_c, new_h))
            new_c = update_gate * state_candidates[-1].c + (1. - update_gate) * state[-1].c
            new_h = update_gate * state_candidates[-1].h + (1. - update_gate) * state[-1].h
            new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
            new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob

            new_states.append(SkipLSTMStateTuple(new_c, new_h, new_update_prob, new_cum_update_prob))
            new_output = SkipLSTMOutputTuple(new_h, update_gate)

            return new_output, new_states

    def trainable_initial_state(self, batch_size):
        """
        Create a trainable initial state for the MultiSkipLSTMCell
        :param batch_size: number of samples per batch
        :return: list of SkipLSTMStateTuple
        """
        initial_states = []
        for idx in range(self._num_layers - 1):
            with tf.variable_scope('layer_%d' % (idx + 1)):
                with tf.variable_scope('initial_c'):
                    initial_c = rnn_ops.create_initial_state(batch_size, self._num_units[idx])
                with tf.variable_scope('initial_h'):
                    initial_h = rnn_ops.create_initial_state(batch_size, self._num_units[idx])
                initial_states.append(LSTMStateTuple(initial_c, initial_h))
        with tf.variable_scope('layer_%d' % self._num_layers):
            with tf.variable_scope('initial_c'):
                initial_c = rnn_ops.create_initial_state(batch_size, self._num_units[-1])
            with tf.variable_scope('initial_h'):
                initial_h = rnn_ops.create_initial_state(batch_size, self._num_units[-1])
            with tf.variable_scope('initial_update_prob'):
                initial_update_prob = rnn_ops.create_initial_state(batch_size, 1, trainable=False,
                                                                   initializer=tf.ones_initializer())
            with tf.variable_scope('initial_cum_update_prob'):
                initial_cum_update_prob = rnn_ops.create_initial_state(batch_size, 1, trainable=False,
                                                                       initializer=tf.zeros_initializer())
            initial_states.append(SkipLSTMStateTuple(initial_c, initial_h,
                                                            initial_update_prob, initial_cum_update_prob))
        return initial_states


class SkipGRUCell(tf.contrib.rnn.RNNCell):
    """
    Single Skip GRU cell. Augments the basic GRU cell with a binary output that decides whether to
    update or copy the cell state. The binary neuron is optimized using the Straight Through Estimator.
    """
    def __init__(self, num_units, activation=tf.tanh, layer_norm=False, update_bias=1.0):
        """
        Initialize the Skip GRU cell
        :param num_units: int, the number of units in the GRU cell
        :param activation: activation function of the inner states
        :param layer_norm: bool, whether to use layer normalization
        :param update_bias: float, initial value for the bias added to the update state gate
        """
        self._num_units = num_units
        self._activation = activation
        self._layer_norm = layer_norm
        self._update_bias = update_bias

    @property
    def state_size(self):
        return SkipGRUStateTuple(self._num_units, 1, 1)

    @property
    def output_size(self):
        return SkipGRUOutputTuple(self._num_units, 1)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h_prev, update_prob_prev, cum_update_prob_prev = state

            # Parameters of gates are concatenated into one multiply for efficiency.
            with tf.variable_scope("gates"):
                concat = rnn_ops.linear([inputs, h_prev], 2 * self._num_units, bias=True, bias_start=1.0)

            # r = reset_gate, u = update_gate
            r, u = tf.split(value=concat, num_or_size_splits=2, axis=1)

            if self._layer_norm:
                r = rnn_ops.layer_norm(r, name="r")
                u = rnn_ops.layer_norm(u, name="u")

            # Apply non-linearity after layer normalization
            r = tf.sigmoid(r)
            u = tf.sigmoid(u)

            with tf.variable_scope("candidate"):
                new_c_tilde = self._activation(rnn_ops.linear([inputs, r * h_prev], self._num_units, True))
            new_h_tilde = u * h_prev + (1 - u) * new_c_tilde

            # Compute value for the update prob
            with tf.variable_scope('state_update_prob'):
                new_update_prob_tilde = rnn_ops.linear(new_h_tilde, 1, True, bias_start=self._update_bias)
                new_update_prob_tilde = tf.sigmoid(new_update_prob_tilde)

            # Compute value for the update gate
            cum_update_prob = cum_update_prob_prev + tf.minimum(update_prob_prev, 1. - cum_update_prob_prev)
            update_gate = _binary_round(cum_update_prob)

            # Apply update gate
            new_h = update_gate * new_h_tilde + (1. - update_gate) * h_prev
            new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
            new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob

            new_state = SkipGRUStateTuple(new_h, new_update_prob, new_cum_update_prob)
            new_output = SkipGRUOutputTuple(new_h, update_gate)

            return new_output, new_state

    def trainable_initial_state(self, batch_size):
        """
        Create a trainable initial state for the SkipGRUCell
        :param batch_size: number of samples per batch
        :return: SkipGRUStateTuple
        """
        with tf.variable_scope('initial_h'):
            initial_h = rnn_ops.create_initial_state(batch_size, self._num_units)
        with tf.variable_scope('initial_update_prob'):
            initial_update_prob = rnn_ops.create_initial_state(batch_size, 1, trainable=False,
                                                               initializer=tf.ones_initializer())
        with tf.variable_scope('initial_cum_update_prob'):
            initial_cum_update_prob = rnn_ops.create_initial_state(batch_size, 1, trainable=False,
                                                                   initializer=tf.zeros_initializer())
        return SkipGRUStateTuple(initial_h, initial_update_prob, initial_cum_update_prob)


class MultiSkipGRUCell(tf.contrib.rnn.RNNCell):
    """
    Stack of Skip GRU cells. The selection binary output is computed from the state of the cell on top of
    the stack.
    """
    def __init__(self, num_units, activation=tf.tanh, layer_norm=False, update_bias=1.0):
        """
        Initialize the stack of Skip GRU cells
        :param num_units: list of int, the number of units in each GRU cell
        :param activation: activation function of the inner states
        :param layer_norm: bool, whether to use layer normalization
        :param update_bias: float, initial value for the bias added to the update state gate
        """
        if not isinstance(num_units, list):
            num_units = [num_units]
        self._num_units = num_units
        self._num_layers = len(self._num_units)
        self._activation = activation
        self._layer_norm = layer_norm
        self._update_bias = update_bias

    @property
    def state_size(self):
        return [num_units for num_units in self._num_units[:-1]] + [SkipGRUStateTuple(self._num_units[-1], 1, 1)]

    @property
    def output_size(self):
        return SkipGRUOutputTuple(self._num_units[-1], 1)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            update_prob_prev, cum_update_prob_prev = state[-1].update_prob, state[-1].cum_update_prob
            cell_input = inputs
            state_candidates = []

            # Compute update candidates for all layers
            for idx in range(self._num_layers):
                with tf.variable_scope('layer_%d' % (idx + 1)):
                    if isinstance(state[idx], SkipGRUStateTuple):
                        h_prev = state[idx].h
                    else:
                        h_prev = state[idx]

                    # Parameters of gates are concatenated into one multiply for efficiency.
                    with tf.variable_scope("gates"):
                        concat = rnn_ops.linear([cell_input, h_prev], 2 * self._num_units[idx], bias=True, bias_start=1.0,)

                    # r = reset_gate, u = update_gate
                    r, u = tf.split(value=concat, num_or_size_splits=2, axis=1)

                    if self._layer_norm:
                        r = rnn_ops.layer_norm(r, name="r")
                        u = rnn_ops.layer_norm(u, name="u")

                    # Apply non-linearity after layer normalization
                    r = tf.sigmoid(r)
                    u = tf.sigmoid(u)

                    with tf.variable_scope("candidate"):
                        new_c_tilde = self._activation(rnn_ops.linear([inputs, r * h_prev], self._num_units[idx], True))
                    new_h_tilde = u * h_prev + (1 - u) * new_c_tilde

                    state_candidates.append(new_h_tilde)
                    cell_input = new_h_tilde

            # Compute value for the update prob
            with tf.variable_scope('state_update_prob'):
                new_update_prob_tilde = rnn_ops.linear(state_candidates[-1], 1, True, bias_start=self._update_bias)
                new_update_prob_tilde = tf.sigmoid(new_update_prob_tilde)

            # Compute value for the update gate
            cum_update_prob = cum_update_prob_prev + tf.minimum(update_prob_prev, 1. - cum_update_prob_prev)
            update_gate = _binary_round(cum_update_prob)

            # Apply update gate
            new_states = []
            for idx in range(self._num_layers - 1):
                new_h = update_gate * state_candidates[idx] + (1. - update_gate) * state[idx]
                new_states.append(new_h)
            new_h = update_gate * state_candidates[-1] + (1. - update_gate) * state[-1].h
            new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
            new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob

            new_states.append(SkipGRUStateTuple(new_h, new_update_prob, new_cum_update_prob))
            new_output = SkipGRUOutputTuple(new_h, update_gate)

            return new_output, new_states

    def trainable_initial_state(self, batch_size):
        """
        Create a trainable initial state for the MultiSkipGRUCell
        :param batch_size: number of samples per batch
        :return: list of tensors and SkipGRUStateTuple
        """
        initial_states = []
        for idx in range(self._num_layers - 1):
            with tf.variable_scope('layer_%d' % (idx + 1)):
                with tf.variable_scope('initial_h'):
                    initial_h = rnn_ops.create_initial_state(batch_size, self._num_units[idx])
                initial_states.append(initial_h)
        with tf.variable_scope('layer_%d' % self._num_layers):
            with tf.variable_scope('initial_h'):
                initial_h = rnn_ops.create_initial_state(batch_size, self._num_units[-1])
            with tf.variable_scope('initial_update_prob'):
                initial_update_prob = rnn_ops.create_initial_state(batch_size, 1, trainable=False,
                                                                   initializer=tf.ones_initializer())
            with tf.variable_scope('initial_cum_update_prob'):
                initial_cum_update_prob = rnn_ops.create_initial_state(batch_size, 1, trainable=False,
                                                                       initializer=tf.zeros_initializer())
            initial_states.append(SkipGRUStateTuple(initial_h, initial_update_prob, initial_cum_update_prob))
        return initial_states
