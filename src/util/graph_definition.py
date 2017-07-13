"""
Graph creation functions.
"""


from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from rnn_cells.basic_rnn_cells import BasicLSTMCell, BasicGRUCell
from rnn_cells.skip_rnn_cells import SkipLSTMCell, MultiSkipLSTMCell
from rnn_cells.skip_rnn_cells import SkipGRUCell, MultiSkipGRUCell


def create_generic_flags():
    """
    Create flags which are shared by all experiments
    """
    # Generic flags
    tf.app.flags.DEFINE_string('model', 'lstm', "Select RNN cell: {lstm, gru, skip_lstm, skip_gru}")
    tf.app.flags.DEFINE_integer("rnn_cells", 110, "Number of RNN cells.")
    tf.app.flags.DEFINE_integer("rnn_layers", 1, "Number of RNN layers.")
    tf.app.flags.DEFINE_integer('batch_size', 256, "Batch size.")
    tf.app.flags.DEFINE_float('learning_rate', 0.0001, "Learning rate.")
    tf.app.flags.DEFINE_float('grad_clip', 1., "Clip gradients at this value. Set to <=0 to disable clipping.")

    # Flags for the Skip RNN cells
    tf.app.flags.DEFINE_float('cost_per_sample', 0., "Cost per used sample. Set to 0 to disable this option.")


def compute_gradients(loss, learning_rate, gradient_clipping=-1):
    """
    Create optimizer, compute gradients and (optionally) apply gradient clipping
    """
    opt = tf.train.AdamOptimizer(learning_rate)
    if gradient_clipping > 0:
        vars_to_optimize = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, vars_to_optimize), clip_norm=gradient_clipping)
        grads_and_vars = list(zip(grads, vars_to_optimize))
    else:
        grads_and_vars = opt.compute_gradients(loss)
    return opt, grads_and_vars


def create_model(model, num_cells, batch_size, learn_initial_state=True):
    """
    Returns a tuple of (cell, initial_state) to use with dynamic_rnn.
    If num_cells is an integer, a single RNN cell will be created. If it is a list, a stack of len(num_cells)
    cells will be created.
    """
    if not model in ['lstm', 'gru', 'skip_lstm', 'skip_gru']:
        raise ValueError('The specified model is not supported. Please use {lstm, gru, skip_lstm, skip_gru}.')
    if isinstance(num_cells, list) and len(num_cells) > 1:
        if model == 'skip_lstm':
            cells = MultiSkipLSTMCell(num_cells)
        elif model == 'skip_gru':
            cells = MultiSkipGRUCell(num_cells)
        elif model == 'lstm':
            cell_list = [BasicLSTMCell(n) for n in num_cells]
            cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        elif model == 'gru':
            cell_list = [BasicGRUCell(n) for n in num_cells]
            cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        if learn_initial_state:
            if model == 'skip_lstm' or model == 'skip_gru':
                initial_state = cells.trainable_initial_state(batch_size)
            else:
                initial_state = []
                for idx, cell in enumerate(cell_list):
                    with tf.variable_scope('layer_%d' % (idx + 1)):
                        initial_state.append(cell.trainable_initial_state(batch_size))
                initial_state = tuple(initial_state)
        else:
            initial_state = None
        return cells, initial_state
    else:
        if isinstance(num_cells, list):
            num_cells = num_cells[0]
        if model == 'skip_lstm':
            cell = SkipLSTMCell(num_cells)
        elif model == 'skip_gru':
            cell = SkipGRUCell(num_cells)
        elif model == 'lstm':
            cell = BasicLSTMCell(num_cells)
        elif model == 'gru':
            cell = BasicGRUCell(num_cells)
        if learn_initial_state:
            initial_state = cell.trainable_initial_state(batch_size)
        else:
            initial_state = None
        return cell, initial_state


def using_skip_rnn(model):
    """
    Helper function determining whether a Skip RNN models is being used
    """
    return model.lower() == 'skip_lstm' or model.lower() == 'skip_gru'


def split_rnn_outputs(model, rnn_outputs):
    """
    Split the output of dynamic_rnn into the actual RNN outputs and the state update gate
    """
    if using_skip_rnn(model):
        return rnn_outputs.h, rnn_outputs.state_gate
    else:
        return rnn_outputs, tf.no_op()


def compute_budget_loss(model, loss, updated_states, cost_per_sample):
    """
    Compute penalization term on the number of updated states (i.e. used samples)
    """
    if using_skip_rnn(model):
        return tf.reduce_mean(tf.reduce_sum(cost_per_sample * updated_states, 1), 0)
    else:
        return tf.zeros(loss.get_shape())
