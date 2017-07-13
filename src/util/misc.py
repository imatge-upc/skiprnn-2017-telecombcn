"""
Generic functions that are used in different scripts.
"""

from __future__ import absolute_import
from __future__ import print_function

import types
from decimal import Decimal

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def print_setup(task_specific_setup=None):
    model_dict = {'lstm': 'LSTM', 'gru': 'GRU', 'skip_lstm': 'SkipLSTM', 'skip_gru': 'SkipGRU'}
    print('\n\n\tExperimental setup')
    print('\t------------------\n')
    print('\tModel: %s' % model_dict[FLAGS.model.lower()])
    print('\tNumber of layers: %d' % FLAGS.rnn_layers)
    print('\tNumber of cells: %d' % FLAGS.rnn_cells)
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tLearning rate: %.2E' % Decimal(FLAGS.learning_rate))

    if FLAGS.grad_clip > 0:
        print('\tGradient clipping: %.1f' % FLAGS.grad_clip)
    else:
        print('\tGradient clipping: No')

    if FLAGS.model.lower().startswith('skip'):
        print('\tCost per sample: %.2E' % Decimal(FLAGS.cost_per_sample))

    if isinstance(task_specific_setup, types.FunctionType):
        print('')
        task_specific_setup()

    print('\n\n')


def sample_fraction(used_inputs, batch_size):
    steps = 0.
    for idx in range(batch_size):
        for idt in range(used_inputs.shape[1]):
            steps += used_inputs[idx, idt]
    return steps / batch_size
