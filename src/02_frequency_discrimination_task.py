"""
Train RNN models on the frequency discrimination task. Sine waves with period in [1, 100] are randomly generated and
the network has to classify those with period in [5, 6].

Batches are stratified. Validation is performed on data generated on the fly.
"""

from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

from util.misc import *
from util.graph_definition import *

# Task-independent flags
create_generic_flags()

# Task-specific flags
tf.app.flags.DEFINE_float('sampling_period', 1., "Sampling period, in milliseconds")
tf.app.flags.DEFINE_float('signal_duration', 100., "Signal duration, in milliseconds")
tf.app.flags.DEFINE_integer('validation_batches', 15, "How many batches to use for validation metrics.")
tf.app.flags.DEFINE_integer('evaluate_every', 300, "How often is the model evaluated.")

FLAGS = tf.app.flags.FLAGS

# Constants
START_PERIOD = 0
END_PERIOD = 100
START_TARGET_PERIOD = 5
END_TARGET_PERIOD = 6
INPUT_SIZE = 1
OUTPUT_SIZE = 2
SEQUENCE_LENGTH = int(FLAGS.signal_duration / FLAGS.sampling_period)


def task_setup():
    print('\tSignal duration: %.1fms' % FLAGS.signal_duration)
    print('\tSampling period: %.1fms' % FLAGS.sampling_period)
    print('\tSequence length: %d' % SEQUENCE_LENGTH)
    print('\tTarget periods: (%.0f, %.0f)' % (START_TARGET_PERIOD, END_TARGET_PERIOD))
    print('\tDistractor periods: (%.0f, %.0f) U (%.0f, %.0f)' % (START_PERIOD, START_TARGET_PERIOD,
                                                                 END_TARGET_PERIOD, END_PERIOD))


def generate_example(t, frequency, phase_shift):
    return np.cos(2 * np.pi * frequency * t + phase_shift)


def random_disjoint_interval(start, end, avoid_start, avoid_end):
    """
    Sample a value in [start, avoid_start] U [avoid_end, end] with uniform probability
    """
    val = random.uniform(start, end - (avoid_end - avoid_start))
    if val > avoid_start:
        val += (avoid_end - avoid_start)
    return val


def generate_batch(batch_size, sampling_period, signal_duration, start_period, end_period,
                   start_target_period, end_target_period):
    """
    Generate a stratified batch of examples. There are two classes:
        class 0: sine waves with period in [start_target_period, end_target_period]
        class 1: sine waves with period in [start_period, start_target_period] U [end_target_period, end_period]
    :param batch_size: number of samples per batch
    :param sampling_period: sampling period in milliseconds
    :param signal_duration: duration of the sine waves in milliseconds

    :return x: batch of examples
    :return y: batch of labels
    """
    seq_length = int(signal_duration / sampling_period)

    n_elems = 1
    x = np.empty((batch_size, seq_length, n_elems))
    y = np.empty(batch_size, dtype=np.int64)

    t = np.linspace(0, signal_duration - sampling_period, seq_length)

    for idx in range(int(batch_size/2)):
        period = random.uniform(start_target_period, end_target_period)
        phase_shift = random.uniform(0, period)
        x[idx, :, 0] = generate_example(t, 1./period, phase_shift)
        y[idx] = 0
    for idx in range(int(batch_size/2), batch_size):
        period = random_disjoint_interval(start_period, end_period,
                                          start_target_period, end_target_period)
        phase_shift = random.uniform(0, period)
        x[idx, :, 0] = generate_example(t, 1./period, phase_shift)
        y[idx] = 1
    return x, y


def train():
    samples = tf.placeholder(tf.float32, [None, None, INPUT_SIZE])  # (batch, time, in)
    ground_truth = tf.placeholder(tf.int64, [None])  # (batch, out)

    cell, initial_state = create_model(model=FLAGS.model,
                                       num_cells=[FLAGS.rnn_cells] * FLAGS.rnn_layers,
                                       batch_size=FLAGS.batch_size)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, samples, dtype=tf.float32, initial_state=initial_state)

    # Split the outputs of the RNN into the actual outputs and the state update gate
    rnn_outputs, updated_states = split_rnn_outputs(FLAGS.model, rnn_outputs)

    out = layers.linear(inputs=rnn_outputs[:, -1, :], num_outputs=OUTPUT_SIZE)

    # Compute cross-entropy loss
    cross_entropy_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=ground_truth)
    cross_entropy = tf.reduce_mean(cross_entropy_per_sample)

    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), ground_truth), tf.float32))

    # Compute loss for each updated state
    budget_loss = compute_budget_loss(FLAGS.model, cross_entropy, updated_states, FLAGS.cost_per_sample)

    # Combine all losses
    loss = cross_entropy + budget_loss

    # Optimizer
    opt, grads_and_vars = compute_gradients(loss, FLAGS.learning_rate, FLAGS.grad_clip)
    train_fn = opt.apply_gradients(grads_and_vars)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    try:
        num_iters = 0
        while True:
            # Generate new batch and perform SGD update
            x, y = generate_batch(FLAGS.batch_size,
                                  FLAGS.sampling_period,
                                  FLAGS.signal_duration,
                                  START_PERIOD, END_PERIOD,
                                  START_TARGET_PERIOD, END_TARGET_PERIOD)
            sess.run([train_fn], feed_dict={samples: x, ground_truth: y})
            num_iters += 1

            # Evaluate on validation data generated on the fly
            if num_iters % FLAGS.evaluate_every == 0:
                valid_accuracy, valid_steps = 0., 0.
                for _ in range(FLAGS.validation_batches):
                    valid_x, valid_y = generate_batch(FLAGS.batch_size,
                                                      FLAGS.sampling_period,
                                                      FLAGS.signal_duration,
                                                      START_PERIOD, END_PERIOD,
                                                      START_TARGET_PERIOD, END_TARGET_PERIOD)
                    valid_iter_accuracy, valid_used_inputs = sess.run(
                        [accuracy, updated_states],
                        feed_dict={
                            samples: valid_x,
                            ground_truth: valid_y})
                    valid_accuracy += valid_iter_accuracy
                    if valid_used_inputs is not None:
                        valid_steps += compute_used_samples(valid_used_inputs)
                    else:
                        valid_steps += SEQUENCE_LENGTH
                valid_accuracy /= FLAGS.validation_batches
                valid_steps /= FLAGS.validation_batches
                print("Iteration %d, "
                      "validation accuracy: %.2f%%, "
                      "validation samples: %.2f (%.2f%%)" % (num_iters,
                                                             100. * valid_accuracy,
                                                             valid_steps,
                                                             100. * valid_steps / SEQUENCE_LENGTH))
    except KeyboardInterrupt:
        pass


def main(argv=None):
    print_setup(task_setup)
    train()


if __name__ == '__main__':
    tf.app.run()
