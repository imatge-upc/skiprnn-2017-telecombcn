"""
Train RNN models on sequential MNIST, where inputs are processed pixel by pixel.

Results should be reported by evaluating on the test set the model with the best performance on the validation set.
To avoid storing checkpoints and having a separate evaluation script, this script evaluates on both validation and
test set after every epoch.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import time
import datetime

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow_datasets as tfds

from util.misc import *
from util.graph_definition import *

# Task-independent flags
create_generic_flags()

# Task-specific flags
tf.app.flags.DEFINE_string('data_path', '../data', 'Path where the MNIST data will be stored.')

FLAGS = tf.app.flags.FLAGS

# Constants
OUTPUT_SIZE = 10
SEQUENCE_LENGTH = 784
TRAIN_SAMPLES = 55000
VALID_SAMPLES = 5000
TEST_SAMPLES = 10000
NUM_EPOCHS = 600

# Load data
ITERATIONS_PER_EPOCH = int(TRAIN_SAMPLES / FLAGS.batch_size)
VALID_ITERS = int(VALID_SAMPLES / FLAGS.batch_size)
TEST_ITERS = int(TEST_SAMPLES / FLAGS.batch_size)


def input_fn(split):
    if split == 'train':
        dataset = tfds.load('mnist', data_dir=FLAGS.data_path, as_supervised=True, split=tfds.Split.TRAIN)
    elif split == 'val':
        dataset = tfds.load('mnist', data_dir=FLAGS.data_path, as_supervised=True, split=tfds.Split.TEST)  # FIXME
    elif split == 'test':
        dataset = tfds.load('mnist', data_dir=FLAGS.data_path, as_supervised=True, split=tfds.Split.TEST)
    else:
        raise ValueError()

    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    dataset = dataset.map(preprocess)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(10)

    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs


def model_fn(mode, inputs, reuse=False):
    samples = tf.reshape(inputs['images'], (-1, SEQUENCE_LENGTH, 1))
    ground_truth = tf.cast(inputs['labels'], tf.int64)

    is_training = (mode == 'train')

    with tf.variable_scope('model', reuse=reuse):
        cell, initial_state = create_model(model=FLAGS.model,
                                           num_cells=[FLAGS.rnn_cells] * FLAGS.rnn_layers,
                                           batch_size=FLAGS.batch_size)

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, samples, dtype=tf.float32, initial_state=initial_state)

        # Split the outputs of the RNN into the actual outputs and the state update gate
        rrnn_outputs, updated_states = split_rnn_outputs(FLAGS.model, rnn_outputs)

        logits = layers.linear(inputs=rnn_outputs[:, -1, :], num_outputs=OUTPUT_SIZE)
        predictions = tf.argmax(logits, 1)

    # Compute cross-entropy loss
    cross_entropy_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth)
    cross_entropy = tf.reduce_mean(cross_entropy_per_sample)

    # Compute loss for each updated state
    budget_loss = compute_budget_loss(FLAGS.model, cross_entropy, updated_states, FLAGS.cost_per_sample)

    # Combine all losses
    loss = cross_entropy + budget_loss
    loss = tf.reshape(loss, [])

    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truth), tf.float32))

    if is_training:
        # Optimizer
        opt, grads_and_vars = compute_gradients(loss, FLAGS.learning_rate, FLAGS.grad_clip)
        train_fn = opt.apply_gradients(grads_and_vars)

    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['samples'] = samples
    model_spec['labels'] = ground_truth
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['updated_states'] = updated_states

    if is_training:
        model_spec['train_fn'] = train_fn

    return model_spec


def scalar_summary(name, value):
    summary = tf.summary.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    return summary


def train():
    train_inputs = input_fn(split='train')
    valid_inputs = input_fn(split='val')
    test_inputs = input_fn(split='test')

    train_model_spec = model_fn('train', train_inputs)
    valid_model_spec = model_fn('val', valid_inputs, reuse=True)
    test_model_spec = model_fn('test', test_inputs, reuse=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    log_dir = os.path.join(FLAGS.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    valid_writer = tf.summary.FileWriter(log_dir + '/val')
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    # Initialize weights
    sess.run(train_model_spec['variable_init_op'])

    try:
        for epoch in range(NUM_EPOCHS):
            train_fn = train_model_spec['train_fn']

            # Load the training dataset into the pipeline
            sess.run(train_model_spec['iterator_init_op'])

            start_time = time.time()
            for iteration in range(ITERATIONS_PER_EPOCH):
                # Perform SGD update
                sess.run([train_fn])
            duration = time.time() - start_time

            # Evaluate on validation data
            accuracy = valid_model_spec['accuracy']
            loss = valid_model_spec['loss']
            updated_states = valid_model_spec['updated_states']

            # Load the validation dataset into the pipeline
            sess.run(valid_model_spec['iterator_init_op'])

            valid_accuracy, valid_loss, valid_steps = 0, 0, 0
            for _ in range(VALID_ITERS):
                valid_iter_accuracy, valid_iter_loss, valid_used_inputs = sess.run([accuracy, loss, updated_states])
                valid_loss += valid_iter_loss
                valid_accuracy += valid_iter_accuracy
                if valid_used_inputs is not None:
                    valid_steps += compute_used_samples(valid_used_inputs)
                else:
                    valid_steps += SEQUENCE_LENGTH
            valid_accuracy /= VALID_ITERS
            valid_loss /= VALID_ITERS
            valid_steps /= VALID_ITERS

            valid_writer.add_summary(scalar_summary('accuracy', valid_accuracy), epoch)
            valid_writer.add_summary(scalar_summary('loss', valid_loss), epoch)
            valid_writer.add_summary(scalar_summary('used_samples', 100 * valid_steps / SEQUENCE_LENGTH), epoch)
            valid_writer.flush()

            # Evaluate on test data
            accuracy = test_model_spec['accuracy']
            loss = test_model_spec['loss']
            updated_states = test_model_spec['updated_states']

            # Load the test dataset into the pipeline
            sess.run(test_model_spec['iterator_init_op'])

            test_accuracy, test_loss, test_steps = 0, 0, 0
            for _ in range(TEST_ITERS):
                test_iter_accuracy, test_iter_loss, test_used_inputs = sess.run([accuracy, loss, updated_states])
                test_accuracy += test_iter_accuracy
                test_loss += test_iter_loss
                if test_used_inputs is not None:
                    test_steps += compute_used_samples(test_used_inputs)
                else:
                    test_steps += SEQUENCE_LENGTH
            test_accuracy /= TEST_ITERS
            test_loss /= TEST_ITERS
            test_steps /= TEST_ITERS

            test_writer.add_summary(scalar_summary('accuracy', test_accuracy), epoch)
            test_writer.add_summary(scalar_summary('loss', test_loss), epoch)
            test_writer.add_summary(scalar_summary('used_samples', 100 * test_steps / SEQUENCE_LENGTH), epoch)
            test_writer.flush()

            print("Epoch %d/%d, "
                  "duration: %.2f seconds, " 
                  "validation accuracy: %.2f%%, "
                  "validation samples: %.2f (%.2f%%), "
                  "test accuracy: %.2f%%, "
                  "test samples: %.2f (%.2f%%)" % (epoch + 1,
                                                   NUM_EPOCHS,
                                                   duration,
                                                   100. * valid_accuracy,
                                                   valid_steps,
                                                   100. * valid_steps / SEQUENCE_LENGTH,
                                                   100. * test_accuracy,
                                                   test_steps,
                                                   100. * test_steps / SEQUENCE_LENGTH))
    except KeyboardInterrupt:
        pass


def main(argv=None):
    print_setup()
    train()


if __name__ == '__main__':
    tf.app.run()
