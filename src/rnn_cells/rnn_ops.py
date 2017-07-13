from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.util import nest


def get_variable(name, shape, initializer=None, dtype=tf.float32, device=None):
    """
    Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
      dtype: data type, defaults to tf.float32
      device: device to which the variable will be pinned
    Returns:
      Variable Tensor
    """
    if device is None:
        device = '/cpu:0'
    if initializer is None:
        with tf.device(device):
            var = tf.get_variable(name, shape, dtype=dtype)
    else:
        with tf.device(device):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def linear(args, output_size, bias, weights_init=None, bias_start=0.0):
    """
    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      weights_init: initializer for the weights.
      bias_start: starting value to initialize the gates bias; 0 by default.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = get_variable("Weights", [total_arg_size, output_size], initializer=weights_init)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = get_variable('Biases', [output_size], initializer=tf.constant_initializer(bias_start, dtype=dtype))
        return tf.nn.bias_add(res, biases)


def create_initial_state(batch_size, state_size, trainable=True, initializer=tf.random_normal_initializer()):
    with tf.device('/cpu:0'):
        s = tf.get_variable('initial_state', shape=[1, state_size], dtype=tf.float32, trainable=trainable,
                            initializer=initializer)
        state = tf.tile(s, tf.stack([batch_size] + [1]))
    return state


def layer_norm(x, axes=1, initial_bias_value=0.0, epsilon=1e-3, name="var"):
    """
    Apply layer normalization to x
    Args:
        x: input variable.
        initial_bias_value: initial value for the LN bias.
        epsilon: small constant value to avoid division by zero.
        scope: scope or name for the LN op.
    Returns:
        LN(x) with same shape as x
    """
    if not isinstance(axes, list):
        axes = [axes]

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope):
        with tf.variable_scope(name):
            mean = tf.reduce_mean(x, axes, keep_dims=True)
            variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True))

            with tf.device('/cpu:0'):
                gain = tf.get_variable('gain', x.get_shape().as_list()[1:],
                                       initializer=tf.constant_initializer(1.0))
                bias = tf.get_variable('bias', x.get_shape().as_list()[1:],
                                       initializer=tf.constant_initializer(initial_bias_value))

            return gain * (x - mean) / (variance + epsilon) + bias