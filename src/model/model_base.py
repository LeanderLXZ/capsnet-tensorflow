from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class ModelBase(object):

  def __init__(self, cfg):

    self.cfg = cfg

  @staticmethod
  def variable_on_cpu(name, shape, initializer, dtype=tf.float32):
    """
    Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
      dtype: data type
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

  @staticmethod
  def _get_act_fn(act_fn):
    """
    Get activation function from name
    """
    if act_fn == 'relu':
      activation_fn = tf.nn.relu
    elif act_fn == 'sigmoid':
      activation_fn = tf.nn.sigmoid
    elif act_fn == 'elu':
      activation_fn = tf.nn.elu
    elif act_fn is None:
      activation_fn = None
    else:
      raise ValueError('Wrong activation function name!')
    return activation_fn

  @staticmethod
  def _batch_norm(x, is_training, batch_norm_decay, batch_norm_epsilon):
    """
    Batch normalization layer
    """
    with tf.name_scope('batch_norm'):
      return tf.contrib.layers.batch_norm(input=x,
                                          decay=batch_norm_decay,
                                          center=True,
                                          scale=True,
                                          epsilon=batch_norm_epsilon,
                                          is_training=is_training,
                                          fused=True)

  @staticmethod
  def _avg_pool(x, pool_size=None, stride=None, padding='SAME'):
    """
    Average pooling
    """
    with tf.name_scope('avg_pool'):
      return tf.layers.average_pooling2d(inputs=x,
                                         pool_size=pool_size,
                                         strides=stride,
                                         padding=padding)

  @staticmethod
  def _global_avg_pool(x):
    """
    Average pooling on full image
    """
    with tf.name_scope('global_avg_pool'):
      assert x.get_shape().ndims == 4
      return tf.reduce_mean(x, [1, 2])

  def _fc_layer(self, x, out_dim=None, act_fn='relu',
                use_bias=True, idx=0):
    """
    Single full_connected layer

    Args:
      x: input tensor
      out_dim: hidden units of full_connected layer
      act_fn: activation function
      use_bias: use bias
      idx: index of layer
    Returns:
      output tensor of full_connected layer
    """
    with tf.name_scope('fc_{}'.format(idx)):
      activation_fn = self._get_act_fn(act_fn)
      weights_initializer = tf.contrib.layers.xavier_initializer()

      if self.cfg.VAR_ON_CPU:
        weights = self.variable_on_cpu(
            name='weights',
            shape=[x.get_shape().as_list()[1], out_dim],
            initializer=weights_initializer,
            dtype=tf.float32)
        biases = self.variable_on_cpu(
            name='biases',
            shape=[out_dim],
            initializer=tf.zeros_initializer(),
            dtype=tf.float32)
        return activation_fn(tf.add(tf.matmul(x, weights), biases))
      else:
        biases_initializer = tf.zeros_initializer() if use_bias else None
        return tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=out_dim,
            activation_fn=activation_fn,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer)

  def _conv_layer(self, x, kernel_size=None, stride=None, n_kernel=None,
                  padding='SAME', act_fn='relu', stddev=None,
                  resize=None, use_bias=True, atrous=False, idx=None):
    """
    Single convolution layer

    Args:
      x: input tensor
      kernel_size: size of convolution kernel
      stride: stride of convolution kernel
      n_kernel: number of convolution kernels
      padding: padding type of convolution kernel
      act_fn: activation function
      stddev: stddev of weights initializer
      resize: if resize, resize every image
      atrous: use atrous convolution
      use_bias: use bias
      idx: index of layer
    Returns:
      output tensor of convolution layer
    """
    with tf.name_scope('conv_{}'.format(idx)):
      # Resize image
      if resize is not None:
        x = tf.image.resize_nearest_neighbor(x, (resize, resize))

      # With atrous
      if not atrous and stride > 1:
        pad = kernel_size - 1
        pad_beg = pad // 2
        pad_end = pad - pad_beg
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        padding = 'VALID'

      activation_fn = self._get_act_fn(act_fn)

      if self.cfg.VAR_ON_CPU:
        kernels = self.variable_on_cpu(
            name='kernels',
            shape=[kernel_size, kernel_size,
                   x.get_shape().as_list()[3], n_kernel],
            initializer=tf.truncated_normal_initializer(
                stddev=stddev, dtype=tf.float32),
            dtype=tf.float32)
        conv = tf.nn.conv2d(input=x,
                            filter=kernels,
                            strides=stride,
                            padding=padding)
        if use_bias:
          biases = self.variable_on_cpu(
              name='biases',
              shape=[n_kernel],
              initializer=tf.zeros_initializer(),
              dtype=tf.float32)
          conv = tf.nn.bias_add(conv, biases)
        return activation_fn(conv)
      else:
        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer() if use_bias else None
        return tf.contrib.layers.conv2d(
            inputs=x,
            num_outputs=n_kernel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation_fn=activation_fn,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer)

  def _conv_t_layer(self, x, kernel_size=None,
                    stride=None, n_kernel=None, padding='SAME',
                    act_fn='relu', use_bias=True, idx=None):
    """
    Single transpose convolution layer

    Args:
      x: input tensor
      kernel_size: size of convolution kernel
      stride: stride of convolution kernel
      n_kernel: number of convolution kernels
      padding: padding type of convolution kernel
      act_fn: activation function
      use_bias: use bias
      idx: index of layer
    Returns:
      output tensor of convolution layer
    """
    with tf.name_scope('conv_t_{}'.format(idx)):
      activation_fn = self._get_act_fn(act_fn)
      weights_initializer = tf.contrib.layers.xavier_initializer()
      biases_initializer = tf.zeros_initializer() if use_bias else None

      return tf.contrib.layers.conv2d_transpose(
          inputs=x,
          num_outputs=n_kernel,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          activation_fn=activation_fn,
          weights_initializer=weights_initializer,
          biases_initializer=biases_initializer)

  def _multi_conv_layers(self, x):
    """
    Build multi-convolution layer.
    """
    conv_layers = [x]
    for iter_conv, conv_param in enumerate(self.cfg.CONV_PARAMS):
      # conv_param: {'kernel_size': None, 'stride': None,
      # 'n_kernel': None, 'padding': 'VALID', 'act_fn': None}
      conv_layer = self._conv_layer(x=conv_layers[iter_conv],
                                    **conv_param,
                                    idx=iter_conv)
      conv_layers.append(conv_layer)

    return conv_layers[-1]

  def _optimizer(self, opt_name='adam', n_train_samples=None, global_step=None):

    if opt_name == 'adam':
      return tf.train.AdamOptimizer(self.cfg.LEARNING_RATE)

    elif opt_name == 'momentum':
      n_batches_per_epoch = \
          n_train_samples // self.cfg.GPU_BATCH_SIZE * self.cfg.GPU_NUMBER
      boundaries = [
          n_batches_per_epoch * x
          for x in np.array(self.cfg.LR_BOUNDARIES, dtype=np.int64)]
      staged_lr = [self.cfg.LEARNING_RATE * x
                   for x in self.cfg.LR_STAGE]
      learning_rate = tf.train.piecewise_constant(
          global_step,
          boundaries, staged_lr)
      return tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=self.cfg.MOMENTUM)

    elif opt_name == 'gd':
      return tf.train.GradientDescentOptimizer(self.cfg.LEARNING_RATE)

    else:
      raise ValueError('Wrong optimizer name!')
