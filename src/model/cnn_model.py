from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model import capsule_layer


class CNNModelBase(object):

    def __init__(self, is_training, cfg):

        self._is_training = is_training
        self._cfg = cfg

    @staticmethod
    def _get_act_fn(act_fn_name):
        """
        Get activation function from name
        """
        if act_fn_name == 'relu':
            activation_fn = tf.nn.relu
        elif act_fn_name == 'sigmoid':
            activation_fn = tf.sigmoid
        elif act_fn_name is None:
            activation_fn = None
        else:
            raise ValueError('Wrong activation function name!')
        return activation_fn

    def _batch_norm(self, x):
        """
        Batch normalization layer
        """
        return tf.contrib.layers.batch_norm(
            input=x,
            decay=self._cfg.BATCH_NORM_DECAY,
            center=True,
            scale=True,
            epsilon=self._cfg.BATCH_NORM_EPSILON,
            is_training=self._is_training,
            fused=True)

    def _fc_layer(self, x, out_dim=None, act_fn_name='relu', use_bias=True, idx=0):
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
            activation_fn = self._get_act_fn(act_fn_name)
            weights_initializer = tf.contrib.layers.xavier_initializer()
            biases_initializer = tf.zeros_initializer() if use_bias else None
            return tf.contrib.layers.fully_connected(
                inputs=x,
                num_outputs=out_dim,
                activation_fn=activation_fn,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer)

    def _conv_layer(self, x, kernel_size=None, stride=None, n_kernel=None, padding=None,
                    act_fn_name='relu', resize=None, use_bias=True, atrous=False, idx=None):
        """
        Single convolution layer

        Args:
            x: input tensor
            kernel_size: size of convolution kernel
            stride: stride of convolution kernel
            n_kernel: number of convolution kernels
            padding: padding type of convolution kernel
            resize: if resize, resize every image
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

            activation_fn = self._get_act_fn(act_fn_name)
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

    def _conv_t_layer(self, x, kernel_size=None, stride=None, n_kernel=None,
                      padding=None, act_fn_name='relu', use_bias=True, idx=None):
        """
        Single transpose convolution layer

        Args:
            x: input tensor
            kernel_size: size of convolution kernel
            stride: stride of convolution kernel
            n_kernel: number of convolution kernels
            padding: padding type of convolution kernel
            use_bias: use bias
            idx: index of layer
        Returns:
            output tensor of convolution layer
        """
        with tf.name_scope('conv_t_{}'.format(idx)):
            activation_fn = self._get_act_fn(act_fn_name)
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

    def _caps_layer(self, x, caps_param, idx=0):
        """
        Single capsule layer

        Args:
            x: input tensor
            caps_param: parameters of capsule layer
        Returns:
            output tensor of capsule layer
        """
        with tf.name_scope('caps_{}'.format(idx)):
            _caps = capsule_layer.CapsuleLayer(self._cfg, **caps_param)
            return _caps(x)

    def _conv2caps_layer(self, x, conv2caps_params):
        """
        Build convolution to capsule layer.

        Args:
            x: input tensor
            conv2caps_params: parameters of conv2caps layer
        Returns:
            output tensor of capsule layer
        """
        with tf.variable_scope('conv2caps'):
            # conv2caps_params: {'kernel_size': None, 'stride': None,
            # 'n_kernel': None, 'vec_dim': None, 'padding': 'VALID'}
            conv2caps_layer = capsule_layer.Conv2Capsule(self._cfg, **conv2caps_params)
            conv2caps = conv2caps_layer(x)

        return conv2caps

    def _multi_conv_layers(self, tensor):
        """
        Build multi-convolution layer.
        """
        conv_layers = [tensor]
        for iter_conv, conv_param in enumerate(self._cfg.CONV_PARAMS):
                # conv_param: {'kernel_size': None, 'stride': None,
                # 'n_kernel': None, 'padding': 'VALID', 'act_fn': None}
                conv_layer = self._conv_layer(x=conv_layers[iter_conv], **conv_param, idx=iter_conv)
                conv_layers.append(conv_layer)

        return conv_layers[-1]

    def _multi_caps_layers(self, tensor):
        """
        Build multi-capsule layer.
        """
        caps_layers = [tensor]
        for iter_caps, caps_param in enumerate(self._cfg.CAPS_PARAMS):
                # caps_param: {'num_caps': None, 'vec_dim': None, 'route_epoch': None}
                caps_layer = self._caps_layer(caps_layers[iter_caps], caps_param, idx=iter_caps)
                caps_layers.append(caps_layer)

        # shape: (batch_size, num_caps_j, vec_dim_j, 1) -> (batch_size, num_caps_j, vec_dim_j)
        return tf.squeeze(caps_layers[-1])
