from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ActivationFunc(object):

    @staticmethod
    def squash(tensor, batch_size, epsilon):
        """
        Squashing function
        :param tensor: A tensor with shape shape: (batch_size, num_caps, vec_dim, 1).
        :param batch_size: Batch size
        :param epsilon: Add epsilon(a very small number) to zeros
        :return: A tensor with the same shape as input tensor but squashed in 'vec_dim' dimension.
        """
        vec_shape = tensor.get_shape().as_list()
        num_caps = vec_shape[1]
        vec_dim = vec_shape[2]

        vec_squared_norm = tf.reduce_sum(tf.square(tensor), -2, keep_dims=True)
        assert vec_squared_norm.get_shape() == (batch_size, num_caps, 1, 1), \
            'Wrong shape of vec_squared_norm: {}'.format(vec_squared_norm.get_shape().as_list())

        scalar_factor = tf.div(vec_squared_norm, 1 + vec_squared_norm)
        assert scalar_factor.get_shape() == (batch_size, num_caps, 1, 1), \
            'Wrong shape of scalar_factor: {}'.format(scalar_factor.get_shape().as_list())

        unit_vec = tf.div(tensor, tf.sqrt(vec_squared_norm + epsilon))
        assert unit_vec.get_shape() == (batch_size, num_caps, vec_dim, 1), \
            'Wrong shape of unit_vec: {}'.format(unit_vec.get_shape().as_list())

        squashed_vec = tf.multiply(scalar_factor, unit_vec)
        assert squashed_vec.get_shape() == (batch_size, num_caps, vec_dim, 1), \
            'Wrong shape of squashed_vec: {}'.format(squashed_vec.get_shape().as_list())

        return squashed_vec
