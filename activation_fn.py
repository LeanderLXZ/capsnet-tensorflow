import tensorflow as tf
from config import cfg


class ActivationFunc(object):

    @staticmethod
    def squash(tensor):
        """
        Squashing function
        :param tensor: A tensor with shape shape: (batch_size, num_caps, vec_dim, 1).
        :return: A tensor with the same shape as input tensor but squashed in 'vec_dim' dimension.
        """
        vec_shape = tensor.get_shape().as_list()
        num_caps = vec_shape[1]
        vec_dim = vec_shape[2]

        vec_squared_norm = tf.reduce_sum(tf.square(tensor), -2, keep_dims=True)
        assert vec_squared_norm.get_shape() == (cfg.BATCH_SIZE, num_caps, 1, 1), \
            'Wrong shape of vec_squared_norm: {}'.format(vec_squared_norm.get_shape().as_list())

        scalar_factor = tf.div(vec_squared_norm, 1 + vec_squared_norm)
        assert scalar_factor.get_shape() == (cfg.BATCH_SIZE, num_caps, 1, 1), \
            'Wrong shape of scalar_factor: {}'.format(scalar_factor.get_shape().as_list())

        unit_vec = tf.div(tensor, tf.sqrt(vec_squared_norm + cfg.EPSILON))
        assert unit_vec.get_shape() == (cfg.BATCH_SIZE, num_caps, vec_dim, 1), \
            'Wrong shape of unit_vec: {}'.format(unit_vec.get_shape().as_list())

        squashed_vec = tf.multiply(scalar_factor, unit_vec)
        assert squashed_vec.get_shape() == (cfg.BATCH_SIZE, num_caps, vec_dim, 1), \
            'Wrong shape of squashed_vec: {}'.format(squashed_vec.get_shape().as_list())

        return squashed_vec
