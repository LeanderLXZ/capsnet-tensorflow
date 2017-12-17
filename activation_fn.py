import tensorflow as tf

epsilon = 1e-9


class ActivationFunc(object):

    def __init__(self):
        pass

    @staticmethod
    def squash(vector):
        """Squashing function
        Args:
            vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
        Returns:
            A tensor with the same shape as vector but squashed in 'vec_len' dimension.
        """
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        squashed_vec = scalar_factor * vector  # element-wise
        return squashed_vec
