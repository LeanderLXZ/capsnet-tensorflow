import tensorflow as tf
from activation_fn import ActivationFunc


class Conv2Capsule(object):

    def __init__(self, cfg, kernel_size=None, stride=None, depth=None, vec_dim=None, padding=None):
        """
        Initialize conv2caps layer.
        :param kernel_size: size of convolution kernel
        :param stride: stride of convolution kernel
        :param depth: depth of convolution kernel
        :param vec_dim: dimensions of vectors of capsule
        :param padding: padding type of convolution kernel
        """
        self.cfg = cfg
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.vec_dim = vec_dim
        self.padding = padding

    def __call__(self, inputs):
        """
        Convert a convolution layer to capsule layer.
        :param inputs: input tensor with shape: (batch_size, height, width, depth)
        :return: tensor of capsules with shape: (batch_size, num_caps_j, vec_dim_j, 1)
        """
        # Convolution layer
        activation_fn = tf.nn.relu
        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()
        caps = tf.contrib.layers.conv2d(inputs=inputs,
                                        num_outputs=self.depth * self.vec_dim,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.padding,
                                        activation_fn=activation_fn,
                                        weights_initializer=weights_initializer,
                                        biases_initializer=biases_initializer)

        # Reshape and generating a capsule layer
        caps_shape = caps.get_shape().as_list()
        num_capsule = caps_shape[1] * caps_shape[2] * self.depth
        caps = tf.reshape(caps, [self.cfg.BATCH_SIZE, -1, self.vec_dim, 1])
        # caps shape: (batch_size, num_caps_j, vec_dim_j, 1)
        assert caps.get_shape() == (self.cfg.BATCH_SIZE, num_capsule, self.vec_dim, 1), \
            'Wrong shape of caps: {}'.format(caps.get_shape().as_list())

        # Applying activation function
        caps_activated = ActivationFunc.squash(caps, self.cfg.BATCH_SIZE, self.cfg.EPSILON)
        # caps_activated shape: (batch_size, num_caps_j, vec_dim_j, 1)
        assert caps_activated.get_shape() == (self.cfg.BATCH_SIZE, num_capsule, self.vec_dim, 1), \
            'Wrong shape of caps_squashed: {}'.format(caps_activated.get_shape().as_list())

        return caps_activated


class CapsuleLayer(object):

    def __init__(self, cfg, num_caps=None, vec_dim=None, route_epoch=None):
        """
        Initialize capsule layer.
        :param num_caps: number of capsules of this layer
        :param vec_dim: dimensions of vectors of capsules
        :param route_epoch: number of dynamic routing iteration
        """
        self.cfg = cfg
        self.num_caps = num_caps
        self.vec_dim = vec_dim
        self.route_epoch = route_epoch

    def __call__(self, inputs):
        """
        Apply dynamic routing.
        :param inputs: input tensor with shape: (batch_size, num_caps_i, vec_dim_i, 1)
        :return: output tensor with shape (batch_size, num_caps_j, vec_dim_j, 1)
        """
        self.v_j = self.dynamic_routing(inputs, self.num_caps, self.vec_dim, self.route_epoch)

        return self.v_j

    def dynamic_routing(self, inputs, num_caps_j, vec_dim_j, route_epoch):
        """
        Dynamic routing according to Hinton's paper.
        :param inputs: input tensor with shape: (batch_size, num_caps_i, vec_dim_i, 1)
        :param num_caps_j: number of capsules of upper layer
        :param vec_dim_j: dimensions of vectors of upper layer
        :param route_epoch: number of dynamic routing iteration
        :return: output tensor with shape (batch_size, num_caps_j, vec_dim_j, 1)
        """
        inputs_shape = inputs.get_shape().as_list()
        num_caps_i = inputs_shape[1]
        vec_dim_i = inputs_shape[2]
        v_j = None

        # Reshape input tensor
        inputs_shape_new = [self.cfg.BATCH_SIZE, num_caps_i, 1, vec_dim_i, 1]
        inputs = tf.reshape(inputs, shape=inputs_shape_new)
        inputs = tf.tile(inputs, [1, 1, num_caps_j, 1, 1], name='input_tensor')
        # inputs shape: (batch_size, num_caps_i, num_caps_j, vec_dim_i, 1)
        assert inputs.get_shape() == (self.cfg.BATCH_SIZE, num_caps_i, num_caps_j, vec_dim_i, 1), \
            'Wrong shape of inputs: {}'.format(inputs.get_shape().as_list())

        # Initializing weights
        weights_shape = [1, num_caps_i, num_caps_j, vec_dim_j, vec_dim_i]
        # Reuse weights
        weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=self.cfg.STDDEV,
                                                  dtype=tf.float32), name='weights')
        # weights_initializer = tf.truncated_normal_initializer(stddev=self.cfg.STDDEV)
        # weights = tf.get_variable('weights_{}'.format(idx), shape=weights_shape,
        #                           dtype=tf.float32, initializer=weights_initializer)
        weights = tf.tile(weights, [self.cfg.BATCH_SIZE, 1, 1, 1, 1])
        # weights shape: (batch_size, num_caps_i, num_caps_j, vec_dim_j, vec_dim_i)
        assert weights.get_shape() == (self.cfg.BATCH_SIZE, num_caps_i, num_caps_j, vec_dim_j, vec_dim_i), \
            'Wrong shape of weights: {}'.format(weights.get_shape().as_list())

        # Calculating u_hat
        # ( , , , vec_dim_j, vec_dim_i) x ( , , , vec_dim_i, 1)
        # -> ( , , , vec_dim_j, 1) -> squeeze -> ( , , , vec_dim_j)
        u_hat = tf.matmul(weights, inputs, name='u_hat')
        # u_hat shape: (batch_size, num_caps_i, num_caps_j, vec_dim_j, 1)
        assert u_hat.get_shape() == (self.cfg.BATCH_SIZE, num_caps_i, num_caps_j, vec_dim_j, 1), \
            'Wrong shape of u_hat: {}'.format(u_hat.get_shape().as_list())

        # u_hat_stop
        # Do not transfer the gradient of u_hat_stop during back-propagation
        u_hat_stop = tf.stop_gradient(u_hat, name='u_hat_stop')

        # Initializing b_ij
        b_ij = tf.zeros([self.cfg.BATCH_SIZE, num_caps_i, num_caps_j, 1, 1], tf.float32, name='b_ij')
        # b_ij shape: (batch_size, num_caps_i, num_caps_j, 1, 1)
        assert b_ij.get_shape() == (self.cfg.BATCH_SIZE, num_caps_i, num_caps_j, 1, 1), \
            'Wrong shape of b_ij: {}'.format(b_ij.get_shape().as_list())

        def _sum_and_activate(_u_hat, _c_ij, cfg_, name=None):
            """
            Get sum of vectors and apply activation function.
            """
            # Calculating s_j(using u_hat)
            # Using u_hat but not u_hat_stop in order to transfer gradients.
            _s_j = tf.reduce_sum(tf.multiply(_u_hat, _c_ij), axis=1)
            # _s_j shape: (batch_size, num_caps_j, vec_dim_j, 1)
            assert _s_j.get_shape() == (self.cfg.BATCH_SIZE, num_caps_j, vec_dim_j, 1), \
                'Wrong shape of _s_j: {}'.format(_s_j.get_shape().as_list())

            # Applying Squashing
            _v_j = ActivationFunc.squash(_s_j, cfg_.BATCH_SIZE, cfg_.EPSILON)
            # _v_j shape: (batch_size, num_caps_j, vec_dim_j, 1)
            assert _v_j.get_shape() == (self.cfg.BATCH_SIZE, num_caps_j, vec_dim_j, 1), \
                'Wrong shape of _v_j: {}'.format(_v_j.get_shape().as_list())

            _v_j = tf.identity(_v_j, name=name)

            return _v_j

        for iter_route in range(route_epoch):

            with tf.variable_scope('iter_route_{}'.format(iter_route)):

                # Calculate c_ij for every epoch
                c_ij = tf.nn.softmax(b_ij, dim=2)

                # c_ij shape: (batch_size, num_caps_i, num_caps_j, 1, 1)
                assert c_ij.get_shape() == (self.cfg.BATCH_SIZE, num_caps_i, num_caps_j, 1, 1), \
                    'Wrong shape of c_ij: {}'.format(c_ij.get_shape().as_list())

                # Applying back-propagation at last epoch.
                if iter_route == route_epoch - 1:
                    # c_ij_stop
                    # Do not transfer the gradient of c_ij_stop during back-propagation
                    c_ij_stop = tf.stop_gradient(c_ij, name='c_ij_stop')

                    # Calculating s_j(using u_hat) and Applying activation function
                    # Using u_hat but not u_hat_stop in order to transfer gradients.
                    v_j = _sum_and_activate(u_hat, c_ij_stop, self.cfg, name='v_j')

                # Do not apply back-propagation if it is not last epoch.
                else:
                    # Calculating s_j(using u_hat_stop) and Applying activation function
                    # Using u_hat_stop so that the gradient will not be transferred to routing processes.
                    v_j = _sum_and_activate(u_hat_stop, c_ij, self.cfg, name='v_j')

                    # Updating: b_ij <- b_ij + vj x u_ij
                    v_j_reshaped = tf.reshape(v_j, shape=[-1, 1, num_caps_j, 1, vec_dim_j])
                    v_j_reshaped = tf.tile(v_j_reshaped, [1, num_caps_i, 1, 1, 1], name='v_j_reshaped')
                    # v_j_reshaped shape: (batch_size, num_caps_i, num_caps_j, 1, vec_dim_j)
                    assert v_j_reshaped.get_shape() == (self.cfg.BATCH_SIZE, num_caps_i, num_caps_j, 1, vec_dim_j), \
                        'Wrong shape of v_j_reshaped: {}'.format(v_j_reshaped.get_shape().as_list())

                    # ( , , , 1, vec_dim_j) x ( , , , vec_dim_j, 1)
                    # -> squeeze -> (batch_size, num_caps_i, num_caps_j, 1, 1)
                    delta_b_ij = tf.matmul(v_j_reshaped, u_hat_stop, name='delta_b_ij')
                    # delta_b_ij shape: (batch_size, num_caps_i, num_caps_j, 1)
                    assert delta_b_ij.get_shape() == (self.cfg.BATCH_SIZE, num_caps_i, num_caps_j, 1, 1), \
                        'Wrong shape of delta_b_ij: {}'.format(delta_b_ij.get_shape().as_list())

                    b_ij = tf.add(b_ij, delta_b_ij, name='b_ij')
                    # b_ij shape: (batch_size, num_caps_i, num_caps_j, 1, 1)
                    assert b_ij.get_shape() == (self.cfg.BATCH_SIZE, num_caps_i, num_caps_j, 1, 1), \
                        'Wrong shape of b_ij: {}'.format(b_ij.get_shape().as_list())

        # v_j_out shape: (batch_size, num_caps_j, vec_dim_j, 1)
        assert v_j.get_shape() == (self.cfg.BATCH_SIZE, num_caps_j, vec_dim_j, 1), \
            'Wrong shape of v_j_out: {}'.format(v_j.get_shape().as_list())

        return v_j
