import tensorflow as tf
from activation_fn import ActivationFunc


class Conv2Capsule(object):

    def __init__(self, kernel_size=None, stride=None, depth=None, vec_dim=None):

        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.vec_dim = vec_dim

    def __call__(self, inputs):

        # inputs shape: (batch_size, height, width, depth)
        inputs_shape = inputs.get_shape()
        batch_size = inputs_shape[0]

        # Convolution layer
        activation_fn = tf.nn.relu,
        weights_initializer = tf.contrib.initializers.xavier_initializer(),
        biases_initializer = tf.zeros_initializer()
        caps = tf.contrib.layers.conv2d(inputs,
                                        num_outputs=self.depth,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding='VALID',
                                        activation_fn=activation_fn,
                                        weights_initializer=weights_initializer,
                                        biases_initializer=biases_initializer)

        # Reshape and generating a capsule layer
        caps = tf.reshape(caps, (batch_size, -1, self.vec_dim, 1))

        # Applying activation function
        caps = ActivationFunc.squash(caps)

        return caps


class CapsuleLayer(object):

    def __init__(self, num_caps=None, vec_dim=None, route_epoch=None):

        self.num_caps = num_caps
        self.vec_dim = vec_dim
        self.route_epoch = route_epoch

    def __call__(self, inputs):

        # Applying dynamic routing
        self.v_j = self.dynamic_routing(inputs, self.num_caps, self.vec_dim, self.route_epoch)

        return self.v_j

    @staticmethod
    def dynamic_routing(inputs, num_caps_j, vec_dim_j, route_epoch):

        # inputs_shape: (batch_size, num_caps_i, vec_dim_i)
        inputs_shape = inputs.get_shape()
        batch_size = inputs_shape[0]
        num_caps_i = inputs_shape[1]
        vec_dim_i = inputs_shape[2]
        v_j = tf.Variable()

        # Reshape input tensor
        inputs_shape_new = (-1, num_caps_i, 1, vec_dim_i, 1)
        inputs = tf.reshape(inputs, shape=inputs_shape_new)
        inputs = tf.tile(inputs, [1, 1, num_caps_j, 1, 1])
        # inputs shape: (batch_size, num_caps_i, num_caps_j, vec_dim_i, 1)
        assert inputs.get_shape() == (batch_size, num_caps_i, num_caps_j, vec_dim_i, 1)

        # Initializing weights
        weights_shape = (1, num_caps_i, num_caps_j, vec_dim_j, vec_dim_i)
        weights_initializer = tf.truncated_normal_initializer(stddev=0.1)
        # Reuse weights
        weights = tf.get_variable('weights', shape=weights_shape,
                                  dtype=tf.float32, initializer=weights_initializer)
        weights = tf.tile(weights, [batch_size, 1, 1, 1, 1])
        # weights shape: (batch_size, num_caps_i, num_caps_j, vec_dim_j, vec_dim_i)
        assert weights.get_shape() == (batch_size, num_caps_i, num_caps_j, vec_dim_j, vec_dim_i)

        # Calculating u_hat
        # ( , , , vec_dim_j, vec_dim_i) x ( , , , vec_dim_i, 1)
        # -> ( , , , vec_dim_j, 1) -> squeeze -> ( , , , vec_dim_j)
        u_hat = tf.squeeze(tf.matmul(weights, inputs), axis=4, name='u_hat')
        # u_hat shape: (batch_size, num_caps_i, num_caps_j, vec_dim_j)
        assert u_hat.get_shape() == (batch_size, num_caps_i, num_caps_j, vec_dim_j)

        # u_hat_stop
        # Do not transfer the gradient of u_hat_stop during back-propagation
        u_hat_stop = tf.stop_gradient(u_hat, name='u_hat_stop')

        # Initializing b_ij
        b_ij = tf.zeros([batch_size, num_caps_i, num_caps_j], tf.float32, name='b_ij')
        # b_ij shape: (batch_size, num_caps_i, num_caps_j)
        assert b_ij.get_shape() == (batch_size, num_caps_i, num_caps_j)

        def _sum_and_activate(_u_hat, _c_ij):

            # Calculating s_j(using u_hat)
            # Using u_hat but not u_hat_stop in order to transfer gradients.
            _s_j = tf.reduce_sum(tf.multiply(_u_hat, tf.expand_dims(_c_ij, -1)), axis=1)
            # _s_j shape: (batch_size, num_caps_j, vec_dim_j)
            assert _s_j.get_shape() == (batch_size, num_caps_j, vec_dim_j)

            # Applying Squashing
            _v_j = ActivationFunc.squash(_s_j)
            # _v_j shape: (batch_size, num_caps_j, vec_dim_j)
            assert _v_j.get_shape() == (batch_size, num_caps_j, vec_dim_j)

            return _v_j

        for iter_route in range(route_epoch):

            with tf.variable_scope('iter_route_{}'.format(iter_route)):

                # Calculate c_ij for every epoch
                c_ij = tf.nn.softmax(b_ij)

                # c_ij shape: (batch_size, num_caps_i, num_caps_j)
                assert c_ij.get_shape() == (batch_size, num_caps_i, num_caps_j)

                # Applying back-propagation at last epoch.
                if iter_route == route_epoch - 1:
                    # Calculating s_j(using u_hat) and Applying activation function
                    # Using u_hat but not u_hat_stop in order to transfer gradients.
                    v_j = _sum_and_activate(u_hat, c_ij)

                # Do not apply back-propagation if it is not last epoch.
                else:
                    # Calculating s_j(using u_hat_stop) and Applying activation function
                    # Using u_hat_stop so that the gradient will not be transferred to routing processes.
                    v_j = _sum_and_activate(u_hat_stop, c_ij)

                    # Updating: b_ij <- b_ij + vj x u_ij
                    v_j_reshaped = tf.reshape(v_j, shape=[-1, 1, num_caps_j, 1, vec_dim_j])
                    v_j_reshaped = tf.tile(v_j_reshaped, [1, num_caps_i, 1, 1, 1])
                    # v_j_reshaped shape: (batch_size, num_caps_i, num_caps_j, 1, vec_dim_j)
                    assert v_j_reshaped.get_shape() == (batch_size, num_caps_i, num_caps_j, 1, vec_dim_j)

                    u_hat_stop_reshaped = tf.expand_dims(u_hat_stop, -1)
                    # u_hat_stop_reshaped shape: (batch_size, num_caps_i, num_caps_j, vec_dim_j, 1)
                    assert u_hat_stop_reshaped.get_shape() == (batch_size, num_caps_i, num_caps_j, vec_dim_j, 1)
                    # ( , , , 1, vec_dim_j) x ( , , , vec_dim_j, 1) -> squeeze -> (batch_size, num_caps_i, num_caps_j)

                    delta_b_ij = tf.squeeze(tf.matmul(v_j, u_hat_stop))
                    # delta_b_ij shape: (batch_size, num_caps_i, num_caps_j)
                    assert delta_b_ij.get_shape() == (batch_size, num_caps_i, num_caps_j)

                    b_ij = tf.add(b_ij, delta_b_ij)
                    # b_ij shape: (batch_size, num_caps_i, num_caps_j)
                    assert b_ij.get_shape() == (batch_size, num_caps_i, num_caps_j)

        # v_j shape: (batch_size, num_caps_j, vec_dim_j)
        return v_j
