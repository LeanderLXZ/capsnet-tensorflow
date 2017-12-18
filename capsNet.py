import utils
import capsule_layer
import tensorflow as tf
from config import cfg


class CapsNet(object):

    @staticmethod
    def _get_inputs(image_size):

        _inputs = tf.placeholder(tf.float32, shape=[None, *image_size], name='inputs')
        _labels = tf.placeholder(tf.float32, shape=[None, None], name='labels')

        return _inputs, _labels

    @staticmethod
    def _margin_loss(logits, label, m_plus=0.9, m_minus=0.1, lambda_=0.5):

        # L = T_c * max(0, m_plus-||v_c||)^2 + lambda_ * (1-T_c) * max(0, ||v_c||-m_minus)^2

        # logits shape: (batch_size, num_caps, vec_dim)
        logits_shape = logits.get_shape()
        batch_size = logits_shape[0]
        num_caps = logits_shape[1]

        max_square_plus = tf.square(tf.maximum(0., m_plus - utils.get_vec_length(logits)))
        max_square_minus = tf.square(tf.maximum(0., utils.get_vec_length(logits) - m_minus))
        # max_square_plus & max_plus shape: (batch_size, num_caps)
        assert max_square_plus.get_shape() == (batch_size, num_caps)

        # label should be one-hot-encoded
        # label shape: (batch_size, num_caps)
        assert label.get_shape() == (batch_size, num_caps)

        loss_c = tf.multiply(label, max_square_plus) + \
            lambda_ * tf.multiply((1-label), max_square_minus)

        # Total margin loss
        margin_loss = tf.reduce_mean(tf.reduce_sum(loss_c, axis=1))

        return margin_loss

    @staticmethod
    def _conv_layer(tensor, kernel_size=None, stride=None, depth=None):

        # Convolution layer
        activation_fn = tf.nn.relu,
        weights_initializer = tf.contrib.initializers.xavier_initializer(),
        biases_initializer = tf.zeros_initializer()
        conv = tf.contrib.layers.conv2d(inputs=tensor,
                                        num_outputs=depth,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding='VALID',
                                        activation_fn=activation_fn,
                                        weights_initializer=weights_initializer,
                                        biases_initializer=biases_initializer)

        return conv

    @staticmethod
    def _caps_layer(tensor, caps_param):

        caps = capsule_layer.CapsuleLayer(**caps_param)

        return caps(tensor)

    def _conv_layers(self, tensor):

        conv_layers = [tensor]

        for iter_conv, conv_param in enumerate(cfg.CONV_PARAMS):
            with tf.name_scope('conv_{}'.format(iter_conv)):
                # conv_param: {'kernel_size': None, 'stride': None, 'depth': None}
                conv_layer = self._conv_layer(tensor=conv_layers[iter_conv], **conv_param)
                conv_layers.append(conv_layer)

        return conv_layers[-1]

    @staticmethod
    def _conv2caps_layer(tensor, conv2caps_params):

        with tf.name_scope('conv2caps'):
            # conv2caps_params: {'kernel_size': None, 'stride': None, 'depth': None, 'vec_dim': None}
            conv2caps_layer = capsule_layer.Conv2Capsule(**conv2caps_params)
            conv2caps = conv2caps_layer(tensor)

        return conv2caps

    def _caps_layers(self, tensor):

        caps_layers = [tensor]

        for iter_caps, caps_param in enumerate(cfg.CAPS_PARAMS):
            with tf.name_scope('caps_{}'.format(iter_caps)):
                # caps_param: {'num_caps': None, 'vec_dim': None, 'route_epoch': None}
                caps_layer = self._caps_layer(caps_layers[iter_caps], caps_param)
                caps_layers.append(caps_layer)

        return caps_layers[-1]

    def build_graph(self, image_size=(None, None, None)):

        # Build graph
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Get input placeholders
            inputs, labels = self._get_inputs(image_size)

            # Build convolution layers
            conv = self._conv_layers(inputs)

            # Transform convolution layer's outputs to capsules
            conv2caps = self._conv2caps_layer(conv, cfg.CONV2CAPS_PARAMS)

            # Build capsule layers
            # logits shape: (batch_size, num_caps, vec_dim)
            logits = self._caps_layers(conv2caps)

            # margin_loss_params: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
            with tf.name_scope('cost'):
                cost = self._margin_loss(logits, labels, **cfg.MARGIN_LOSS_PARAMS)
                tf.summary.scalar('cost', cost)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(cfg.LEARNING_RATE).minimize(cost)

            # Accuracy
            with tf.name_scope('accuracy'):
                correct_pred = tf.equal(tf.argmax(utils.get_vec_length(logits), axis=1), tf.argmax(labels, axis=1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar('accuracy', accuracy)

        return train_graph, inputs, labels, cost, optimizer, accuracy
