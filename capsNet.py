import utils
import capsule_layer
import tensorflow as tf
from config import cfg


class CapsNet(object):

    @staticmethod
    def _get_inputs(image_size, num_class):
        """
        Get input tensors.
        :param image_size: the size of input images, should be 3 dimensional
        :param num_class: number of class of label
        :return: input tensors
        """
        _inputs = tf.placeholder(tf.float32, shape=[cfg.BATCH_SIZE, *image_size], name='inputs')
        _labels = tf.placeholder(tf.float32, shape=[cfg.BATCH_SIZE, num_class], name='labels')

        return _inputs, _labels

    @staticmethod
    def _margin_loss(logits, label, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """
        Calculate margin loss according to Hinton's paper.
        :param logits: output tensor of capsule layers.
        :param label: labels
        :param m_plus: truncation of positive item
        :param m_minus: truncation of negative item
        :param lambda_: lambda
        :return: margin loss
        """
        # L = T_c * max(0, m_plus-||v_c||)^2 + lambda_ * (1-T_c) * max(0, ||v_c||-m_minus)^2

        # logits shape: (batch_size, num_caps, vec_dim)
        logits_shape = logits.get_shape()
        num_caps = logits_shape[1]
        vec_dim = logits_shape[2]

        # logits shape: (batch_size, num_caps, vec_dim)
        assert logits.get_shape() == (cfg.BATCH_SIZE, num_caps, vec_dim), \
            'Wrong shape of logits: {}'.format(logits.get_shape().as_list())

        max_square_plus = tf.square(tf.maximum(0., m_plus - utils.get_vec_length(logits)))
        max_square_minus = tf.square(tf.maximum(0., utils.get_vec_length(logits) - m_minus))
        # max_square_plus & max_plus shape: (batch_size, num_caps)
        assert max_square_plus.get_shape() == (cfg.BATCH_SIZE, num_caps), \
            'Wrong shape of max_square_plus: {}'.format(max_square_plus.get_shape().as_list())

        # label should be one-hot-encoded
        # label shape: (batch_size, num_caps)
        assert label.get_shape() == (cfg.BATCH_SIZE, num_caps)

        loss_c = tf.multiply(label, max_square_plus) + \
            lambda_ * tf.multiply((1-label), max_square_minus)

        # Total margin loss
        margin_loss = tf.reduce_mean(tf.reduce_sum(loss_c, axis=1))

        return margin_loss

    @staticmethod
    def _conv_layer(tensor, kernel_size=None, stride=None, depth=None, padding=None, act_fn='relu', resize=None):
        """
        Single convolution layer
        :param tensor: input tensor
        :param kernel_size: size of convolution kernel
        :param stride: stride of convolution kernel
        :param depth: depth of convolution kernel
        :param padding: padding type of convolution kernel
        :param resize: if resize, resize every image
        :return: output tensor of convolution layer
        """
        # Resize image
        if resize is not None:
            conv = tf.image.resize_nearest_neighbor(tensor, (resize, resize))
        else:
            conv = tensor

        # Convolution layer
        if act_fn == 'relu':
            activation_fn = tf.nn.relu
        elif act_fn == 'sigmoid':
            activation_fn = tf.sigmoid
        elif act_fn is None:
            activation_fn = None
        else:
            raise ValueError('Wrong activation function!')
        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()
        conv = tf.contrib.layers.conv2d(inputs=conv,
                                        num_outputs=depth,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        activation_fn=activation_fn,
                                        weights_initializer=weights_initializer,
                                        biases_initializer=biases_initializer)

        return conv

    @staticmethod
    def _fc_layer(tensor, num_outputs=None, act_fn='relu'):
        """
        Single full_connected layer
        :param tensor: input tensor
        :param num_outputs: hidden units of full_connected layer
        :param act_fn: activation function
        :return: output tensor of full_connected layer
        """
        # Full connected layer
        if act_fn == 'relu':
            activation_fn = tf.nn.relu
        elif act_fn == 'sigmoid':
            activation_fn = tf.sigmoid
        elif act_fn is None:
            activation_fn = None
        else:
            raise ValueError('Wrong activation function!')

        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()
        fc = tf.contrib.layers.fully_connected(inputs=tensor,
                                               num_outputs=num_outputs,
                                               activation_fn=activation_fn,
                                               weights_initializer=weights_initializer,
                                               biases_initializer=biases_initializer)

        return fc

    @staticmethod
    def _conv_transpose_layer(tensor, kernel_size=None, stride=None, depth=None, padding=None):
        """
        Single transpose convolution layer
        :param tensor: input tensor
        :param kernel_size: size of convolution kernel
        :param stride: stride of convolution kernel
        :param depth: depth of convolution kernel
        :param padding: padding type of convolution kernel
        :return: output tensor of transpose convolution layer
        """
        # Transpose convolution layer
        activation_fn = tf.nn.relu
        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()
        conv_t = tf.contrib.layers.conv2d_transpose(inputs=tensor,
                                                    num_outputs=depth,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    activation_fn=activation_fn,
                                                    weights_initializer=weights_initializer,
                                                    biases_initializer=biases_initializer)

        return conv_t

    @staticmethod
    def _caps_layer(tensor, caps_param):
        """
        Single capsule layer
        :param tensor: input tensor
        :param caps_param: parameters of capsule layer
        :return: output tensor of capsule layer
        """
        caps = capsule_layer.CapsuleLayer(**caps_param)

        return caps(tensor)

    def _conv_layers(self, tensor):
        """
        Build multi-convolution layer.
        """
        conv_layers = [tensor]

        for iter_conv, conv_param in enumerate(cfg.CONV_PARAMS):
            with tf.variable_scope('conv_{}'.format(iter_conv)):
                # conv_param: {'kernel_size': None, 'stride': None, 'depth': None, 'padding': 'VALID', 'act_fn': None}
                conv_layer = self._conv_layer(tensor=conv_layers[iter_conv], **conv_param)
                conv_layers.append(conv_layer)

        return conv_layers[-1]

    @staticmethod
    def _conv2caps_layer(tensor, conv2caps_params):
        """
        Build convolution to capsule layer.
        """
        with tf.variable_scope('conv2caps'):
            # conv2caps_params: {'kernel_size': None, 'stride': None,
            #                    'depth': None, 'vec_dim': None, 'padding': 'VALID'}
            conv2caps_layer = capsule_layer.Conv2Capsule(**conv2caps_params)
            conv2caps = conv2caps_layer(tensor)

        return conv2caps

    def _caps_layers(self, tensor):
        """
        Build multi-capsule layer.
        """
        caps_layers = [tensor]

        for iter_caps, caps_param in enumerate(cfg.CAPS_PARAMS):
            with tf.variable_scope('caps_{}'.format(iter_caps)):
                # caps_param: {'num_caps': None, 'vec_dim': None, 'route_epoch': None}
                caps_layer = self._caps_layer(caps_layers[iter_caps], caps_param)
                caps_layers.append(caps_layer)

        # shape: (batch_size, num_caps_j, vec_dim_j, 1) -> (batch_size, num_caps_j, vec_dim_j)
        caps_out = tf.squeeze(caps_layers[-1])

        return caps_out

    def _decoder(self, tensor):
        """
        Decoder of reconstruction layer
        """
        decoder_layers = [tensor]

        # Using full_connected layers
        if cfg.DECODER_TYPE == 'FC':
            for iter_fc, decoder_param in enumerate(cfg.DECODER_PARAMS):
                with tf.variable_scope('decoder_{}'.format(iter_fc)):
                    # decoder_param: {'num_outputs':None, 'act_fn': None}
                    decoder_layer = self._fc_layer(tensor=decoder_layers[iter_fc], **decoder_param)
                    decoder_layers.append(decoder_layer)

        # Using convolution layers
        elif cfg.DECODER_TYPE == 'CONV':
            decoder_layers[0] = \
                tf.reshape(tensor, (cfg.BATCH_SIZE, 4, 4, 1), name='reshape')
            for iter_conv, decoder_param in enumerate(cfg.DECODER_PARAMS):
                with tf.variable_scope('decoder_{}'.format(iter_conv)):
                    # decoder_param:
                    # {'kernel_size': None, 'stride': None, 'depth': None,
                    #  'padding': 'VALID', 'act_fn':None, 'resize': None}
                    decoder_layer = self._conv_layer(tensor=decoder_layers[iter_conv], **decoder_param)
                    decoder_layers.append(decoder_layer)
            decoder_layer = tf.reshape(decoder_layers[-1], (cfg.BATCH_SIZE, -1), name='flatten')
            decoder_layers.append(decoder_layer)

        # Using transpose convolution layers
        elif cfg.DECODER_TYPE == 'CONV_T':
            decoder_layers[0] = \
                tf.reshape(tensor, (cfg.BATCH_SIZE, -1, 1, 1), name='reshape')
            for iter_conv, decoder_param in enumerate(cfg.DECODER_PARAMS):
                with tf.variable_scope('decoder_{}'.format(iter_conv)):
                    # decoder_param: {'kernel_size': None, 'stride': None, 'depth': None, 'padding': 'VALID'}
                    decoder_layer = self._conv_transpose_layer(tensor=decoder_layers[iter_conv], **decoder_param)
                    decoder_layers.append(decoder_layer)
            decoder_layer = tf.reshape(decoder_layers[-1], (cfg.BATCH_SIZE, -1), name='flatten')
            decoder_layers.append(decoder_layer)

        return decoder_layers[-1]

    def _reconstruct_layers(self, tensor, labels):
        """
        Reconstruction layer
        :param tensor: input tensor
        :param labels: labels
        :return: output tensor of reconstruction layer
        """
        with tf.variable_scope('masking'):
            # tensor shape: (batch_size, n_class, vec_dim_j)
            # labels shape: (batch_size, n_class)
            # _masked shape: (batch_size, vec_dim_j)
            _masked = tf.reduce_sum(tf.multiply(tensor, tf.expand_dims(labels, axis=-1)), axis=1)

        with tf.variable_scope('decoder'):
            # _reconstructed shape: (batch_size, image_size*image_size)
            _reconstructed = self._decoder(_masked)

        return _reconstructed

    def build_graph(self, image_size=(None, None, None), num_class=None):
        """
        Build the graph of CapsNet.
        :param image_size: size of input images, should be 3 dimensional
        :param num_class: number of class of label
        :return: tuple of (train_graph, inputs, labels, cost, optimizer, accuracy)
        """
        # Build graph
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Get input placeholders
            inputs, labels = self._get_inputs(image_size, num_class)

            # Build convolution layers
            conv = self._conv_layers(inputs)
            if cfg.SHOW_TRAINING_DETAILS:
                conv = tf.Print(conv, [tf.constant(1)],
                                message="\n[1] CONVOLUTION layers passed...")

            # Transform convolution layer's outputs to capsules
            conv2caps = self._conv2caps_layer(conv, cfg.CONV2CAPS_PARAMS)
            if cfg.SHOW_TRAINING_DETAILS:
                conv2caps = tf.Print(conv2caps, [tf.constant(2)],
                                     message="\n[2] CON2CAPS layers passed...")

            # Build capsule layers
            # logits shape: (batch_size, num_caps, vec_dim)
            logits = self._caps_layers(conv2caps)
            logits = tf.identity(logits, name='logits')
            if cfg.SHOW_TRAINING_DETAILS:
                logits = tf.Print(logits, [tf.constant(3)],
                                  message="\n[3] CAPSULE layers passed...")

            # Build reconstruction part
            if cfg.WITH_RECONSTRUCTION:
                # Reconstruction layers
                # reconstructed shape: (batch_size, image_size*image_size)
                reconstructed = self._reconstruct_layers(logits, labels)
                if cfg.SHOW_TRAINING_DETAILS:
                    reconstructed = tf.Print(reconstructed, [tf.constant(4)],
                                             message="\n[4] RECONSTRUCTION layers passed...")

                with tf.name_scope('reconstructed_images'):
                    reconstructed_images = tf.reshape(reconstructed, shape=[-1, *image_size])

                # Reconstruction cost
                with tf.name_scope('reconstruct_cost'):
                    inputs_flatten = tf.contrib.layers.flatten(inputs)
                    if cfg.DECODER_TYPE != 'fc':
                        reconstructed = tf.contrib.layers.flatten(reconstructed)
                    reconstruct_cost = tf.reduce_mean(tf.square(reconstructed - inputs_flatten))
                    tf.summary.scalar('reconstruct_cost', reconstruct_cost)

                # margin_loss_params: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
                with tf.name_scope('train_cost'):
                    train_cost = self._margin_loss(logits, labels, **cfg.MARGIN_LOSS_PARAMS)
                    tf.summary.scalar('train_cost', train_cost)

                with tf.name_scope('cost'):
                    cost = train_cost + cfg.RECONSTRUCT_COST_SCALE * reconstruct_cost
                    tf.summary.scalar('cost', cost)
                    if cfg.SHOW_TRAINING_DETAILS:
                        cost = tf.Print(cost, [tf.constant(5)],
                                        message="\n[5] COST calculated...")
            else:
                # margin_loss_params: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
                with tf.name_scope('cost'):
                    cost = self._margin_loss(logits, labels, **cfg.MARGIN_LOSS_PARAMS)
                    tf.summary.scalar('cost', cost)
                reconstruct_cost = None
                reconstructed_images = None

                # Optimizer
            if cfg.SHOW_TRAINING_DETAILS:
                cost = tf.Print(cost, [tf.constant(6)],
                                message="\n[6] Updating GRADIENTS...")
            optimizer = tf.train.AdamOptimizer(cfg.LEARNING_RATE).minimize(cost)

            # Accuracy
            with tf.name_scope('accuracy'):
                correct_pred = tf.equal(tf.argmax(utils.get_vec_length(logits), axis=1), tf.argmax(labels, axis=1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar('accuracy', accuracy)

        return train_graph, inputs, labels, cost, optimizer, accuracy, reconstruct_cost, reconstructed_images
