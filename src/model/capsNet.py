from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model import utils
from model.model_base import ModelBase


class CapsNet(ModelBase):

    def __init__(self, cfg):

        super(CapsNet, self).__init__(cfg)
        self.cfg = cfg

    def _get_inputs(self, image_size, num_class):
        """
        Get input tensors.

        Args:
            image_size: the size of input images, should be 3 dimensional
            num_class: number of class of label
        Returns:
            input tensors
        """
        _inputs = tf.placeholder(tf.float32, shape=[self.cfg.BATCH_SIZE, *image_size], name='inputs')
        _labels = tf.placeholder(tf.float32, shape=[self.cfg.BATCH_SIZE, num_class], name='labels')

        return _inputs, _labels

    def _margin_loss(self, logits, label, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """
        Calculate margin loss according to Hinton's paper.
        L = T_c * max(0, m_plus-||v_c||)^2 + lambda_ * (1-T_c) * max(0, ||v_c||-m_minus)^2

        Args:
            logits: output tensor of capsule layers.
            label: labels
            m_plus: truncation of positive item
            m_minus: truncation of negative item
            lambda_: lambda
        Returns:
            margin loss
        """
        # logits shape: (batch_size, num_caps, vec_dim)
        logits_shape = logits.get_shape()
        num_caps = logits_shape[1]
        vec_dim = logits_shape[2]

        # logits shape: (batch_size, num_caps, vec_dim)
        assert logits.get_shape() == (self.cfg.BATCH_SIZE, num_caps, vec_dim)

        max_square_plus = tf.square(tf.maximum(
            0., m_plus - utils.get_vec_length(logits, self.cfg.BATCH_SIZE, self.cfg.EPSILON)))
        max_square_minus = tf.square(tf.maximum(
            0., utils.get_vec_length(logits, self.cfg.BATCH_SIZE, self.cfg.EPSILON) - m_minus))
        # max_square_plus & max_plus shape: (batch_size, num_caps)
        assert max_square_plus.get_shape() == (self.cfg.BATCH_SIZE, num_caps)

        # label should be one-hot-encoded
        # label shape: (batch_size, num_caps)
        assert label.get_shape() == (self.cfg.BATCH_SIZE, num_caps)

        loss_c = tf.multiply(label, max_square_plus) + \
            lambda_ * tf.multiply((1-label), max_square_minus)

        # Total margin loss
        margin_loss = tf.reduce_mean(tf.reduce_sum(loss_c, axis=1))

        return margin_loss

    def _decoder(self, x):
        """
        Decoder of reconstruction layer
        """
        with tf.variable_scope('decoder'):
            decoder_layers = [x]

            # Using full_connected layers
            if self.cfg.DECODER_TYPE == 'FC':
                for iter_fc, decoder_param in enumerate(self.cfg.DECODER_PARAMS):
                        # decoder_param: {'num_outputs':None, 'act_fn': None}
                        decoder_layer = self._fc_layer(
                            x=decoder_layers[iter_fc], **decoder_param, idx=iter_fc)
                        decoder_layers.append(decoder_layer)

            # Using convolution layers
            elif self.cfg.DECODER_TYPE == 'CONV':
                decoder_layers[0] = tf.reshape(
                    x, (self.cfg.BATCH_SIZE, *self.cfg.CONV_RESHAPE_SIZE, 1), name='reshape')
                for iter_conv, decoder_param in enumerate(self.cfg.DECODER_PARAMS):
                        # decoder_param:
                        # {'kernel_size': None, 'stride': None, 'depth': None,
                        #  'padding': 'VALID', 'act_fn':None, 'resize': None}
                        decoder_layer = self._conv_layer(
                            x=decoder_layers[iter_conv], **decoder_param, idx=iter_conv)
                        decoder_layers.append(decoder_layer)

            # Using transpose convolution layers
            elif self.cfg.DECODER_TYPE == 'CONV_T':
                decoder_layers[0] = tf.reshape(
                    x, (self.cfg.BATCH_SIZE, *self.cfg.CONV_RESHAPE_SIZE, 1), name='reshape')
                for iter_conv_t, decoder_param in enumerate(self.cfg.DECODER_PARAMS):
                        # decoder_param:
                        # {'kernel_size': None, 'stride': None, 'depth': None, 'padding': 'VALID', 'act_fn': None}
                        decoder_layer = self._conv_t_layer(
                            x=decoder_layers[iter_conv_t], **decoder_param, idx=iter_conv_t)
                        decoder_layers.append(decoder_layer)

            return decoder_layers[-1]

    def _reconstruct_layers(self, x, labels):
        """
        Reconstruction layer

        Args:
            x: input tensor
            labels: labels
        Returns:
            output tensor of reconstruction layer
        """
        with tf.variable_scope('masking'):
            # tensor shape: (batch_size, n_class, vec_dim_j)
            # labels shape: (batch_size, n_class)
            # _masked shape: (batch_size, vec_dim_j)
            _masked = tf.reduce_sum(
                tf.multiply(x, tf.expand_dims(labels, axis=-1)), axis=1)

        with tf.variable_scope('decoder'):
            # _reconstructed shape: (batch_size, image_size*image_size)
            _reconstructed = self._decoder(_masked)

        return _reconstructed

    def build_graph(self, image_size=(None, None, None), num_class=None):
        """
        Build the graph of CapsNet.

        Args:
            image_size: size of input images, should be 3 dimensional
            num_class: number of class of label
        Returns:
            tuple of (train_graph, inputs, labels, cost, optimizer, accuracy)
        """
        # Build graph
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Get input placeholders
            inputs, labels = self._get_inputs(image_size, num_class)

            # Build convolution layers
            conv = self._multi_conv_layers(inputs)
            if self.cfg.SHOW_TRAINING_DETAILS:
                conv = tf.Print(conv, [tf.constant(1)],
                                message="\n[1] CONVOLUTION layers passed...")

            # Transform convolution layer's outputs to capsules
            conv2caps = self._conv2caps_layer(conv, self.cfg.CONV2CAPS_PARAMS)
            if self.cfg.SHOW_TRAINING_DETAILS:
                conv2caps = tf.Print(conv2caps, [tf.constant(2)],
                                     message="\n[2] CON2CAPS layers passed...")

            # Build capsule layers
            # logits shape: (batch_size, num_caps, vec_dim)
            logits = self._multi_caps_layers(conv2caps)
            logits = tf.identity(logits, name='logits')
            if self.cfg.SHOW_TRAINING_DETAILS:
                logits = tf.Print(logits, [tf.constant(3)],
                                  message="\n[3] CAPSULE layers passed...")

            # Build reconstruction part
            if self.cfg.WITH_RECONSTRUCTION:
                # Reconstruction layers
                # reconstructed shape: (batch_size, image_size*image_size)
                reconstructed = self._reconstruct_layers(logits, labels)
                if self.cfg.SHOW_TRAINING_DETAILS:
                    reconstructed = tf.Print(reconstructed, [tf.constant(4)],
                                             message="\n[4] RECONSTRUCTION layers passed...")

                reconstructed_images = tf.reshape(reconstructed, shape=[-1, *image_size], name='rec_images')

                # Reconstruction cost
                if self.cfg.RECONSTRUCTION_LOSS == 'mse':
                    inputs_flatten = tf.contrib.layers.flatten(inputs)
                    if self.cfg.DECODER_TYPE != 'FC':
                        reconstructed_ = tf.contrib.layers.flatten(reconstructed)
                    else:
                        reconstructed_ = reconstructed
                    reconstruct_cost = tf.reduce_mean(tf.square(reconstructed_ - inputs_flatten))
                elif self.cfg.RECONSTRUCTION_LOSS == 'cross_entropy':
                    if self.cfg.DECODER_TYPE == 'FC':
                        inputs_ = tf.contrib.layers.flatten(inputs)
                    else:
                        inputs_ = inputs
                    reconstruct_cost = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs_, logits=reconstructed))
                else:
                    reconstruct_cost = None
                reconstruct_cost = tf.identity(reconstruct_cost, name='rec_cost')
                tf.summary.scalar('reconstruct_cost', reconstruct_cost)

                # margin_loss_params: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
                classifier_cost = self._margin_loss(logits, labels, **self.cfg.MARGIN_LOSS_PARAMS)
                classifier_cost = tf.identity(classifier_cost, name='classifier_cost')
                tf.summary.scalar('classifier_cost', classifier_cost)

                cost = classifier_cost + self.cfg.RECONSTRUCT_COST_SCALE * reconstruct_cost
                cost = tf.identity(cost, name='cost')
                tf.summary.scalar('cost', cost)
                if self.cfg.SHOW_TRAINING_DETAILS:
                    cost = tf.Print(cost, [tf.constant(5)],
                                    message="\n[5] COST calculated...")

            else:
                # margin_loss_params: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
                cost = self._margin_loss(logits, labels, **self.cfg.MARGIN_LOSS_PARAMS)
                cost = tf.identity(cost, name='cost')
                tf.summary.scalar('cost', cost)
                classifier_cost = None
                reconstruct_cost = None
                reconstructed_images = None

                # Optimizer
            if self.cfg.SHOW_TRAINING_DETAILS:
                cost = tf.Print(cost, [tf.constant(6)],
                                message="\n[6] Updating GRADIENTS...")
            optimizer = tf.train.AdamOptimizer(self.cfg.LEARNING_RATE).minimize(cost)

            # Accuracy
            correct_pred = tf.equal(tf.argmax(utils.get_vec_length(
                logits, self.cfg.BATCH_SIZE, self.cfg.EPSILON), axis=1), tf.argmax(labels, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', accuracy)

        return train_graph, inputs, labels, cost, optimizer, accuracy, \
            classifier_cost, reconstruct_cost, reconstructed_images
