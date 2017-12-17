import time
import os
import utils
import capsule_layer
from os.path import isdir
import tensorflow as tf


class CapsNet(object):

    def __init__(self, parameters):

        self.conv_params = parameters['conv_params']
        self.conv2caps_params = parameters['conv2caps_params']
        self.caps_params = parameters['caps_params']
        self.margin_loss_params = parameters['margin_loss_params']
        self.log_path = parameters['margin_loss_params']
        self.training_params = parameters['log_path']

        self.learning_rate = self.training_params['learning_rate']
        self.epochs = self.training_params['epochs']
        self.batch_size = self.training_params['batch_size']
        self.display_step = self.training_params['display_step']

        self.image_size = None

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

    @staticmethod
    def get_inputs(image_size):

        _inputs = tf.placeholder(tf.float32, shape=[None, *image_size], name='inputs')
        _labels = tf.placeholder(tf.float32, shape=[None, None], name='labels')

        return _inputs, _labels

    @staticmethod
    def conv_layer(inputs, kernel_size=None, stride=None, depth=None):

        # Convolution layer
        activation_fn = tf.nn.relu,
        weights_initializer = tf.contrib.initializers.xavier_initializer(),
        biases_initializer = tf.zeros_initializer()
        conv = tf.contrib.layers.conv2d(inputs,
                                        num_outputs=depth,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding='VALID',
                                        activation_fn=activation_fn,
                                        weights_initializer=weights_initializer,
                                        biases_initializer=biases_initializer)

        return conv

    @staticmethod
    def margin_loss(logits, label, m_plus=0.9, m_minus=0.1, lambda_=0.5):

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

    def conv_layers(self, tensor):

        _conv_layers = [tensor]

        for iter_conv, conv_param in enumerate(self.conv_params):
            with tf.name_scope('conv_{}'.format(iter_conv)):
                # conv_param: {'kernel_size': None, 'stride': None, 'depth': None}
                _conv_layer = self.conv_layer(inputs=_conv_layers[iter_conv], **conv_param)
                _conv_layers.append(_conv_layer)

        return _conv_layers[-1]

    def caps_layers(self, tensor):

        _caps_layers = [tensor]

        for iter_caps, caps_param in enumerate(self.caps_params):
            with tf.name_scope('caps_{}'.format(iter_caps)):
                # caps_param: {'num_caps': None, 'vec_dim': None, 'route_epoch': None}
                caps = capsule_layer.CapsuleLayer(**caps_param)
                _caps_layer = caps(_caps_layers[iter_caps])
                _caps_layers.append(_caps_layer)

        return _caps_layers[-1]

    def build_graph(self, image_size=(None, None, None)):

        # Build graph
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Get input placeholders
            inputs, labels = self.get_inputs(image_size)

            # Build convolution layers
            conv = self.conv_layers(inputs)

            # Transform convolution layer's outputs to capsules
            # conv2caps_params: {'kernel_size': None, 'stride': None, 'depth': None, 'vec_dim': None}
            conv2caps_layer = capsule_layer.Conv2Capsule(**self.conv2caps_params)
            conv2caps = conv2caps_layer(conv)

            # Build capsule layers
            # logits shape: (batch_size, num_caps, vec_dim)
            logits = self.caps_layers(conv2caps)

            # margin_loss_params: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
            cost = self.margin_loss(logits, labels, **self.margin_loss_params)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

            # Accuracy
            correct_pred = tf.equal(tf.argmax(utils.get_vec_length(logits), axis=1), tf.argmax(labels, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        return train_graph, inputs, labels, cost, optimizer, accuracy

    def get_batches(self, x, y, batch_size):

        yield x_batch, y_batch

    def print_status(self, sess, merged, cost, accuracy, inputs, labels,
                     start_time, epoch_i, batch_counter, train_writer, valid_writer):

        cost_train_all = []
        cost_valid_all = []
        acc_train_all = []
        acc_valid_all = []

        for train_batch_x, train_batch_y in self.get_batches(self.x_train, self.y_train, self.batch_size):
            summary_train_i, cost_train_i, acc_train_i = \
                sess.run([merged, cost, accuracy], feed_dict={inputs: train_batch_x, labels: train_batch_y})
            train_writer.add_summary(summary_train_i, batch_counter)
            cost_train_all.append(cost_train_i)
            acc_train_all.append(acc_train_i)

        for valid_batch_x, valid_batch_y in self.get_batches(self.x_valid, self.y_valid, self.batch_size):
            summary_valid_i, cost_valid_i, acc_valid_i = \
                sess.run([merged, cost, accuracy], feed_dict={inputs: valid_batch_x, labels: valid_batch_y})
            valid_writer.add_summary(summary_valid_i, batch_counter)
            cost_valid_all.append(cost_valid_i)
            acc_valid_all.append(acc_valid_i)

        cost_train = sum(cost_train_all) / len(cost_train_all)
        cost_valid = sum(cost_valid_all) / len(cost_valid_all)
        acc_train = sum(acc_train_all) / len(acc_train_all)
        acc_valid = sum(acc_valid_all) / len(acc_valid_all)

        total_time = time.time() - start_time

        print('Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
              'Batch: {} |'.format(batch_counter),
              'Time: {:.2f}s |'.format(total_time),
              'Train_Loss: {:.4f} |'.format(cost_train),
              'Valid_Loss: {:.4f}'.format(cost_valid),
              'Train_Accuracy: {:.2f}% |'.format(acc_train*100),
              'Valid_Accuracy: {:.2f}%'.format(acc_valid*100))

    def train(self):

        # Build graph
        train_graph, inputs, labels, cost, optimizer, accuracy = self.build_graph(self.image_size)

        train_log_path = self.log_path + '/train'
        valid_log_path = self.log_path + '/valid'
        utils.check_dir([self.log_path, train_log_path, valid_log_path])

        with tf.Session(graph=train_graph) as sess:

            # Merge all the summaries
            merged = tf.summary.merge_all()

            train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
            valid_writer = tf.summary.FileWriter(valid_log_path)

            sess.run(tf.global_variables_initializer())

            start_time = time.time()
            batch_counter = 0

            for epoch_i in range(self.epochs):

                for x_batch, y_batch in self.get_batches(self.x_train, self.y_train, self.batch_size):

                    batch_counter += 1

                    # Training optimizer
                    sess.run(optimizer, feed_dict={inputs: x_batch, labels: y_batch})

                    if batch_counter % self.display_step == 0:

                        self.print_status(sess, merged, cost, accuracy, inputs, labels,
                                          start_time, epoch_i, batch_counter, train_writer, valid_writer)
