import time
import utils
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from capsNet import CapsNet
from config import cfg


class Main(object):

    def __init__(self, model):
        """
        Load data and initialize model.
        :param model: the model which will be trained
        """
        # Global start time
        self.start_time = time.time()

        # Get log path, append information if the directory exist.
        self.log_path = cfg.LOG_PATH
        i_append_info = 1
        while os.path.isdir(self.log_path):
            self.log_path = cfg.LOG_PATH + '({})'.format(i_append_info)
            i_append_info += 1

        # Save config
        utils.check_dir([self.log_path])
        utils.save_config_log(self.log_path)

        # Load data
        utils.thick_line()
        print('Loading data...')
        utils.thin_line()
        x_train = utils.load_data_from_pickle('./data/source_data/mnist/train_image.p')
        y_train = utils.load_data_from_pickle('./data/source_data/mnist/train_label.p')
        x_test = utils.load_data_from_pickle('./data/source_data/mnist/test_image.p')
        y_test = utils.load_data_from_pickle('./data/source_data/mnist/test_label.p')

        # Split training/validation/test set
        x_train = x_train.reshape([-1, 28, 28, 1])
        self.x_valid = x_train[55000:60000]
        assert self.x_valid.shape == (5000, 28, 28, 1), self.x_valid.shape
        self.x_train = x_train[:55000]
        assert self.x_train.shape == (55000, 28, 28, 1), self.x_train.shape
        self.y_valid = y_train[55000:60000]
        assert self.y_valid.shape == (5000, 10), self.y_valid.shape
        self.y_train = y_train[:55000]
        assert self.y_train.shape == (55000, 10), self.y_train.shape
        self.x_test = x_test.reshape([-1, 28, 28, 1])
        assert self.x_test.shape == (10000, 28, 28, 1), self.x_test.shape
        self.y_test = y_test
        assert self.y_test.shape == (10000, 10), self.y_test.shape

        self.n_batch_train = len(self.y_train) // cfg.BATCH_SIZE
        self.n_batch_valid = len(self.y_valid) // cfg.BATCH_SIZE
        self.n_batch_test = len(self.y_test) // cfg.BATCH_SIZE

        # Build graph
        utils.thick_line()
        print('Building graph...')
        self.train_graph, self.inputs, self.labels, self.cost, self.optimizer, self.accuracy = \
            model.build_graph(image_size=self.x_train.shape[1:], num_class=self.y_train.shape[1])

    @staticmethod
    def _get_batches(x, y):
        """
        Split features and labels into batches.
        """
        for start in range(0, len(x)-cfg.BATCH_SIZE, cfg.BATCH_SIZE):
            end = start + cfg.BATCH_SIZE
            yield x[start:end], y[start:end]

    def _display_status(self, sess, x_batch, y_batch, epoch_i, batch_counter):
        """
        Display information during training.
        """
        valid_batch_idx = np.random.choice(range(len(self.x_valid)), cfg.BATCH_SIZE).tolist()
        x_valid_batch = self.x_valid[valid_batch_idx]
        y_valid_batch = self.y_valid[valid_batch_idx]

        cost_train, acc_train = sess.run([self.cost, self.accuracy],
                                         feed_dict={self.inputs: x_batch, self.labels: y_batch})
        cost_valid, acc_valid = sess.run([self.cost, self.accuracy],
                                         feed_dict={self.inputs: x_valid_batch, self.labels: y_valid_batch})

        print('Epoch: {}/{} |'.format(epoch_i + 1, cfg.EPOCHS),
              'Batch: {} |'.format(batch_counter),
              'Time: {:.2f}s |'.format(time.time() - self.start_time),
              'Train_Loss: {:.4f} |'.format(cost_train),
              'Train_Accuracy: {:.2f}% |'.format(acc_train * 100),
              'Valid_Loss: {:.4f} |'.format(cost_valid),
              'Valid_Accuracy: {:.2f}% |'.format(acc_valid * 100))

    def _save_logs(self, sess, train_writer, valid_writer,
                   merged, x_batch, y_batch, epoch_i, batch_counter):
        """
        Save logs and ddd summaries to TensorBoard while training.
        """
        valid_batch_idx = np.random.choice(range(len(self.x_valid)), cfg.BATCH_SIZE).tolist()
        x_valid_batch = self.x_valid[valid_batch_idx]
        y_valid_batch = self.y_valid[valid_batch_idx]

        summary_train, cost_train, acc_train = \
            sess.run([merged, self.cost, self.accuracy],
                     feed_dict={self.inputs: x_batch, self.labels: y_batch})
        summary_valid, cost_valid, acc_valid = \
            sess.run([merged, self.cost, self.accuracy],
                     feed_dict={self.inputs: x_valid_batch, self.labels: y_valid_batch})

        train_writer.add_summary(summary_train, batch_counter)
        valid_writer.add_summary(summary_valid, batch_counter)
        utils.save_log(self.log_path+'train_log.csv', epoch_i,
                       batch_counter, time.time()-self.start_time,
                       cost_train, acc_train, cost_valid, acc_valid)

    def _full_set_eval(self, sess, epoch_i, batch_counter):
        """
        Evaluate on the full data set and print information.
        """
        eval_start_time = time.time()

        utils.thick_line()
        print('Calculating losses using full data set...')
        cost_train_all = []
        cost_valid_all = []
        acc_train_all = []
        acc_valid_all = []

        if cfg.EVAL_WITH_FULL_TRAIN_SET:
            utils.thin_line()
            print('Calculating loss and accuracy on full train set...')
            _train_batch_generator = self._get_batches(self.x_train, self.y_train)
            for _ in tqdm(range(self.n_batch_train), total=self.n_batch_train, ncols=100, unit='batch'):
                train_batch_x, train_batch_y = next(_train_batch_generator)
                cost_train_i, acc_train_i = \
                    sess.run([self.cost, self.accuracy],
                             feed_dict={self.inputs: train_batch_x, self.labels: train_batch_y})
                cost_train_all.append(cost_train_i)
                acc_train_all.append(acc_train_i)

        utils.thin_line()
        print('Calculating loss and accuracy on full valid set...')
        _valid_batch_generator = self._get_batches(self.x_valid, self.y_valid)
        for _ in tqdm(range(self.n_batch_valid), total=self.n_batch_valid, ncols=100, unit='batch'):
            valid_batch_x, valid_batch_y = next(_valid_batch_generator)
            cost_valid_i, acc_valid_i = \
                sess.run([self.cost, self.accuracy],
                         feed_dict={self.inputs: valid_batch_x, self.labels: valid_batch_y})
            cost_valid_all.append(cost_valid_i)
            acc_valid_all.append(acc_valid_i)
        print('Evaluation done! Using time: {:.2f}'.format(time.time() - eval_start_time))

        cost_train = sum(cost_train_all) / len(cost_train_all)
        cost_valid = sum(cost_valid_all) / len(cost_valid_all)
        acc_train = sum(acc_train_all) / len(acc_train_all)
        acc_valid = sum(acc_valid_all) / len(acc_valid_all)

        utils.thin_line()
        print('Epoch: {}/{} |'.format(epoch_i + 1, cfg.EPOCHS),
              'Batch: {} |'.format(batch_counter),
              'Time: {:.2f}s |'.format(time.time() - self.start_time))
        utils.thin_line()
        if cfg.EVAL_WITH_FULL_TRAIN_SET:
            print('Full_Set_Train_Loss: {:.4f}\n'.format(cost_train),
                  'Full_Set_Train_Accuracy: {:.2f}%'.format(acc_train * 100))
        print('Full_Set_Valid_Loss: {:.4f}\n'.format(cost_valid),
              'Full_Set_Valid_Accuracy: {:.2f}%'.format(acc_valid*100))

        file_path = self.log_path+'full_set_eval_log.csv'
        utils.thin_line()
        print('Saving {}...'.format(file_path))
        utils.save_log(file_path, epoch_i, batch_counter, time.time()-self.start_time,
                       cost_train, acc_train, cost_valid, acc_valid)
        utils.thick_line()

    def _test_after_training(self, sess):
        """
        Evaluate on the test set after training.
        """
        test_start_time = time.time()

        utils.thick_line()
        print('Testing on test set...')
        cost_test_all = []
        acc_test_all = []

        if cfg.EVAL_WITH_FULL_TRAIN_SET:
            utils.thin_line()
            print('Calculating loss and accuracy on test set...')
            _test_batch_generator = self._get_batches(self.x_test, self.y_test)
            for _ in tqdm(range(self.n_batch_test), total=self.n_batch_test, ncols=100, unit='batch'):
                test_batch_x, test_batch_y = next(_test_batch_generator)
                cost_test_i, acc_test_i = \
                    sess.run([self.cost, self.accuracy],
                             feed_dict={self.inputs: test_batch_x, self.labels: test_batch_y})
                cost_test_all.append(cost_test_i)
                acc_test_all.append(acc_test_i)

        cost_test = sum(cost_test_all) / len(cost_test_all)
        acc_test = sum(acc_test_all) / len(acc_test_all)

        utils.thin_line()
        print('Test_Loss: {:.4f}\n'.format(cost_test),
              'Test_Accuracy: {:.2f}%'.format(acc_test * 100))
        utils.save_test_log(self.log_path, cost_test, acc_test)

        utils.thin_line()
        print('Testing finished! Using time: {:.2f}'.format(time.time() - test_start_time))

    def train(self):
        """
        Training model
        """
        with tf.Session(graph=self.train_graph) as sess:

            utils.thick_line()
            print('Training...')

            # Merge all the summaries and create writers
            merged = tf.summary.merge_all()
            train_log_path = cfg.SUMMARY_PATH + '/train'
            valid_log_path = cfg.SUMMARY_PATH + '/valid'
            utils.check_dir([cfg.SUMMARY_PATH, train_log_path, valid_log_path])
            train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
            valid_writer = tf.summary.FileWriter(valid_log_path)

            if cfg.FULL_SET_EVAL_STEP is not None:
                if cfg.FULL_SET_EVAL_STEP == 'per_epoch':
                    full_set_eval_step = self.n_batch_train
                else:
                    full_set_eval_step = cfg.FULL_SET_EVAL_STEP

            sess.run(tf.global_variables_initializer())
            batch_counter = 0

            for epoch_i in range(cfg.EPOCHS):

                epoch_start_time = time.time()
                utils.thick_line()
                print('Training on epoch: {}/{}'.format(epoch_i+1, cfg.EPOCHS))

                if cfg.DISPLAY_STEP is not None:
                    for x_batch, y_batch in self._get_batches(self.x_train, self.y_train):

                        batch_counter += 1

                        # Training optimizer
                        sess.run(self.optimizer, feed_dict={self.inputs: x_batch, self.labels: y_batch})

                        if batch_counter % cfg.DISPLAY_STEP == 0:
                            self._display_status(sess, x_batch, y_batch, epoch_i, batch_counter)
                        if cfg.SAVE_LOG_STEP is not None:
                            if batch_counter % cfg.SAVE_LOG_STEP == 0:
                                self._save_logs(sess, train_writer, valid_writer, merged,
                                                x_batch, y_batch, epoch_i, batch_counter)
                        if cfg.FULL_SET_EVAL_STEP is not None:
                            if (epoch_i + 1) % full_set_eval_step == 0:
                                self._full_set_eval(sess, epoch_i, batch_counter)
                else:
                    utils.thin_line()
                    train_batch_generator = self._get_batches(self.x_train, self.y_train)
                    for _ in tqdm(range(self.n_batch_train), total=self.n_batch_train, ncols=100, unit='batch'):

                        batch_counter += 1
                        x_batch, y_batch = next(train_batch_generator)

                        # Training optimizer
                        sess.run(self.optimizer, feed_dict={self.inputs: x_batch, self.labels: y_batch})

                        if cfg.SAVE_LOG_STEP is not None:
                            if batch_counter % cfg.SAVE_LOG_STEP == 0:
                                self._save_logs(sess, train_writer, valid_writer, merged,
                                                x_batch, y_batch, epoch_i, batch_counter)
                        if cfg.FULL_SET_EVAL_STEP is not None:
                            if (epoch_i+1) % full_set_eval_step == 0:
                                self._full_set_eval(sess, epoch_i, batch_counter)

                utils.thin_line()
                print('Epoch done! Using time: {:.2f}'.format(time.time() - epoch_start_time))

            utils.thick_line()
            print('Training finished! Using time: {:.2f}'.format(time.time() - self.start_time))
            utils.thick_line()

            # Evaluate on test set after training
            if cfg.TEST_AFTER_TRAINING:
                self._test_after_training(sess)

        utils.thick_line()
        print('All task finished! Total time: {:.2f}'.format(time.time() - self.start_time))
        utils.thick_line()


if __name__ == '__main__':

    CapsNet_ = CapsNet()
    Main_ = Main(CapsNet_)
    Main_.train()
