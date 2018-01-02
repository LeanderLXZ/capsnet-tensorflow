import time
import utils
import os
import math
from PIL import Image
from matplotlib import pyplot as plt
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
        i_append_info = 0
        while os.path.isdir(self.log_path):
            i_append_info += 1
            self.log_path = cfg.LOG_PATH + '({})'.format(i_append_info)
        if i_append_info > 0:
            self.summary_path = cfg.SUMMARY_PATH + '({})'.format(i_append_info)
        else:
            self.summary_path = cfg.SUMMARY_PATH

        # Images saving path
        self.img_path = os.path.join(self.log_path, 'images')

        # Save config
        utils.check_dir([self.log_path, self.img_path])
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
        x_train = np.divide(x_train, 255.)
        x_train = x_train.reshape([-1, 28, 28, 1])
        self.x_valid = x_train[55000:60000]
        assert self.x_valid.shape == (5000, 28, 28, 1), self.x_valid.shape
        self.x_train = x_train[:55000]
        assert self.x_train.shape == (55000, 28, 28, 1), self.x_train.shape
        self.y_valid = y_train[55000:60000]
        assert self.y_valid.shape == (5000, 10), self.y_valid.shape
        self.y_train = y_train[:55000]
        assert self.y_train.shape == (55000, 10), self.y_train.shape
        x_test = np.divide(x_test, 255.)
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
        self.train_graph, self.inputs, self.labels, self.cost, self.optimizer, \
            self.accuracy, self.reconstruct_cost, self.reconstructed_images = \
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

        if cfg.WITH_RECONSTRUCTION:
            cost_train, cost_rec_train, acc_train = \
                sess.run([self.cost, self.reconstruct_cost, self.accuracy],
                         feed_dict={self.inputs: x_batch, self.labels: y_batch})
            cost_valid, cost_rec_valid, acc_valid = \
                sess.run([self.cost, self.reconstruct_cost, self.accuracy],
                         feed_dict={self.inputs: x_valid_batch, self.labels: y_valid_batch})
        else:
            cost_train, acc_train = sess.run([self.cost, self.accuracy],
                                             feed_dict={self.inputs: x_batch, self.labels: y_batch})
            cost_valid, acc_valid = sess.run([self.cost, self.accuracy],
                                             feed_dict={self.inputs: x_valid_batch, self.labels: y_valid_batch})
            cost_rec_train, cost_rec_valid = None

        utils.print_status(epoch_i, batch_counter, self.start_time, cost_train,
                           cost_rec_train, acc_train, cost_valid, cost_rec_valid, acc_valid)

    def _save_logs(self, sess, train_writer, valid_writer,
                   merged, x_batch, y_batch, epoch_i, batch_counter):
        """
        Save logs and ddd summaries to TensorBoard while training.
        """
        valid_batch_idx = np.random.choice(range(len(self.x_valid)), cfg.BATCH_SIZE).tolist()
        x_valid_batch = self.x_valid[valid_batch_idx]
        y_valid_batch = self.y_valid[valid_batch_idx]

        if cfg.WITH_RECONSTRUCTION:
            summary_train, cost_train, cost_rec_train, acc_train = \
                sess.run([merged, self.cost, self.reconstruct_cost, self.accuracy],
                         feed_dict={self.inputs: x_batch, self.labels: y_batch})
            summary_valid, cost_valid, cost_rec_valid, acc_valid = \
                sess.run([merged, self.cost, self.reconstruct_cost, self.accuracy],
                         feed_dict={self.inputs: x_valid_batch, self.labels: y_valid_batch})
        else:
            summary_train, cost_train, acc_train = \
                sess.run([merged, self.cost, self.accuracy],
                         feed_dict={self.inputs: x_batch, self.labels: y_batch})
            summary_valid, cost_valid, acc_valid = \
                sess.run([merged, self.cost, self.accuracy],
                         feed_dict={self.inputs: x_valid_batch, self.labels: y_valid_batch})
            cost_rec_train, cost_rec_valid = None

        train_writer.add_summary(summary_train, batch_counter)
        valid_writer.add_summary(summary_valid, batch_counter)
        utils.save_log(os.path.join(self.log_path, 'train_log.csv'),
                       epoch_i+1, batch_counter, time.time()-self.start_time,
                       cost_train, cost_rec_train, acc_train,
                       cost_valid, cost_rec_valid, acc_valid)

    def _eval_on_batches(self, mode, sess, x, y, n_batch, cost_all, rec_cost_all, acc_all, silent=False):
        """
        Calculate losses and accuracies of full train set.
        """
        if not silent:
            utils.thin_line()
            print('Calculating loss and accuracy of full {} set...'.format(mode))
            _batch_generator = self._get_batches(x, y)
            if cfg.WITH_RECONSTRUCTION:
                for _ in tqdm(range(n_batch), total=n_batch, ncols=100, unit=' batches'):
                    x_batch, y_batch = next(_batch_generator)
                    cost_i, rec_cost_i, acc_i = \
                        sess.run([self.cost, self.reconstruct_cost, self.accuracy],
                                 feed_dict={self.inputs: x_batch, self.labels: y_batch})
                    cost_all.append(cost_i)
                    rec_cost_all.append(rec_cost_i)
                    acc_all.append(acc_i)
            else:
                for _ in tqdm(range(n_batch), total=n_batch, ncols=100, unit=' batches'):
                    x_batch, y_batch = next(_batch_generator)
                    cost_i, acc_i = \
                        sess.run([self.cost, self.accuracy],
                                 feed_dict={self.inputs: x_batch, self.labels: y_batch})
                    cost_all.append(cost_i)
                    acc_all.append(acc_i)
                rec_cost_all = None
        else:
            if cfg.WITH_RECONSTRUCTION:
                for x_batch, y_batch in self._get_batches(x, y):
                    cost_i, rec_cost_i, acc_i = \
                        sess.run([self.cost, self.reconstruct_cost, self.accuracy],
                                 feed_dict={self.inputs: x_batch, self.labels: y_batch})
                    cost_all.append(cost_i)
                    rec_cost_all.append(rec_cost_i)
                    acc_all.append(acc_i)
            else:
                for x_batch, y_batch in self._get_batches(x, y):
                    cost_i, acc_i = \
                        sess.run([self.cost, self.accuracy],
                                 feed_dict={self.inputs: x_batch, self.labels: y_batch})
                    cost_all.append(cost_i)
                    acc_all.append(acc_i)
                rec_cost_all = None

        return cost_all, rec_cost_all, acc_all

    def _eval_on_full_set(self, sess, epoch_i, batch_counter, silent=False):
        """
        Evaluate on the full data set and print information.
        """
        eval_start_time = time.time()

        if not silent:
            utils.thick_line()
            print('Calculating losses using full data set...')
        cost_train_all = []
        cost_valid_all = []
        rec_cost_train_all = []
        rec_cost_valid_all = []
        acc_train_all = []
        acc_valid_all = []

        # Calculate losses and accuracies of full train set
        if cfg.EVAL_WITH_FULL_TRAIN_SET:
            cost_train_all, rec_cost_train_all, acc_train_all = \
                self._eval_on_batches('train', sess, self.x_train, self.y_train, self.n_batch_train,
                                      cost_train_all, rec_cost_train_all, acc_train_all, silent=silent)
            cost_train = sum(cost_train_all) / len(cost_train_all)
            acc_train = sum(acc_train_all) / len(acc_train_all)
            rec_cost_train = sum(rec_cost_train_all) / len(rec_cost_train_all)
        else:
            cost_train, rec_cost_train, acc_train = None, None, None

        # Calculate losses and accuracies of full valid set
        cost_valid_all, rec_cost_valid_all, acc_valid_all = \
            self._eval_on_batches('valid', sess, self.x_valid, self.y_valid, self.n_batch_valid,
                                  cost_valid_all, rec_cost_valid_all, acc_valid_all, silent=silent)
        cost_valid = sum(cost_valid_all) / len(cost_valid_all)
        acc_valid = sum(acc_valid_all) / len(acc_valid_all)
        rec_cost_valid = sum(rec_cost_valid_all) / len(rec_cost_valid_all)

        if not silent:
            utils.thin_line()
            print('Epoch: {}/{} |'.format(epoch_i + 1, cfg.EPOCHS),
                  'Batch: {} |'.format(batch_counter),
                  'Time: {:.2f}s |'.format(time.time() - self.start_time))
            utils.thin_line()
            if cfg.EVAL_WITH_FULL_TRAIN_SET:
                print('Full_Set_Train_Loss: {:.4f}'.format(cost_train))
                if cfg.WITH_RECONSTRUCTION:
                    print('Reconstruction_Train_Loss: {:.4f}'.format(rec_cost_train))
                print('Full_Set_Train_Accuracy: {:.2f}%'.format(acc_train * 100))
            print('Full_Set_Valid_Loss: {:.4f}'.format(cost_valid))
            if cfg.WITH_RECONSTRUCTION:
                print('Reconstruction_Valid_Loss: {:.4f}'.format(rec_cost_valid))
            print('Full_Set_Valid_Accuracy: {:.2f}%'.format(acc_valid*100))

        file_path = os.path.join(self.log_path, 'full_set_eval_log.csv')
        if not silent:
            utils.thin_line()
            print('Saving {}...'.format(file_path))
        utils.save_log(file_path, epoch_i+1, batch_counter, time.time()-self.start_time,
                       cost_train, rec_cost_train, acc_train,
                       cost_valid, rec_cost_valid, acc_valid)
        if not silent:
            utils.thin_line()
            print('Evaluation done! Using time: {:.2f}'.format(time.time() - eval_start_time))

    def _save_images(self, sess, x_batch, y_batch, epoch_i, batch_counter):

        rec_images = sess.run(self.reconstructed_images,
                              feed_dict={self.inputs: x_batch, self.labels: y_batch})
        real_images = x_batch

        # Get maximum size for square grid of images
        save_col_size = math.floor(np.sqrt(rec_images.shape[0] * 2))
        save_row_size = save_col_size // 2

        # Scale to 0-255
        rec_images = np.divide(((rec_images - rec_images.min()) * 255),
                               (rec_images.max() - rec_images.min()))

        # Put images in a square arrangement
        rec_images_in_square = np.reshape(rec_images[: save_row_size*save_col_size],
                                          (save_row_size, save_col_size, rec_images.shape[1],
                                           rec_images.shape[2], rec_images.shape[3]))
        real_images_in_square = np.reshape(real_images[:save_col_size * save_row_size],
                                           (save_row_size, save_col_size, real_images.shape[1],
                                            real_images.shape[2], real_images.shape[3]))

        if cfg.DATABASE_NAME == 'mnist':
            mode = 'L'
            rec_images_in_square = np.squeeze(rec_images_in_square, 4)
            real_images_in_square = np.squeeze(real_images_in_square, 4)
        else:
            mode = 'RGB'

        # Combine images to grid image
        gap = 2
        new_im = Image.new(mode, ((rec_images.shape[1]++gap) * save_row_size * 2 - gap,
                                  (rec_images.shape[2]+gap) * save_col_size - gap))
        for row_i in range(save_row_size*2):
            for col_i in range(save_col_size):
                if (row_i+1) % 2 == 0:
                    if mode == 'L':
                        image = rec_images_in_square[(row_i+1)//2-1, col_i, :, :]
                    else:
                        image = rec_images_in_square[(row_i + 1) // 2 - 1, col_i, :, :, :]
                else:
                    if mode == 'L':
                        image = real_images_in_square[int((row_i+1)//2), col_i, :, :]
                    else:
                        image = real_images_in_square[int((row_i + 1) // 2), col_i, :, :, :]
                im = Image.fromarray(image, mode)
                new_im.paste(im, (row_i*(rec_images.shape[1]+gap)-gap,
                                  col_i*(rec_images.shape[2]+gap)-gap))

        new_im.save(os.path.join(self.img_path, 'epoch_{}_batch_{}.jpg'.format(epoch_i, batch_counter)))

    def _test_after_training(self, sess):
        """
        Evaluate on the test set after training.
        """
        test_start_time = time.time()

        utils.thick_line()
        print('Testing on test set...')
        cost_test_all = []
        acc_test_all = []

        utils.thin_line()
        print('Calculating loss and accuracy on test set...')
        _test_batch_generator = self._get_batches(self.x_test, self.y_test)
        for _ in tqdm(range(self.n_batch_test), total=self.n_batch_test, ncols=100, unit=' batches'):
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
            train_log_path = os.path.join(self.summary_path, 'train')
            valid_log_path = os.path.join(self.summary_path, 'valid')
            utils.check_dir([train_log_path, valid_log_path])
            train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
            valid_writer = tf.summary.FileWriter(valid_log_path)

            full_set_eval_in_loop = False
            if cfg.FULL_SET_EVAL_STEP is not None:
                if cfg.FULL_SET_EVAL_STEP != 'per_epoch':
                    full_set_eval_in_loop = True

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
                        if cfg.WITH_RECONSTRUCTION:
                            if cfg.SAVE_IMAGE_STEP is not None:
                                if batch_counter % cfg.SAVE_IMAGE_STEP == 0:
                                    self._save_images(sess, x_batch, y_batch, epoch_i, batch_counter)
                        if full_set_eval_in_loop:
                            if batch_counter % cfg.FULL_SET_EVAL_STEP == 0:
                                self._eval_on_full_set(sess, epoch_i, batch_counter)
                                utils.thick_line()
                else:
                    utils.thin_line()
                    train_batch_generator = self._get_batches(self.x_train, self.y_train)
                    for _ in tqdm(range(self.n_batch_train), total=self.n_batch_train, ncols=100, unit=' batches'):

                        batch_counter += 1
                        x_batch, y_batch = next(train_batch_generator)

                        # Training optimizer
                        sess.run(self.optimizer, feed_dict={self.inputs: x_batch, self.labels: y_batch})

                        if cfg.SAVE_LOG_STEP is not None:
                            if batch_counter % cfg.SAVE_LOG_STEP == 0:
                                self._save_logs(sess, train_writer, valid_writer, merged,
                                                x_batch, y_batch, epoch_i, batch_counter)
                        if cfg.WITH_RECONSTRUCTION:
                            if cfg.SAVE_IMAGE_STEP is not None:
                                if batch_counter % cfg.SAVE_IMAGE_STEP == 0:
                                    self._save_images(sess, x_batch, y_batch, epoch_i, batch_counter)
                        if full_set_eval_in_loop:
                            if batch_counter % cfg.FULL_SET_EVAL_STEP == 0:
                                self._eval_on_full_set(sess, epoch_i, batch_counter, silent=True)

                if cfg.FULL_SET_EVAL_STEP == 'per_epoch':
                    self._eval_on_full_set(sess, epoch_i, batch_counter)

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
