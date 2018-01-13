from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getopt
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from config import config
from model import utils
from multi_gpu.capsNet_distribute import CapsNetDistribute


class Main(object):

  def __init__(self, model, cfg):
    """
    Load data and initialize model.

    Args:
      model: the model which will be trained
    """
    # Global start time
    self.start_time = time.time()

    # Config
    self.cfg = cfg

    # Get log path, append information if the directory exist.
    self.log_path = self.cfg.LOG_PATH
    test_log_path_ = os.path.join(self.cfg.TEST_LOG_PATH, self.cfg.TEST_VERSION)
    i_append_info = 0
    while os.path.isdir(self.log_path):
      i_append_info += 1
      self.log_path = self.cfg.LOG_PATH + '({})'.format(i_append_info)

    if i_append_info > 0:
      self.summary_path = self.cfg.SUMMARY_PATH + '({})'.format(i_append_info)
      self.checkpoint_path = \
          self.cfg.CHECKPOINT_PATH + '({})'.format(i_append_info)
      self.test_log_path = test_log_path_ + '({})'.format(i_append_info)
    else:
      self.summary_path = self.cfg.SUMMARY_PATH
      self.checkpoint_path = self.cfg.CHECKPOINT_PATH
      self.test_log_path = test_log_path_

    # Images saving path
    self.train_image_path = os.path.join(self.log_path, 'images')
    self.test_image_path = os.path.join(self.test_log_path, 'images')

    # Check directory of paths
    utils.check_dir([self.log_path, self.checkpoint_path])
    if self.cfg.WITH_RECONSTRUCTION:
      if self.cfg.SAVE_IMAGE_STEP is not None:
        utils.check_dir([self.train_image_path])

    # Save config
    utils.save_config_log(self.log_path, self.cfg)

    # Load data
    utils.thick_line()
    print('Loading data...')
    utils.thin_line()
    x_train = utils.load_data_from_pickle(
        os.path.join(self.cfg.SOURCE_DATA_PATH, 'mnist/train_image.p'))
    y_train = utils.load_data_from_pickle(
        os.path.join(self.cfg.SOURCE_DATA_PATH, 'mnist/train_label.p'))

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

    # Calculate number of batches
    self.n_batch_train = len(self.y_train) // self.cfg.BATCH_SIZE
    self.n_batch_valid = len(self.y_valid) // self.cfg.BATCH_SIZE

    # Build graph
    utils.thick_line()
    print('Building graph...')
    tf.reset_default_graph()
    self.step, self.train_graph, self.inputs, self.labels, self.optimizer, \
        self.saver, self.summary, self.loss, self.accuracy, self.cls_loss, \
        self.rec_loss, self.rec_images = model.build_graph(
            image_size=self.x_train.shape[1:],
            num_class=self.y_train.shape[1])

  def _display_status(self, sess, x_batch, y_batch, epoch_i, step):
    """
    Display information during training.
    """
    valid_stepdx = np.random.choice(
        range(len(self.x_valid)), self.cfg.BATCH_SIZE).tolist()
    x_valid_batch = self.x_valid[valid_stepdx]
    y_valid_batch = self.y_valid[valid_stepdx]

    if self.cfg.WITH_RECONSTRUCTION:
      loss_train, cls_loss_train, rec_loss_train, acc_train = \
          sess.run([self.loss, self.cls_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch})
      loss_valid, cls_loss_valid, rec_loss_valid, acc_valid = \
          sess.run([self.loss, self.cls_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch})
    else:
      loss_train, acc_train = \
          sess.run([self.loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch})
      loss_valid, acc_valid = \
          sess.run([self.loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch})
      cls_loss_train, rec_loss_train, cls_loss_valid, rec_loss_valid = \
          None, None, None, None

    utils.print_status(
        epoch_i, self.cfg.EPOCHS, step, self.start_time,
        loss_train, cls_loss_train, rec_loss_train, acc_train,
        loss_valid, cls_loss_valid, rec_loss_valid, acc_valid,
        self.cfg.WITH_RECONSTRUCTION)

  def _save_logs(self, sess, train_writer, valid_writer,
                 x_batch, y_batch, epoch_i, step):
    """
    Save logs and ddd summaries to TensorBoard while training.
    """
    valid_stepdx = np.random.choice(
        range(len(self.x_valid)), self.cfg.BATCH_SIZE).tolist()
    x_valid_batch = self.x_valid[valid_stepdx]
    y_valid_batch = self.y_valid[valid_stepdx]

    if self.cfg.WITH_RECONSTRUCTION:
      summary_train, loss_train, cls_loss_train, rec_loss_train, acc_train = \
          sess.run([self.summary, self.loss, self.cls_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch})
      summary_valid, loss_valid, cls_loss_valid, rec_loss_valid, acc_valid = \
          sess.run([self.summary, self.loss, self.cls_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch})
    else:
      summary_train, loss_train, acc_train = \
          sess.run([self.summary, self.loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch})
      summary_valid, loss_valid, acc_valid = \
          sess.run([self.summary, self.loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch})
      cls_loss_train, rec_loss_train, cls_loss_valid, rec_loss_valid = \
          None, None, None, None

    train_writer.add_summary(summary_train, step)
    valid_writer.add_summary(summary_valid, step)
    utils.save_log(
        os.path.join(self.log_path, 'train_log.csv'), epoch_i + 1, step,
        time.time() - self.start_time, loss_train, cls_loss_train,
        rec_loss_train, acc_train, loss_valid, cls_loss_valid, rec_loss_valid,
        acc_valid, self.cfg.WITH_RECONSTRUCTION)

  def _eval_on_batches(self, mode, sess, x, y, n_batch, silent=False):
    """
    Calculate losses and accuracies of full train set.
    """
    loss_all = []
    acc_all = []
    cls_loss_all = []
    rec_loss_all = []

    if not silent:
      utils.thin_line()
      print('Calculating loss and accuracy of full {} set...'.format(mode))
      _batch_generator = utils.get_batches(x, y, self.cfg.BATCH_SIZE)

      if self.cfg.WITH_RECONSTRUCTION:
        for _ in tqdm(range(n_batch), total=n_batch,
                      ncols=100, unit=' batches'):
          x_batch, y_batch = next(_batch_generator)
          loss_i, cls_loss_i, rec_loss_i, acc_i = sess.run(
              [self.loss, self.cls_loss, self.rec_loss, self.accuracy],
              feed_dict={self.inputs: x_batch, self.labels: y_batch})
          loss_all.append(loss_i)
          cls_loss_all.append(cls_loss_i)
          rec_loss_all.append(rec_loss_i)
          acc_all.append(acc_i)
        cls_loss = sum(cls_loss_all) / len(cls_loss_all)
        rec_loss = sum(rec_loss_all) / len(rec_loss_all)
      else:
        for _ in tqdm(range(n_batch), total=n_batch,
                      ncols=100, unit=' batches'):
          x_batch, y_batch = next(_batch_generator)
          loss_i, acc_i = sess.run(
              [self.loss, self.accuracy],
              feed_dict={self.inputs: x_batch, self.labels: y_batch})
          loss_all.append(loss_i)
          acc_all.append(acc_i)
        cls_loss, rec_loss = None, None

    else:
      if self.cfg.WITH_RECONSTRUCTION:
        for x_batch, y_batch in utils.get_batches(x, y, self.cfg.BATCH_SIZE):
          loss_i, cls_loss_i, rec_loss_i, acc_i = sess.run(
              [self.loss, self.cls_loss, self.rec_loss, self.accuracy],
              feed_dict={self.inputs: x_batch, self.labels: y_batch})
          loss_all.append(loss_i)
          cls_loss_all.append(cls_loss_i)
          rec_loss_all.append(rec_loss_i)
          acc_all.append(acc_i)
        cls_loss = sum(cls_loss_all) / len(cls_loss_all)
        rec_loss = sum(rec_loss_all) / len(rec_loss_all)
      else:
        for x_batch, y_batch in utils.get_batches(x, y, self.cfg.BATCH_SIZE):
          loss_i, acc_i = sess.run(
              [self.loss, self.accuracy],
              feed_dict={self.inputs: x_batch, self.labels: y_batch})
          loss_all.append(loss_i)
          acc_all.append(acc_i)
        cls_loss, rec_loss = None, None

    loss = sum(loss_all) / len(loss_all)
    accuracy = sum(acc_all) / len(acc_all)

    return loss, cls_loss, rec_loss, accuracy

  def _eval_on_full_set(self, sess, epoch_i, step, silent=False):
    """
    Evaluate on the full data set and print information.
    """
    eval_start_time = time.time()

    if not silent:
      utils.thick_line()
      print('Calculating losses using full data set...')

    # Calculate losses and accuracies of full train set
    if self.cfg.EVAL_WITH_FULL_TRAIN_SET:
      loss_train, cls_loss_train, rec_loss_train, acc_train = \
          self._eval_on_batches('train', sess, self.x_train, self.y_train,
                                self.n_batch_train, silent=silent)
    else:
      loss_train, cls_loss_train, rec_loss_train, acc_train = \
          None, None, None, None

    # Calculate losses and accuracies of full valid set
    loss_valid, cls_loss_valid, rec_loss_valid, acc_valid = \
        self._eval_on_batches('valid', sess, self.x_valid, self.y_valid,
                              self.n_batch_valid, silent=silent)

    if not silent:
      utils.print_full_set_eval(
          epoch_i, self.cfg.EPOCHS, step, self.start_time,
          loss_train, cls_loss_train, rec_loss_train, acc_train,
          loss_valid, cls_loss_valid, rec_loss_valid, acc_valid,
          self.cfg.EVAL_WITH_FULL_TRAIN_SET, self.cfg.WITH_RECONSTRUCTION)

    file_path = os.path.join(self.log_path, 'full_set_eval_log.csv')
    if not silent:
      utils.thin_line()
      print('Saving {}...'.format(file_path))
    utils.save_log(
      file_path, epoch_i + 1, step, time.time() - self.start_time,
      loss_train, cls_loss_train, rec_loss_train, acc_train,
      loss_valid, cls_loss_valid, rec_loss_valid, acc_valid,
      self.cfg.WITH_RECONSTRUCTION)

    if not silent:
      utils.thin_line()
      print('Evaluation done! Using time: {:.2f}'
            .format(time.time() - eval_start_time))

  def _save_images(self, sess, img_path, x_batch, y_batch,
                   step, silent=False, epoch_i=None):
    """
    Save reconstruction images.
    """
    rec_images = sess.run(
        self.rec_images, feed_dict={self.inputs: x_batch,
                                    self.labels: y_batch})

    # Get maximum size for square grid of images
    save_col_size = math.floor(np.sqrt(rec_images.shape[0] * 2))
    if save_col_size > self.cfg.MAX_IMAGE_IN_COL:
      save_col_size = self.cfg.MAX_IMAGE_IN_COL
    save_row_size = save_col_size // 2

    # Scale to 0-255
    rec_images = np.divide(((rec_images - rec_images.min()) * 255),
                           (rec_images.max() - rec_images.min()))
    real_images = np.divide(((x_batch - x_batch.min()) * 255),
                            (x_batch.max() - x_batch.min()))

    # Put images in a square arrangement
    rec_images_in_square = np.reshape(
        rec_images[: save_row_size * save_col_size],
        (save_row_size, save_col_size, rec_images.shape[1],
         rec_images.shape[2], rec_images.shape[3])).astype(np.uint8)
    real_images_in_square = np.reshape(
        real_images[: save_row_size * save_col_size],
        (save_row_size, save_col_size, real_images.shape[1],
         real_images.shape[2], real_images.shape[3])).astype(np.uint8)

    if self.cfg.DATABASE_NAME == 'mnist':
      mode = 'L'
      rec_images_in_square = np.squeeze(rec_images_in_square, 4)
      real_images_in_square = np.squeeze(real_images_in_square, 4)
    else:
      mode = 'RGB'

    # Combine images to grid image
    thin_gap = 1
    thick_gap = 3
    avg_gap = (thin_gap + thick_gap) / 2
    new_im = Image.new(mode, (
        int((rec_images.shape[2] + thin_gap) *
            save_col_size - thin_gap + thick_gap * 2),
        int((rec_images.shape[1] + avg_gap) *
            save_row_size * 2 + thick_gap)), 'white')

    for row_i in range(save_row_size * 2):
      for col_i in range(save_col_size):
        if (row_i + 1) % 2 == 0:  # Odd
          if mode == 'L':
            image = rec_images_in_square[(row_i + 1) // 2 - 1, col_i, :, :]
          else:
            image = rec_images_in_square[(row_i + 1) // 2 - 1, col_i, :, :, :]
          im = Image.fromarray(image, mode)
          new_im.paste(im, (
              int(col_i * (rec_images.shape[2] + thin_gap) + thick_gap),
              int(row_i * rec_images.shape[1] + (row_i + 1) * avg_gap)))
        else:  # Even
          if mode == 'L':
            image = real_images_in_square[int((row_i + 1) // 2), col_i, :, :]
          else:
            image = real_images_in_square[int((row_i + 1) // 2), col_i, :, :, :]
          im = Image.fromarray(image, mode)
          new_im.paste(im, (
              int(col_i * (rec_images.shape[2] + thin_gap) + thick_gap),
              int(row_i * (rec_images.shape[1] + avg_gap) + thick_gap)))

    if epoch_i is None:
      save_image_path = os.path.join(
          img_path, 'batch_{}.jpg'.format(step))
    else:
      save_image_path = os.path.join(
          img_path,
          'epoch_{}_batch_{}.jpg'.format(epoch_i, step))
    if not silent:
      utils.thin_line()
      print('Saving image to {}...'.format(save_image_path))
    new_im.save(save_image_path)

  def _save_model(self, sess, saver, step, silent=False):
    """
    Save model.
    """
    save_path = os.path.join(self.checkpoint_path, 'model.ckpt')
    if not silent:
      utils.thin_line()
      print('Saving model to {}...'.format(save_path))
    saver.save(sess, save_path, global_step=step)

  def _test_after_training(self, sess):
    """
    Evaluate on the test set after training.
    """
    test_start_time = time.time()

    utils.thick_line()
    print('Testing...')

    # Check directory of paths
    utils.check_dir([self.test_log_path])
    if self.cfg.TEST_WITH_RECONSTRUCTION:
      if self.cfg.TEST_SAVE_IMAGE_STEP is not None:
        utils.check_dir([self.test_image_path])

    # Load data
    utils.thin_line()
    print('Loading test set...')
    utils.thin_line()
    x_test = utils.load_data_from_pickle(
        './data/source_data/mnist/test_image.p')
    y_test = utils.load_data_from_pickle(
        './data/source_data/mnist/test_label.p')
    x_test = np.divide(x_test, 255.)
    x_test = x_test.reshape([-1, 28, 28, 1])
    assert x_test.shape == (10000, 28, 28, 1), x_test.shape
    assert y_test.shape == (10000, 10), y_test.shape
    n_batch_test = len(y_test) // self.cfg.BATCH_SIZE

    utils.thin_line()
    print('Calculating loss and accuracy on test set...')
    loss_test_all = []
    acc_test_all = []
    cls_loss_test_all = []
    rec_loss_test_all = []
    step = 0
    _test_batch_generator = utils.get_batches(
        x_test, y_test, self.cfg.BATCH_SIZE)

    if self.cfg.TEST_WITH_RECONSTRUCTION:
      for _ in tqdm(range(n_batch_test), total=n_batch_test,
                    ncols=100, unit=' batches'):
        step += 1
        test_batch_x, test_batch_y = next(_test_batch_generator)
        loss_test_i, cls_loss_i, rec_loss_i, acc_test_i = sess.run(
            [self.loss, self.cls_loss, self.rec_loss, self.accuracy],
            feed_dict={self.inputs: test_batch_x, self.labels: test_batch_y})
        loss_test_all.append(loss_test_i)
        acc_test_all.append(acc_test_i)
        cls_loss_test_all.append(cls_loss_i)
        rec_loss_test_all.append(rec_loss_i)

        # Save reconstruct images
        if self.cfg.TEST_SAVE_IMAGE_STEP is not None:
          if step % self.cfg.TEST_SAVE_IMAGE_STEP == 0:
            self._save_images(sess, self.test_image_path, test_batch_x,
                              test_batch_y, step, silent=False)

      cls_loss_test = sum(cls_loss_test_all) / len(cls_loss_test_all)
      rec_loss_test = sum(rec_loss_test_all) / len(rec_loss_test_all)

    else:
      for _ in tqdm(range(n_batch_test), total=n_batch_test,
                    ncols=100, unit=' batches'):
        test_batch_x, test_batch_y = next(_test_batch_generator)
        loss_test_i, acc_test_i = sess.run(
            [self.loss, self.accuracy],
            feed_dict={self.inputs: test_batch_x, self.labels: test_batch_y})
        loss_test_all.append(loss_test_i)
        acc_test_all.append(acc_test_i)
      cls_loss_test, rec_loss_test = None, None

    loss_test = sum(loss_test_all) / len(loss_test_all)
    acc_test = sum(acc_test_all) / len(acc_test_all)

    # Print losses and accuracy
    utils.thin_line()
    print('Test_Loss: {:.4f}\n'.format(loss_test),
          'Test_Accuracy: {:.2f}%'.format(acc_test * 100))
    if self.cfg.TEST_WITH_RECONSTRUCTION:
      utils.thin_line()
      print('Test_Train_Loss: {:.4f}\n'.format(cls_loss_test),
            'Test_Reconstruction_Loss: {:.4f}'.format(rec_loss_test))

    # Save test log
    utils.save_test_log(
        self.test_log_path, loss_test, acc_test, cls_loss_test,
        rec_loss_test, self.cfg.TEST_WITH_RECONSTRUCTION)

    utils.thin_line()
    print('Testing finished! Using time: {:.2f}'
          .format(time.time() - test_start_time))

  def train(self):
    """
    Training model
    """
    session_cfg = tf.ConfigProto()
    session_cfg.gpu_options.allow_growth = True
    with tf.Session(graph=self.train_graph, config=session_cfg) as sess:
      with tf.device('/cpu:0'):

        utils.thick_line()
        print('Training...')

        # Merge all the summaries and create writers
        train_log_path = os.path.join(self.summary_path, 'train')
        valid_log_path = os.path.join(self.summary_path, 'valid')
        utils.check_dir([train_log_path, valid_log_path])
        train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_log_path)

        sess.run(tf.global_variables_initializer())
        step = 0

        for epoch_i in range(self.cfg.EPOCHS):

          epoch_start_time = time.time()
          utils.thick_line()
          print('Training on epoch: {}/{}'.format(epoch_i + 1, self.cfg.EPOCHS))

          if self.cfg.DISPLAY_STEP is not None:
            for x_batch, y_batch in utils.get_batches(self.x_train,
                                                      self.y_train,
                                                      self.cfg.BATCH_SIZE):
              step += 1

              # Training optimizer
              sess.run(self.optimizer, feed_dict={self.inputs: x_batch,
                                                  self.labels: y_batch,
                                                  self.step: step-1})

              # Display training information
              if step % self.cfg.DISPLAY_STEP == 0:
                self._display_status(sess, x_batch, y_batch, epoch_i, step)

              # Save training logs
              if self.cfg.SAVE_LOG_STEP is not None:
                if step % self.cfg.SAVE_LOG_STEP == 0:
                  self._save_logs(sess, train_writer, valid_writer,
                                  x_batch, y_batch, epoch_i, step)

              # Save reconstruction images
              if self.cfg.SAVE_IMAGE_STEP is not None:
                if self.cfg.WITH_RECONSTRUCTION:
                  if step % self.cfg.SAVE_IMAGE_STEP == 0:
                    self._save_images(sess, self.train_image_path, x_batch,
                                      y_batch, step, epoch_i=epoch_i)

              # Save model
              if self.cfg.SAVE_MODEL_MODE == 'per_batch':
                if step % self.cfg.SAVE_MODEL_STEP == 0:
                  self._save_model(sess, self.saver, step)

              # Evaluate on full set
              if self.cfg.FULL_SET_EVAL_MODE == 'per_batch':
                if step % self.cfg.FULL_SET_EVAL_STEP == 0:
                  self._eval_on_full_set(sess, epoch_i, step)
                  utils.thick_line()
          else:
            utils.thin_line()
            train_batch_generator = utils.get_batches(
                self.x_train, self.y_train, self.cfg.BATCH_SIZE)
            for _ in tqdm(range(self.n_batch_train),
                          total=self.n_batch_train,
                          ncols=100, unit=' batches'):

              step += 1
              x_batch, y_batch = next(train_batch_generator)

              # Training optimizer
              sess.run(self.optimizer, feed_dict={self.inputs: x_batch,
                                                  self.labels: y_batch,
                                                  self.step: step-1})

              # Save training logs
              if self.cfg.SAVE_LOG_STEP is not None:
                if step % self.cfg.SAVE_LOG_STEP == 0:
                  self._save_logs(sess, train_writer, valid_writer,
                                  x_batch, y_batch, epoch_i, step)

              # Save reconstruction images
              if self.cfg.SAVE_IMAGE_STEP is not None:
                if self.cfg.WITH_RECONSTRUCTION:
                  if step % self.cfg.SAVE_IMAGE_STEP == 0:
                    self._save_images(sess, self.train_image_path,
                                      x_batch, y_batch, step,
                                      silent=True, epoch_i=epoch_i)

              # Save model
              if self.cfg.SAVE_MODEL_MODE == 'per_batch':
                if step % self.cfg.SAVE_MODEL_STEP == 0:
                  self._save_model(sess, self.saver, step, silent=True)

              # Evaluate on full set
              if self.cfg.FULL_SET_EVAL_MODE == 'per_batch':
                if step % self.cfg.FULL_SET_EVAL_STEP == 0:
                  self._eval_on_full_set(sess, epoch_i, step, silent=True)

          if self.cfg.SAVE_MODEL_MODE == 'per_epoch':
            if (epoch_i + 1) % self.cfg.SAVE_MODEL_STEP == 0:
              self._save_model(sess, self.saver, epoch_i)
          if self.cfg.FULL_SET_EVAL_MODE == 'per_epoch':
            if (epoch_i + 1) % self.cfg.FULL_SET_EVAL_MODE == 0:
              self._eval_on_full_set(sess, epoch_i, step)

          utils.thin_line()
          print('Epoch done! Using time: {:.2f}'
                .format(time.time() - epoch_start_time))

        utils.thick_line()
        print('Training finished! Using time: {:.2f}'
              .format(time.time() - self.start_time))
        utils.thick_line()

        # Evaluate on test set after training
        if self.cfg.TEST_AFTER_TRAINING:
          self._test_after_training(sess)

        utils.thick_line()
        print('All task finished! Total time: {:.2f}'
              .format(time.time() - self.start_time))
        utils.thick_line()


if __name__ == '__main__':

  opts, args = getopt.getopt(sys.argv[1:], "g", ['gpu-id'])
  for op, value in opts:
    if op == "-g":
      os.environ["CUDA_VISIBLE_DEVICES"] = str(value)

  CapsNet_ = CapsNetDistribute(config)
  Main_ = Main(CapsNet_, config)
  Main_.train()