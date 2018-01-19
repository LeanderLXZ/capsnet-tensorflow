from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from os.path import join
from sklearn.preprocessing import LabelBinarizer

from models import utils
from config import config


class DataPreProcess(object):

  def __init__(self, cfg):
    """
    Preprocess data and save as pickle files.

    Args:
      cfg: configuration
    """
    self.cfg = cfg
    self.preprocessed_path = join(cfg.DPP_DATA_PATH, cfg.DATABASE_NAME)
    self.source_data_path = join(cfg.SOURCE_DATA_PATH, cfg.DATABASE_NAME)

  def _load_data(self):
    """
    Load data set from files.
    """
    utils.thin_line()
    print('Loading...')

    if self.cfg.DATABASE_NAME == 'mnist':
      self.x = utils.load_data_from_pkl(
          join(self.source_data_path, 'train_image.p'))
      self.y = utils.load_data_from_pkl(
          join(self.source_data_path, 'train_label.p'))
      self.x_test = utils.load_data_from_pkl(
          join(self.source_data_path, 'test_image.p'))
      self.y_test = utils.load_data_from_pkl(
          join(self.source_data_path, 'test_label.p'))
    else:
      raise ValueError('Wrong database name!')

  def _augment_data(self):
    """
    Augment data set and add noises.
    """
    pass

  def _scaling(self):
    """
    Scaling input images to (0, 1).
    """
    utils.thin_line()
    print('Scaling...')
    
    self.x = np.divide(self.x, 255.)
    self.x_test = np.divide(self.x_test, 255.)

  def _one_hot_encoding(self):
    """
    Scaling images to (0, 1).
    """
    utils.thin_line()
    print('One-hot-encoding...')
    
    encoder = LabelBinarizer()
    encoder.fit(self.y)
    self.y = encoder.transform(self.y)
    self.y_test = encoder.transform(self.y_test)

  def _split_data(self):
    """
    Split data set for training, validation and testing.
    """
    utils.thin_line()
    print('Splitting...')
    
    if self.cfg.DATABASE_NAME == 'mnist':
      self.x = self.x.reshape([-1, 28, 28, 1])
      self.x_test = self.x_test.reshape([-1, 28, 28, 1])
      if self.cfg.DPP_TEST_AS_VALID:
        self.x_train = self.x
        self.y_train = self.y
        self.x_valid = self.x_test
        self.y_valid = self.y_test
      else:
        self.x_train = self.x[:55000]
        self.x_valid = self.x[55000:60000]
        self.y_train = self.y[:55000]
        self.y_valid = self.y[55000:60000]
    else:
      raise ValueError('Wrong database name!')

    assert self.x_train.shape == (55000, 28, 28, 1), self.x_train.shape
    assert self.y_train.shape == (55000, 10), self.y_train.shape
    assert self.x_valid.shape == (5000, 28, 28, 1), self.x_valid.shape
    assert self.y_valid.shape == (5000, 10), self.y_valid.shape
    assert self.x_test.shape == (10000, 28, 28, 1), self.x_test.shape
    assert self.y_test.shape == (10000, 10), self.y_test.shape

  def _save_data(self):
    """
    Save data set to pickle files.
    """
    utils.thin_line()
    print('Saving...')

    utils.check_dir([self.preprocessed_path])
    
    utils.save_data_to_pkl(
        self.x_train, join(self.preprocessed_path, 'x_train.p'))
    utils.save_data_to_pkl(
        self.y_train, join(self.preprocessed_path, 'y_train.p'))
    utils.save_data_to_pkl(
        self.x_valid, join(self.preprocessed_path, 'x_valid.p'))
    utils.save_data_to_pkl(
        self.y_valid, join(self.preprocessed_path, 'y_valid.p'))
    utils.save_data_to_pkl(
        self.x_test, join(self.preprocessed_path, 'x_test.p'))
    utils.save_data_to_pkl(
        self.y_test, join(self.preprocessed_path, 'y_test.p'))

  def pipeline(self):
    """
    Pipeline.
    """
    utils.thick_line()
    print('Start Preprocessing...')

    start_time = time.time()

    # Load data
    self._load_data()

    # Augment data
    self._augment_data()

    # Scaling images to (0, 1)
    self._scaling()

    # One-hot-encoding labels
    self._one_hot_encoding()

    # Split data set into train/valid/test
    self._split_data()

    # Save data to pickles
    self._save_data()

    utils.thin_line()
    print('Done! Using {:.3}s'.format(time.time() - start_time))
    utils.thick_line()


if __name__ == '__main__':

  DPP = DataPreProcess(config)
  DPP.pipeline()
