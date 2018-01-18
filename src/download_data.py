from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join

from config import config as cfg
from model import utils


# Download database
def download_data():
  """
  Download database.
  """

  if cfg.DATABASE_NAME == 'mnist':

    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    utils.check_dir(['./data/', cfg.SOURCE_DATA_PATH])

    utils.download_and_extract_mnist(
        url=SOURCE_URL + TRAIN_IMAGES,
        save_path=join(cfg.SOURCE_DATA_PATH, TRAIN_IMAGES),
        extract_path=join(cfg.SOURCE_DATA_PATH, 'train_image'),
        database_name=cfg.DATABASE_NAME,
        data_type='image')

    utils.download_and_extract_mnist(
        url=SOURCE_URL + TRAIN_LABELS,
        save_path=join(cfg.SOURCE_DATA_PATH, TRAIN_LABELS),
        extract_path=join(cfg.SOURCE_DATA_PATH, 'train_label'),
        database_name=cfg.DATABASE_NAME,
        data_type='label')

    utils.download_and_extract_mnist(
        url=SOURCE_URL + TEST_IMAGES,
        save_path=join(cfg.SOURCE_DATA_PATH, TEST_IMAGES),
        extract_path=join(cfg.SOURCE_DATA_PATH, 'test_image'),
        database_name=cfg.DATABASE_NAME,
        data_type='image')

    utils.download_and_extract_mnist(
        url=SOURCE_URL + TEST_LABELS,
        save_path=join(cfg.SOURCE_DATA_PATH, TEST_LABELS),
        extract_path=join(cfg.SOURCE_DATA_PATH, 'test_label'),
        database_name=cfg.DATABASE_NAME,
        data_type='label')

  else:
    raise ValueError('Wrong database name!')


if __name__ == '__main__':

  download_data()
