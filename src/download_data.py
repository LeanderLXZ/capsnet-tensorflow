from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join

from config import config as cfg
from models import utils


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

    source_data_path_ = join(cfg.SOURCE_DATA_PATH, cfg.DATABASE_NAME)
    utils.check_dir([source_data_path_])

    utils.download_and_extract_mnist(
        url=SOURCE_URL + TRAIN_IMAGES,
        save_path=join(source_data_path_, TRAIN_IMAGES),
        extract_path=join(source_data_path_, 'train_image'),
        database_name=cfg.DATABASE_NAME,
        data_type='image')

    utils.download_and_extract_mnist(
        url=SOURCE_URL + TRAIN_LABELS,
        save_path=join(source_data_path_, TRAIN_LABELS),
        extract_path=join(source_data_path_, 'train_label'),
        database_name=cfg.DATABASE_NAME,
        data_type='label')

    utils.download_and_extract_mnist(
        url=SOURCE_URL + TEST_IMAGES,
        save_path=join(source_data_path_, TEST_IMAGES),
        extract_path=join(source_data_path_, 'test_image'),
        database_name=cfg.DATABASE_NAME,
        data_type='image')

    utils.download_and_extract_mnist(
        url=SOURCE_URL + TEST_LABELS,
        save_path=join(source_data_path_, TEST_LABELS),
        extract_path=join(source_data_path_, 'test_label'),
        database_name=cfg.DATABASE_NAME,
        data_type='label')

  else:
    raise ValueError('Wrong database name!')


if __name__ == '__main__':

  download_data()
