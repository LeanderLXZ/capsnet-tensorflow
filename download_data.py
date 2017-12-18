import os
import utils
from config import cfg


# Download database
def download_data():

    database_mnist_name = 'mnist'

    if cfg.DATABASE_NAME == database_mnist_name:

        SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

        utils.check_dir([os.path.join(cfg.DATA_PATH, 'mnist')])

        utils.download_and_extract_mnist(url=SOURCE_URL + TRAIN_IMAGES,
                                         data_path=cfg.DATA_PATH,
                                         save_path=os.path.join(cfg.DATA_PATH, TRAIN_IMAGES),
                                         extract_path=os.path.join(cfg.DATA_PATH, 'mnist/train_image'),
                                         database_name=cfg.DATABASE_NAME,
                                         data_type='image')

        utils.download_and_extract_mnist(url=SOURCE_URL + TRAIN_LABELS,
                                         data_path=cfg.DATA_PATH,
                                         save_path=os.path.join(cfg.DATA_PATH, TRAIN_LABELS),
                                         extract_path=os.path.join(cfg.DATA_PATH, 'mnist/train_label'),
                                         database_name=cfg.DATABASE_NAME,
                                         data_type='label')

        utils.download_and_extract_mnist(url=SOURCE_URL + TEST_IMAGES,
                                         data_path=cfg.DATA_PATH,
                                         save_path=os.path.join(cfg.DATA_PATH, TEST_IMAGES),
                                         extract_path=os.path.join(cfg.DATA_PATH, 'mnist/test_image'),
                                         database_name=cfg.DATABASE_NAME,
                                         data_type='image')

        utils.download_and_extract_mnist(url=SOURCE_URL + TEST_LABELS,
                                         data_path=cfg.DATA_PATH,
                                         save_path=os.path.join(cfg.DATA_PATH, TEST_LABELS),
                                         extract_path=os.path.join(cfg.DATA_PATH, 'mnist/test_label'),
                                         database_name=cfg.DATABASE_NAME,
                                         data_type='label')

    else:
        raise ValueError('Wrong database name!')


if __name__ == '__main__':

    download_data()
