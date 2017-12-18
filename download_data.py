import os
import utils
from config import cfg


# Download database
def download_data():

    database_mnist_name = 'mnist'

    if cfg.DATABASE_NAME == database_mnist_name:

        utils.download_and_extract(url='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                                   data_path=cfg.DATA_PATH,
                                   save_path=os.path.join(cfg.DATA_PATH, 'train-images-idx3-ubyte.gz'),
                                   extract_path=os.path.join(cfg.DATA_PATH, 'mnist_train_images'),
                                   database_name=cfg.DATABASE_NAME,
                                   extract_fn=utils.ungzip_image)

        utils.download_and_extract(url='http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                                   data_path=cfg.DATA_PATH,
                                   save_path=os.path.join(cfg.DATA_PATH, 't10k-images-idx3-ubyte.gz'),
                                   extract_path=os.path.join(cfg.DATA_PATH, 'mnist_test_images'),
                                   database_name=cfg.DATABASE_NAME,
                                   extract_fn=utils.ungzip_image)

        # utils.download_and_extract(url='http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        #                            data_path=cfg.DATA_PATH,
        #                            save_path=os.path.join(cfg.DATA_PATH, 'train-labels-idx1-ubyte.gz'),
        #                            extract_path=os.path.join(cfg.DATA_PATH, 'mnist_train_label'),
        #                            database_name=cfg.DATABASE_NAME)


if __name__ == '__main__':

    download_data()
