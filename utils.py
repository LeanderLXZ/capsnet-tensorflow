import os
import csv
import time
import gzip
import shutil
import pickle
import numpy as np
import tensorflow as tf
from os.path import isdir
from config import cfg
from tqdm import tqdm
from urllib.request import urlretrieve


def save_data_to_pickle(data, data_path):
    """
    Save data to pickle file.
    """
    with open(data_path, 'wb') as f:
        print('Saving {}...'.format(f.name))
        pickle.dump(data, f)


def load_data_from_pickle(data_path):
    """
    Load data from pickle file.
    """

    with open(data_path, 'rb') as f:
        print('Loading {}...'.format(f.name))
        return pickle.load(f)


def get_vec_length(vec):
    """
    Get the length of a vector.
    """
    vec_shape = vec.get_shape().as_list()
    num_caps = vec_shape[1]
    vec_dim = vec_shape[2]

    # vec shape: (batch_size, num_caps, vec_dim)
    assert vec.get_shape() == (cfg.BATCH_SIZE, num_caps, vec_dim), \
        'Wrong shape of vec: {}'.format(vec.get_shape().as_list())

    vec_length = tf.reduce_sum(tf.square(vec), axis=2, keep_dims=True) + cfg.EPSILON
    vec_length = tf.sqrt(tf.squeeze(vec_length))
    # vec_length shape: (batch_size, num_caps)
    assert vec_length.get_shape() == (cfg.BATCH_SIZE, num_caps), \
        'Wrong shape of vec_length: {}'.format(vec_length.get_shape().as_list())

    return vec_length


def check_dir(path_list):
    """
    Check if directories exit or not.
    """
    for dir_path in path_list:
        if not isdir(dir_path):
            os.makedirs(dir_path)


def _read32(bytestream):
    """
    Read 32-bit integer from bytesteam
    :param bytestream: A bytestream
    :return: 32-bit integer
    """
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _dense_to_one_hot(labels_dense, num_classes):
    """
    Convert class labels from scalars to one-hot vectors.
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_image(save_path, extract_path):
    """
    Extract the images into a 4D unit8 numpy array [index, y, x, depth].
    """
    # Get data from save_path
    with open(save_path, 'rb') as f:

        print('Extracting {}...'.format(f.name))

        with gzip.GzipFile(fileobj=f) as bytestream:

            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number {} in file: {}'.format(magic, f.name))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)
            save_data_to_pickle(data, extract_path + '.p')


def extract_labels(save_path, extract_path, one_hot=False, num_classes=10):
    """
    Extract the labels into a 1D uint8 numpy array [index].
    """
    # Get data from save_path
    with open(save_path, 'rb') as f:

        print('Extracting {}...'.format(f.name))

        with gzip.GzipFile(fileobj=f) as bytestream:

            magic = _read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                                 (magic, f.name))
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            if one_hot:
                labels = _dense_to_one_hot(labels, num_classes)
            save_data_to_pickle(labels, extract_path + '.p')


def download_and_extract_mnist(url, save_path, extract_path, database_name, data_type):

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1,
                        desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(url, save_path, pbar.hook)

    try:
        if data_type == 'image':
            extract_image(save_path, extract_path)
        elif data_type == 'label':
            extract_labels(save_path, extract_path, one_hot=True, num_classes=10)
        else:
            raise ValueError('Wrong data_type!')
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)


def thin_line():
    print('------------------------------------------------------')


def thick_line():
    print('======================================================')


def print_status(epoch_i, batch_counter, start_time, cost_train,
                 cost_rec_train, acc_train, cost_valid, cost_rec_valid, acc_valid):

    if cfg.WITH_RECONSTRUCTION:
        print('Epoch: {}/{} |'.format(epoch_i + 1, cfg.EPOCHS),
              'Batch: {} |'.format(batch_counter),
              'Time: {:.2f}s |'.format(time.time() - start_time),
              'Train_Loss: {:.4f} |'.format(cost_train),
              'Reconstruction_Train_Loss: {:.4f} |'.format(cost_rec_train),
              'Train_Accuracy: {:.2f}% |'.format(acc_train * 100),
              'Valid_Loss: {:.4f} |'.format(cost_valid),
              'Reconstruction_Valid_Loss: {:.4f} |'.format(cost_rec_valid),
              'Valid_Accuracy: {:.2f}% |'.format(acc_valid * 100))
    else:
        print('Epoch: {}/{} |'.format(epoch_i + 1, cfg.EPOCHS),
              'Batch: {} |'.format(batch_counter),
              'Time: {:.2f}s |'.format(time.time() - start_time),
              'Train_Loss: {:.4f} |'.format(cost_train),
              'Train_Accuracy: {:.2f}% |'.format(acc_train * 100),
              'Valid_Loss: {:.4f} |'.format(cost_valid),
              'Valid_Accuracy: {:.2f}% |'.format(acc_valid * 100))


def save_config_log(file_path):
    """
    Save config of training.
    """
    file_path = os.path.join(file_path, 'config_log.txt')
    thick_line()
    print('Saving {}...'.format(file_path))

    with open(file_path, 'a') as f:
        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
        f.write('=====================================================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------\n')
        for key in cfg.keys():
            f.write('{}: {}\n'.format(key, cfg[key]))
        f.write('=====================================================')


def save_log(file_path, epoch_i, batch_counter, using_time, cost_train,
             cost_rec_train, acc_train, cost_valid, cost_rec_valid, acc_valid):
    """
    Save losses and accuracies while training.
    """
    if cfg.WITH_RECONSTRUCTION:
        if not os.path.isfile(file_path):
            with open(file_path, 'w') as f:
                header = ['Local_Time', 'Epoch', 'Batch', 'Time', 'Train_Loss', 'Reconstruction_Train_Loss',
                          'Train_Accuracy', 'Valid_Loss', 'Reconstruction_Valid_Loss', 'Valid_Accuracy']
                writer = csv.writer(f)
                writer.writerow(header)

        with open(file_path, 'a') as f:
            local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
            log = [local_time, epoch_i, batch_counter, using_time, cost_train,
                   cost_rec_train, acc_train, cost_valid,  cost_rec_valid, acc_valid]
            writer = csv.writer(f)
            writer.writerow(log)
    else:
        if not os.path.isfile(file_path):
            with open(file_path, 'w') as f:
                header = ['Local_Time', 'Epoch', 'Batch', 'Time', 'Train_Loss',
                          'Train_Accuracy', 'Valid_Loss', 'Valid_Accuracy']
                writer = csv.writer(f)
                writer.writerow(header)

        with open(file_path, 'a') as f:
            local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
            log = [local_time, epoch_i, batch_counter, using_time,
                   cost_train, acc_train, cost_valid, acc_valid]
            writer = csv.writer(f)
            writer.writerow(log)


def save_test_log(file_path, cost_test, acc_test):
    """
    Save losses and accuracies of testing.
    """
    file_path = os.path.join(file_path, 'test_log.txt')
    thick_line()
    print('Saving {}...'.format(file_path))

    with open(file_path, 'a') as f:
        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
        f.write('=====================================================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------\n')
        f.write('Test_Loss: {:.4f}\n'.format(cost_test))
        f.write('Test_Accuracy: {:.2f}%'.format(acc_test * 100))
        f.write('=====================================================')


class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
