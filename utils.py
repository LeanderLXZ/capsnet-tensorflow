import os
import gzip
import hashlib
import shutil
import pickle
import numpy as np
import tensorflow as tf
from os.path import isdir
from config import cfg
from tqdm import tqdm
from urllib.request import urlretrieve


# Get the length of a vector
def get_vec_length(vec):

    # vec shape: (batch_size, num_caps, vec_dim)
    vec_length = tf.sqrt(tf.reduce_sum(tf.square(vec), axis=2, keep_dims=True) + cfg.EPSILON)

    # vec_length shape: (batch_size, num_caps)
    return vec_length


# Check if directories exit or not
def check_dir(path_list):

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
    Extract the images into a 4D uint8 numpy array [index, y, x, depth].
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

            with open(extract_path + '.p', 'wb') as f_p:
                pickle.dump(data, f_p)


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

            with open(extract_path + '.p', 'wb') as f_p:
                pickle.dump(labels, f_p)


def download_and_extract_mnist(url, data_path, save_path, extract_path, database_name, data_type):

    if not os.path.exists(data_path):
        os.makedirs(data_path)

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
