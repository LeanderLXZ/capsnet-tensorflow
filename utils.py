import os
import tensorflow as tf
from os.path import isdir


# Get the length of a vector
def get_vec_length(vec, epsilon=1e-9):

    # vec shape: (batch_size, num_caps, vec_dim)
    vec_length = tf.sqrt(tf.reduce_sum(tf.square(vec), axis=2, keep_dims=True) + epsilon)

    # vec_length shape: (batch_size, num_caps)
    return vec_length


# Check if directories exit or not
def check_dir(path_list):

    for dir_path in path_list:
        if not isdir(dir_path):
            os.makedirs(dir_path)
