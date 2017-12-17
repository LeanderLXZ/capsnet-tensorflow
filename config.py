import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('conv1_k_size', 128, 'conv1_k_size')
flags.DEFINE_integer('conv1_stride', 128, 'conv1_stride')
flags.DEFINE_integer('conv1_depth', 128, 'conv1_depth')

cfg = tf.app.flags.FLAGS
