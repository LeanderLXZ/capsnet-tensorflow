from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from capsNet_arch import classifier
from capsNet_arch import decoder
from models import utils
from models import capsule_layer
from models.model_base import ModelBase


class CapsNet(ModelBase):

  def __init__(self, cfg):
    super(CapsNet, self).__init__(cfg)

    self.batch_size = cfg.BATCH_SIZE

  def _get_inputs(self, image_size, num_class):
    """
    Get input tensors.

    Args:
      image_size: the size of input images, should be 3 dimensional
      num_class: number of class of label
    Returns:
      input tensors
    """
    _inputs = tf.placeholder(
        tf.float32, shape=[self.cfg.BATCH_SIZE, *image_size], name='inputs')
    _labels = tf.placeholder(
        tf.float32, shape=[self.cfg.BATCH_SIZE, num_class], name='labels')

    return _inputs, _labels

  def _caps_layer(self, x, caps_param, idx=0):
    """
    Single capsule layer

    Args:
      x: input tensor
      caps_param: parameters of capsule layer
    Returns:
      output tensor of capsule layer
    """
    _caps = capsule_layer.CapsuleLayer(
        self.cfg, **caps_param, batch_size=self.batch_size, idx=idx)
    return _caps(x)

  def _conv2caps_layer(self, x, conv2caps_params):
    """
    Build convolution to capsule layer.

    Args:
      x: input tensor
      conv2caps_params: parameters of conv2caps layer
    Returns:
      output tensor of capsule layer
        - shape: (batch_size, num_caps, vec_dim, 1)
    """
    with tf.variable_scope('conv2caps'):
      # conv2caps_params:
      # {'kernel_size': None, 'stride': None, 'n_kernel': None,
      #  'vec_dim': None, 'padding': 'VALID'}
      _conv2caps = capsule_layer.Conv2Capsule(
          self.cfg, **conv2caps_params, batch_size=self.batch_size)
      return _conv2caps(x)

  def _multi_caps_layers(self, x):
    """
    Build multi-capsule layer.

    Args:
      x: input tensor
    Returns:
      multi capsule layers' output tensor
        - shape: (batch_size, num_caps, vec_dim, 1)
    """
    caps_layers = [x]
    for iter_caps, caps_param in enumerate(self.cfg.CAPS_PARAMS):
      # caps_param:
      #       {'num_caps': None, 'vec_dim': None, 'route_epoch': None}
      caps_layer = self._caps_layer(
          caps_layers[iter_caps], caps_param, idx=iter_caps)
      caps_layers.append(caps_layer)

    return caps_layers[-1]

  def _margin_loss(self, logits, labels, m_plus=0.9,
                   m_minus=0.1, lambda_=0.5):
    """
    Calculate margin loss according to Hinton's paper.
    L = T_c * max(0, m_plus-||v_c||)^2 +
        lambda_ * (1-T_c) * max(0, ||v_c||-m_minus)^2

    Args:
      logits: output tensor of capsule layers.
        - shape: (batch_size, num_caps, vec_dim)
      labels: labels
        - shape: (batch_size, num_caps)
      m_plus: truncation of positive item
      m_minus: truncation of negative item
      lambda_: lambda
    Returns:
      margin loss
    """
    logits_shape = logits.get_shape()
    num_caps = logits_shape[1]

    max_square_plus = tf.square(tf.maximum(
        0., m_plus - utils.get_vec_length(
            logits, self.batch_size, self.cfg.EPSILON)))
    max_square_minus = tf.square(tf.maximum(
        0., utils.get_vec_length(
            logits, self.batch_size, self.cfg.EPSILON) - m_minus))
    # max_square_plus & max_plus shape: (batch_size, num_caps)
    assert max_square_plus.get_shape() == (self.batch_size, num_caps)

    loss_c = tf.multiply(labels, max_square_plus) + \
        lambda_ * tf.multiply((1 - labels), max_square_minus)

    # Total margin loss
    margin_loss = tf.reduce_mean(tf.reduce_sum(loss_c, axis=1))

    return margin_loss

  def _reconstruct_layers(self, inputs, labels):
    """
    Reconstruction layer

    Args:
      inputs: input tensor
        - shape: (batch_size, n_class, vec_dim_j)
      labels: labels
        - shape: (batch_size, n_class)
    Returns:
      output tensor of reconstruction layer
    """
    with tf.variable_scope('masking'):
      # _masked shape: (batch_size, vec_dim_j)
      _masked = tf.reduce_sum(
          tf.multiply(inputs, tf.expand_dims(labels, axis=-1)), axis=1)

    with tf.variable_scope('decoder'):
      # _reconstructed shape: (batch_size, image_size*image_size)
      _reconstructed = decoder(_masked, self.cfg, batch_size=self.batch_size)

    return _reconstructed

  def _loss_without_rec(self, logits, labels):
    """
    Calculate loss without reconstruction.

    Args:
      logits: output tensor of models
        - shape (batch_size, num_caps, vec_dim)
      labels: labels
    Return:
      total loss
    """
    # margin_loss_params: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
    loss = self._margin_loss(logits, labels, **self.cfg.MARGIN_LOSS_PARAMS)

    return loss

  def _loss_with_rec(self, inputs, logits, labels, image_size):
    """
    Calculate loss with reconstruction.

    Args:
      inputs: input tensor
        - shape (batch_size, *image_size)
      logits: output tensor of models
        - shape (batch_size, num_caps, vec_dim)
      labels: labels
      image_size: size of image, 3D
    Return:
      Total loss
    """
    # Reconstruction layers
    # reconstructed shape: (batch_size, image_size*image_size)
    reconstructed = self._reconstruct_layers(logits, labels)
    if self.cfg.SHOW_TRAINING_DETAILS:
      reconstructed = tf.Print(
          reconstructed, [tf.constant(4)],
          message="\nRECONSTRUCTION layers passed...")

    reconstructed_images = tf.reshape(
        reconstructed, shape=[-1, *image_size], name='rec_images')

    # Reconstruction loss
    if self.cfg.RECONSTRUCTION_LOSS == 'mse':
      inputs_flatten = tf.contrib.layers.flatten(inputs)
      if self.cfg.DECODER_TYPE != 'fc':
        reconstructed_ = tf.contrib.layers.flatten(reconstructed)
      else:
        reconstructed_ = reconstructed
      reconstruct_loss = tf.reduce_mean(
          tf.square(reconstructed_ - inputs_flatten))
    elif self.cfg.RECONSTRUCTION_LOSS == 'ce':
      if self.cfg.DECODER_TYPE == 'cf':
        inputs_ = tf.contrib.layers.flatten(inputs)
      else:
        inputs_ = inputs
      reconstruct_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=inputs_, logits=reconstructed))
    else:
      reconstruct_loss = None
    reconstruct_loss = tf.identity(reconstruct_loss, name='rec_loss')

    # margin_loss_params: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
    classifier_loss = self._margin_loss(
        logits, labels, **self.cfg.MARGIN_LOSS_PARAMS)
    classifier_loss = tf.identity(classifier_loss, name='classifier_loss')

    loss = classifier_loss + \
        self.cfg.RECONSTRUCT_LOSS_SCALE * reconstruct_loss

    if self.cfg.SHOW_TRAINING_DETAILS:
      loss = tf.Print(loss, [tf.constant(5)], message="\nloss calculated...")

    return loss, classifier_loss, reconstruct_loss, reconstructed_images

  def _total_loss(self, inputs, logits, labels, image_size):
    """
    Get Losses and reconstructed images tensor.
    """
    if self.cfg.WITH_RECONSTRUCTION:
      loss, classifier_loss, reconstruct_loss, reconstructed_images = \
          self._loss_with_rec(inputs, logits, labels, image_size)
    else:
      loss = self._loss_without_rec(logits, labels)
      classifier_loss, reconstruct_loss, reconstructed_images = \
          None, None, None

    loss = tf.identity(loss, name='loss')

    return loss, classifier_loss, reconstruct_loss, reconstructed_images

  def _inference(self, inputs, labels):
    """
    Build inference graph.

    Args:
      inputs: input tensor
        - shape (batch_size, *image_size)
      labels: labels tensor
    Return:
      logits: output tensor of models
        - shape: (batch_size, num_caps, vec_dim)
    """
    logits = classifier(inputs, self.cfg, self.batch_size)

    # Logits shape: (batch_size, num_caps, vec_dim, 1)
    logits = tf.squeeze(logits, name='logits')
    if self.cfg.SHOW_TRAINING_DETAILS:
      logits = tf.Print(logits, [tf.constant(3)],
                        message="\nCAPSULE layers passed...")

    # Accuracy
    correct_pred = tf.equal(tf.argmax(utils.get_vec_length(
        logits, self.batch_size, self.cfg.EPSILON), axis=1),
        tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(
        correct_pred, tf.float32), name='accuracy')

    return logits, accuracy

  def build_graph(self, image_size=(None, None, None), num_class=None):
    """
    Build the graph of CapsNet.

    Args:
      image_size: size of input images, should be 3 dimensional
      num_class: number of class of label
    Returns:
      tuple of (train_graph, inputs, labels, loss,
                optimizer, accuracy, classifier_loss,
                reconstruct_loss, reconstructed_images)
    """
    tf.reset_default_graph()
    train_graph = tf.Graph()

    with train_graph.as_default():

      # Get input placeholders
      inputs, labels = self._get_inputs(image_size, num_class)

      # Build inference Graph
      logits, accuracy = self._inference(inputs, labels)

      # Build reconstruction part
      loss, classifier_loss, reconstruct_loss, reconstructed_images = \
          self._total_loss(inputs, logits, labels, image_size)

      # Optimizer
      if self.cfg.SHOW_TRAINING_DETAILS:
        loss = tf.Print(loss, [tf.constant(6)],
                        message="\nUpdating gradients...")
      opt = self._optimizer(opt_name=self.cfg.OPTIMIZER)
      train_op = opt.minimize(loss)

      # Create a saver.
      saver = tf.train.Saver(tf.global_variables(),
                             max_to_keep=self.cfg.MAX_TO_KEEP_CKP)

      # Build the summary operation from the last tower summaries.
      tf.summary.scalar('accuracy', accuracy)
      tf.summary.scalar('loss', loss)
      if self.cfg.WITH_RECONSTRUCTION:
        tf.summary.scalar('cls_loss', classifier_loss)
        tf.summary.scalar('rec_loss', reconstruct_loss)
      summary_op = tf.summary.merge_all()

      return train_graph, inputs, labels, train_op, saver, \
          summary_op, loss, accuracy, classifier_loss, \
          reconstruct_loss, reconstructed_images
