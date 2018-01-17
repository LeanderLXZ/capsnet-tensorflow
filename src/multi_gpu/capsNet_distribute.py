from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.capsNet import CapsNet


class CapsNetDistribute(CapsNet):

  def __init__(self, cfg):
    super(CapsNet, self).__init__(cfg)

    self.cfg = cfg
    self.batch_size = cfg.GPU_BATCH_SIZE

  def _tower_loss(self, inputs, labels, image_size):
    """
    Calculate the total loss on a single tower running the model.

    Args:
      inputs: inputs. 4D tensor
        - shape:  (batch_size, *image_size)
      labels: labels. 1D tensor of shape [batch_size]
      image_size: size of input images, should be 3 dimensional
    Returns:
      Tuple: (loss, classifier_loss,
              reconstruct_loss, reconstructed_images)
    """
    # Build inference Graph.
    logits, accuracy = self._inference(inputs, labels)

    # Calculating the loss.
    loss, classifier_loss, reconstruct_loss, reconstructed_images = \
        self._total_loss(inputs, logits, labels, image_size)

    return loss, accuracy, classifier_loss, \
        reconstruct_loss, reconstructed_images

  @staticmethod
  def _average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    This function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
                   is over individual gradients. The inner list is over the
                   gradient calculation for each tower.
        - shape: [[(grad0_gpu0, var0_gpu0), ..., (gradM_gpu0, varM_gpu0)],
                   ...,
                  [(grad0_gpuN, var0_gpuN), ..., (gradM_gpuN, varM_gpuN)]]
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Each grad_and_vars looks like:
      # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for grad, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        print(grad)
        expanded_grad = tf.expand_dims(grad, 0)
        # Append on a 'tower' dimension which we will average over.
        grads.append(expanded_grad)

      # grads: [[grad0_gpu0], [grad0_gpu1], ..., [grad0_gpuN]]
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # The Variables are redundant because they are shared across towers.
      # So we will just return the first tower's pointer to the Variable.
      v = grad_and_vars[0][1]  # varI_gpu0
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)

    return average_grads

  def _generate_batches(self, n_batches):

    def _split_data(_file_path):
      _inputs_train = None
      return tf.split(
          axis=0,
          num_or_size_splits=n_batches,
          value=_inputs_train,
          name='x_batches_train')

    self.x_batches_train = _split_data(self.x_train_path)
    self.y_batches_train = _split_data(self.y_train_path)
    self.x_batches_valid = _split_data(self.x_valid_path)
    self.y_batches_valid = _split_data(self.y_valid_path)
    self.x_batches_test = _split_data(self.x_test_path)
    self.y_batches_test = _split_data(self.y_test_path)

  def _get_batches(self, eval_mode):

    train_mode = tf.constant('train', tf.string)
    valid_mode = tf.constant('valid', tf.string)
    test_mode = tf.constant('test', tf.string)

    _x_batches = tf.case(
        {tf.equal(eval_mode, train_mode): lambda: self.x_batches_train,
         tf.equal(eval_mode, valid_mode): lambda: self.x_batches_valid,
         tf.equal(eval_mode, test_mode): lambda: self.x_batches_test},
        default=lambda: self.x_batches_train,
        exclusive=True)
    _y_batches = tf.case(
        {tf.equal(eval_mode, train_mode): lambda: self.y_batches_train,
         tf.equal(eval_mode, valid_mode): lambda: self.y_batches_valid,
         tf.equal(eval_mode, test_mode): lambda: self.y_batches_test},
        default=lambda: self.y_batches_train,
        exclusive=True)

    return _x_batches, _y_batches

  def build_graph(self, image_size=(None, None, None),
                  num_class=None, n_train_samples=None):
    """
    Build the graph of CapsNet.

    Args:
      image_size: size of input images, should be 3 dimensional
      num_class: number of class of label
      n_train_samples: number of train samples
    Returns:
      tuple of (train_graph, inputs, labels, loss,
                optimizer, accuracy, classifier_loss,
                reconstruct_loss, reconstructed_images)
    """
    tf.reset_default_graph()
    train_graph = tf.Graph()
    loss, accuracy, classifier_loss, \
        reconstruct_loss, reconstructed_images = \
        None, None, None, None, None

    with train_graph.as_default(), tf.device('/cpu:0'):

      # Get batch
      n_batches = tf.placeholder(tf.int16, name='n_batches')
      eval_mode = tf.placeholder(tf.string, name='eval_mode')
      batch_i = tf.placeholder(tf.int16, name='batch_i')
      self._generate_batches(n_batches)
      x_batches, y_batches = self._get_batches(eval_mode)
      x_batch, y_batch = x_batches[batch_i], y_batches[batch_i]

      # Global step
      global_step = tf.placeholder(tf.int16, name='global_step')

      # Optimizer
      optimizer = self._optimizer(self.cfg.OPTIMIZER,
                                  n_train_samples=n_train_samples,
                                  global_step=global_step)

      # Split data for each tower
      x_splits = tf.split(
          axis=0, num_or_size_splits=self.cfg.GPU_NUMBER, value=inputs)
      y_splits = tf.split(
          axis=0, num_or_size_splits=self.cfg.GPU_NUMBER, value=labels)

      # Calculate the gradients for each model tower.
      tower_grads = []
      for i in range(self.cfg.GPU_NUMBER):
        with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):

              # Dequeues one batch for the GPU
              x_tower, y_tower = x_splits[i], y_splits[i]

              # Calculate the loss for one tower.
              loss, accuracy, classifier_loss, reconstruct_loss, \
                  reconstructed_images = self._tower_loss(
                      x_tower, y_tower, image_size)

              # Calculate the gradients on this tower.
              grads = optimizer.compute_gradients(loss)

              # Keep track of the gradients across all towers.
              tower_grads.append(grads)

      # Calculate the mean of each gradient.
      grads = self._average_gradients(tower_grads)

      # Apply the gradients to adjust the shared variables.
      apply_gradient_op = optimizer.apply_gradients(grads)

      # Track the moving averages of all trainable variables.
      variable_averages = tf.train.ExponentialMovingAverage(
          self.cfg.MOVING_AVERAGE_DECAY)
      variables_averages_op = variable_averages.apply(
          tf.trainable_variables())

      # Group all updates to into a single train op.
      train_op = tf.group(apply_gradient_op, variables_averages_op)

      # Create a saver.
      saver = tf.train.Saver(max_to_keep=self.cfg.MAX_TO_KEEP_CKP)

      # Build the summary operation from the last tower summaries.
      tf.summary.scalar('accuracy', accuracy)
      tf.summary.scalar('loss', loss)
      if self.cfg.WITH_RECONSTRUCTION:
        tf.summary.scalar('cls_loss', classifier_loss)
        tf.summary.scalar('rec_loss', reconstruct_loss)
      summary_op = tf.summary.merge_all()

      return global_step, train_graph, inputs, labels, train_op, \
          saver, summary_op, loss, accuracy, classifier_loss, \
          reconstruct_loss, reconstructed_images
