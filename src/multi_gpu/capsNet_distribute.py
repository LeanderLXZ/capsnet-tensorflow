from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.capsNet import CapsNet


class CapsNetDistribute(CapsNet):

    def __init__(self, cfg, var_on_cpu):

        super(CapsNet, self).__init__(cfg, var_on_cpu)

        self.cfg = cfg
        self.var_on_cpu = var_on_cpu

    def tower_loss(self, inputs, labels, image_size):
        """
        Calculate the total loss on a single tower running the model.

        Args:
            inputs: inputs 4D tensor
                - shape:  (batch_size, *image_size)
            labels: Labels. 1D tensor of shape [batch_size]
            image_size: size of input images, should be 3 dimensional
        Returns:
            Tuple: (loss, classifier_loss,
                    reconstruct_loss, reconstructed_images)
        """
        # Build inference Graph.
        logits = self._inference(inputs)

        # alculating the loss.
        loss, classifier_loss, reconstruct_loss, reconstructed_images = \
            self._total_loss(inputs, logits, labels, image_size)

        return loss, classifier_loss, reconstruct_loss, reconstructed_images

    @staticmethod
    def average_gradients(tower_grads):
        """
        Calculate the average gradient for each shared variable across all
        towers. This function provides a synchronization point across all
        towers.

        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer
                         list is over individual gradients. The inner list is
                         over the gradient calculation for each tower.
            - [[(grad0_gpu0, var0_gpu0), ..., (gradM_gpu0, varM_gpu0)],
               ...,
               [(grad0_gpuN, var0_gpuN), ..., (gradM_gpuN, varM_gpuN)]]
        Returns:
            List of pairs of (gradient, variable) where the gradient has been
            averaged across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Each grad_and_vars looks like:
            # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for grad, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
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
