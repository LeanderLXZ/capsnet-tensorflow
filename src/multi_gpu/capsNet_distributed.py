from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model.capsNet import CapsNet


class CapsNetDistributed(CapsNet):

    def __init__(self, cfg):

        super(CapsNet, self).__init__(cfg)
        self.cfg = cfg

    def tower_loss(self):