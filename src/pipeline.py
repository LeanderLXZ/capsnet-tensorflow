from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config_pipeline import *
from main import Main
from model.capsNet import CapsNet


def training_capsnet(cfg):

  CapsNet_ = CapsNet(cfg)
  Main_ = Main(CapsNet_, cfg)
  Main_.train()


def pipeline():

  # training_capsnet(cfg_1)
  # training_capsnet(cfg_2)
  training_capsnet(cfg_3)
  training_capsnet(cfg_4)
  training_capsnet(cfg_5)
  training_capsnet(cfg_6)
  training_capsnet(cfg_7)


if __name__ == '__main__':

  pipeline()
