from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config_pipeline import *
from main_distribute import MainDistribute
from models.capsNet_distribute import CapsNetDistribute


def training_capsnet(cfg):

  CapsNet_ = CapsNetDistribute(cfg)
  Main_ = MainDistribute(CapsNet_, cfg)
  Main_.train()


def pipeline():

  training_capsnet(cfg_1)
  training_capsnet(cfg_2)
  training_capsnet(cfg_3)
  training_capsnet(cfg_4)
  training_capsnet(cfg_5)
  training_capsnet(cfg_6)
  training_capsnet(cfg_7)


if __name__ == '__main__':

  pipeline()
