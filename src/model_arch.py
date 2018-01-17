from model.model_base import Sequential
from model.model_base import ConvLayer
from model.capsule_layer import *


def get_model(inputs, cfg, batch_size=None):

  model = Sequential(inputs)
  model.add(ConvLayer(
      cfg,
      kernel_size=9,
      stride=1,
      n_kernel=256,
      padding='VALID',
      act_fn='relu',
      stddev=None,
      resize=None,
      use_bias=True,
      atrous=False,
      idx=0
  ))
  model.add(Conv2Capsule(
      cfg,
      kernel_size=9,
      stride=2,
      n_kernel=32,
      vec_dim=8,
      padding='VALID',
      act_fn='relu',
      use_bias=True,
      batch_size=batch_size
  ))
  # model.add(Dense2Capsule(
  #     cfg,
  #     identity_map=True,
  #     num_caps=None,
  #     act_fn='relu',
  #     vec_dim=8,
  #     batch_size=batch_size
  # ))
  model.add(CapsuleLayer(
      cfg,
      num_caps=10,
      vec_dim=16,
      route_epoch=3,
      batch_size=batch_size,
      idx=0
  ))

  return model.top_layer
