from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.model_base import *
from models.capsule_layer import *


def classifier(inputs, cfg, batch_size=None):

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
  model.add(Conv2CapsLayer(
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
  # models.add(Dense2Capsule(
  #     cfg,
  #     identity_map=True,
  #     num_caps=None,
  #     act_fn='relu',
  #     vec_dim=8,
  #     batch_size=batch_size
  # ))
  model.add(CapsLayer(
      cfg,
      num_caps=10,
      vec_dim=16,
      route_epoch=3,
      batch_size=batch_size,
      idx=0
  ))

  return model.top_layer, model.info


def decoder(inputs, cfg, batch_size=None):

  model = Sequential(inputs)
  act_fn_last = None if cfg.RECONSTRUCTION_LOSS == 'ce' else 'relu'

  if cfg.DECODER_TYPE == 'fc':
    model.add(DenseLayer(
        cfg,
        out_dim=512,
        idx=0))
    model.add(DenseLayer(
        cfg,
        out_dim=1024,
        idx=1))
    model.add(DenseLayer(
        cfg,
        out_dim=784,
        act_fn=act_fn_last,
        idx=2))

  elif cfg.DECODER_TYPE == 'conv':
    model.add(Reshape(
        (batch_size, 4, 4, -1), name='reshape'))
    model.add(ConvLayer(
        cfg,
        kernel_size=3,
        stride=1,
        n_kernel=16,
        resize=7,
        idx=0))
    model.add(ConvLayer(
        cfg,
        kernel_size=3,
        stride=1,
        n_kernel=32,
        resize=14,
        idx=1))
    model.add(ConvLayer(
        cfg,
        kernel_size=3,
        stride=1,
        n_kernel=32,
        resize=28,
        idx=2))
    model.add(ConvLayer(
        cfg,
        kernel_size=3,
        stride=1,
        n_kernel=1,
        act_fn=act_fn_last,
        idx=3))

  elif cfg.DECODER_TYPE == 'conv_t':
    model.add(Reshape(
        (batch_size, 1, 1, -1), name='reshape'))
    model.add(ConvTLayer(
        cfg,
        kernel_size=4,
        stride=1,
        n_kernel=16,
        output_shape=[batch_size, 4, 4, 16],
        padding='VALID',
        idx=0))
    model.add(ConvTLayer(
        cfg,
        kernel_size=9,
        stride=1,
        n_kernel=32,
        output_shape=[batch_size, 12, 12, 32],
        padding='VALID',
        idx=1))
    model.add(ConvTLayer(
        cfg,
        kernel_size=9,
        stride=1,
        n_kernel=16,
        output_shape=[batch_size, 20, 20, 16],
        padding='VALID',
        idx=2))
    model.add(ConvTLayer(
        cfg,
        kernel_size=9,
        stride=1,
        n_kernel=8,
        output_shape=[batch_size, 28, 28, 8],
        padding='VALID',
        idx=3))
    model.add(ConvTLayer(
        cfg,
        kernel_size=3,
        stride=1,
        n_kernel=1,
        output_shape=[batch_size, 28, 28, 1],
        act_fn=act_fn_last,
        idx=4))

  else:
    raise ValueError('Wrong decoder type!')

  return model.top_layer, model.info
