from easydict import EasyDict

__C = EasyDict()

# Database name
__C.DATABASE_NAME = 'mnist'

# Learning rate
__C.LEARNING_RATE = 0.001

# Epochs
__C.EPOCHS = 1000

# Batch size
__C.BATCH_SIZE = 128

# Display step
__C.DISPLAY_STEP = 100

# Architecture parameters of convolution layers
# [{'kernel_size': None, 'stride': None, 'depth': None}, ]
__C.CONV_PARAMS = [{'kernel_size': None, 'stride': None, 'depth': None}, ]

# Architecture parameters of conv2capsule layer
# {'kernel_size': None, 'stride': None, 'depth': None, 'vec_dim': None}
__C.CONV2CAPS_PARAMS = {'kernel_size': None, 'stride': None, 'depth': None, 'vec_dim': None}

# Architecture parameters of capsule layers
# [{'num_caps': None, 'vec_dim': None, 'route_epoch': None}, ]
__C.CAPS_PARAMS = [{'num_caps': None, 'vec_dim': None, 'route_epoch': None}, ]

# Parameter of margin loss
# default: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
__C.MARGIN_LOSS_PARAMS = {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}

# Add epsilon(a very small number) to zeros
__C.EPSILON = 1e-9

# Data directory path
__C.DATA_PATH = './data/'

# Log directory path
__C.LOG_PATH = './logs/'

# get config by: from config import cfg
cfg = __C
