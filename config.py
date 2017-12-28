from easydict import EasyDict

__C = EasyDict()

# Learning rate
__C.LEARNING_RATE = 0.001

# Epochs
__C.EPOCHS = 50

# Batch size
__C.BATCH_SIZE = 128

# Display step
__C.DISPLAY_STEP = 1

# Architecture parameters of convolution layers
# [{'kernel_size': None, 'stride': None, 'depth': None}, ]
__C.CONV_PARAMS = [{'kernel_size': 9, 'stride': 1, 'depth': 256, 'padding': 'VALID'},
                   # {'kernel_size': 3, 'stride': 1, 'depth': 128, 'padding': 'VALID'}
                   ]

# Architecture parameters of conv2capsule layer
# {'kernel_size': None, 'stride': None, 'depth': None, 'vec_dim': None, 'padding': 'VALID'}
__C.CONV2CAPS_PARAMS = {'kernel_size': 9, 'stride': 2, 'depth': 32, 'vec_dim': 8, 'padding': 'VALID'}

# Architecture parameters of capsule layers
# [{'num_caps': None, 'vec_dim': None, 'route_epoch': None}, ]
__C.CAPS_PARAMS = [{'num_caps': 10, 'vec_dim': 16, 'route_epoch': 3},
                   # {'num_caps': 10, 'vec_dim': 32, 'route_epoch': 3}
                   ]

# Training with reconstruction
__C.WITH_RECONSTRUCTION = True

# Type of decoder of reconstruction:
# 'FC': full_connected layers
# 'CONV': convolution layers
# 'CONV_T': transpose convolution layers
__C.DECODER_TYPE = 'FC'

# Architecture parameters of decoders of reconstruction
# 'FC': [{'num_outputs':None, 'act_fn': None}, ]  # 'act_fn': 'relu', 'sigmoid'
# 'CONV': [{'kernel_size': None, 'stride': None, 'depth': None, 'padding': 'VALID', 'resize': None}, ]
# 'CONV_T': [{'kernel_size': None, 'stride': None, 'depth': None, 'padding': 'VALID'}, ]
__C.DECODER_PARAMS = [{'num_outputs': 512, 'act_fn': 'relu'},
                      {'num_outputs': 1024, 'act_fn': 'relu'},
                      {'num_outputs': 784, 'act_fn': 'sigmoid'}]

# Parameter of margin loss
# default: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
__C.MARGIN_LOSS_PARAMS = {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}

# Scaling for reconstruction loss
__C.RECONSTRUCT_COST_SCALE = 0.392  # 0.0005*784=0.392

# Add epsilon(a very small number) to zeros
__C.EPSILON = 1e-9

# stddev of tf.truncated_normal_initializer()
__C.STDDEV = 0.01

# Database name
__C.DATABASE_NAME = 'mnist'

# Data directory path
__C.SOURCE_DATA_PATH = './data/source_data/'

# Log directory path
__C.LOG_PATH = './logs/'

# get config by: from config import cfg
cfg = __C
