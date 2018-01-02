import os
from easydict import EasyDict

__C = EasyDict()

# ===========================================
# #             Hyperparameters             #
# ===========================================

# Database name
__C.VERSION = 'without_reconstruction'

# Learning rate
__C.LEARNING_RATE = 0.001

# Epochs
__C.EPOCHS = 50

# Batch size
__C.BATCH_SIZE = 128

# ===========================================
# #            Model Architecture           #
# ===========================================

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

# Parameters of margin loss
# default: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
__C.MARGIN_LOSS_PARAMS = {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}

# Add epsilon(a very small number) to zeros
__C.EPSILON = 1e-9

# stddev of tf.truncated_normal_initializer()
__C.STDDEV = 0.01

# -------------------------------------------
# Reconstruction

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

# Scaling for reconstruction loss
__C.RECONSTRUCT_COST_SCALE = 0.392  # 0.0005*784=0.392

# -------------------------------------------
# Test

# Evaluate on test set after training
__C.TEST_AFTER_TRAINING = True

# ===========================================
# #                 Display                 #
# ===========================================

# Display step
# Set None to not display
__C.DISPLAY_STEP = 1  # per batch

# Save summary step
# Set None to not save summaries
__C.SAVE_LOG_STEP = 20  # per batch

# Save reconstructed images
# Set None to not save images
__C.SAVE_IMAGE_STEP = 1

# Calculate train loss and valid loss using full data set
# None: not display
# 'per_epoch': evaluate when every epoch finished
__C.FULL_SET_EVAL_STEP = 50  # per batch

# Calculate the train loss of full data set, which may take lots of time.
__C.EVAL_WITH_FULL_TRAIN_SET = False

# Show details of training progress
__C.SHOW_TRAINING_DETAILS = False

# ===========================================
# #                  Others                 #
# ===========================================

# Database name
__C.DATABASE_NAME = 'mnist'

# Source data directory path
__C.SOURCE_DATA_PATH = './data/source_data'

# Path for saving logs
__C.LOG_PATH = os.path.join('./logs', __C.VERSION)

# Path for saving summaries
__C.SUMMARY_PATH = os.path.join('./logs/summaries', __C.VERSION)

# get config by: from config import cfg
cfg = __C
