import os
from easydict import EasyDict

__C = EasyDict()

# ===========================================
# #             Hyperparameters             #
# ===========================================

# Training version
__C.VERSION = 'no_rec'

# Learning rate
__C.LEARNING_RATE = 0.001

# Epochs
__C.EPOCHS = 30

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
__C.WITH_RECONSTRUCTION = False

# Type of decoder of reconstruction:
# 'FC': full_connected layers
# 'CONV': convolution layers
# 'CONV_T': transpose convolution layers
__C.DECODER_TYPE = 'FC'

# Architecture parameters of decoders of reconstruction
# 'FC': [{'num_outputs':None, 'act_fn': None}, ]  # 'act_fn': 'relu', 'sigmoid'
# 'CONV': [{'kernel_size': None, 'stride': None, 'depth': None, 'padding': 'VALID', 'act_fn': None, 'resize': None}, ]
# 'CONV_T': [{'kernel_size': None, 'stride': None, 'depth': None, 'padding': 'VALID'}, ]
__C.DECODER_PARAMS = [{'num_outputs': 512, 'act_fn': 'relu'},
                      {'num_outputs': 1024, 'act_fn': 'relu'},
                      {'num_outputs': 784, 'act_fn': 'sigmoid'}]

# Reconstruction loss
# 'mse': Mean Square Error
# 'cross_entropy' : sigmoid_cross_entropy_with_logits
__C.RECONSTRUCTION_LOSS = 'mse'

# Scaling for reconstruction loss
__C.RECONSTRUCT_COST_SCALE = 0.392  # 0.0005*784=0.392

# __C.DECODER_TYPE = 'CONV'
# __C.RECONSTRUCTION_LOSS = 'cross_entropy'
# __C.CONV_RESHAPE_SIZE = (4, 4)
# __C.DECODER_PARAMS = [{'kernel_size': 3, 'stride': 1, 'depth': 16, 'padding': 'SAME', 'act_fn': 'relu', 'resize': 7},
#                       {'kernel_size': 3, 'stride': 1, 'depth': 32, 'padding': 'SAME', 'act_fn': 'relu', 'resize': 14},
#                       {'kernel_size': 3, 'stride': 1, 'depth': 32, 'padding': 'SAME', 'act_fn': 'relu', 'resize': 28},
#                       {'kernel_size': 3, 'stride': 1, 'depth': 1, 'padding': 'SAME', 'act_fn': None}]

# __C.DECODER_TYPE = 'CONV_T'
# __C.RECONSTRUCTION_LOSS = 'mse'
# __C.CONV_RESHAPE_SIZE = (4, 4)
# __C.DECODER_PARAMS = [{'kernel_size': 9, 'stride': 1, 'depth': 16, 'padding': 'VALID', 'act_fn': 'relu'},  # 12x12
#                       {'kernel_size': 9, 'stride': 1, 'depth': 32, 'padding': 'VALID', 'act_fn': 'relu'},  # 20x20
#                       {'kernel_size': 9, 'stride': 1, 'depth': 16, 'padding': 'VALID', 'act_fn': 'relu'},   # 28x28
#                       {'kernel_size': 3, 'stride': 1, 'depth': 1, 'padding': 'SAME', 'act_fn': 'sigmoid'}]

# -------------------------------------------
# Test

# Evaluate on test set after training
__C.TEST_AFTER_TRAINING = True

# ===========================================
# #             Training Config             #
# ===========================================

# Display step
# Set None to not display
__C.DISPLAY_STEP = 10  # batches

# Save summary step
# Set None to not save summaries
__C.SAVE_LOG_STEP = 20  # batches

# Save reconstructed images
# Set None to not save images
__C.SAVE_IMAGE_STEP = 50  # batches

# Calculate train loss and valid loss using full data set
# 'per_epoch': evaluate on full set when n epochs finished
# 'per_batch': evaluate on full set when n batches finished
__C.FULL_SET_EVAL_MODE = 'per_batch'
# None: not evaluate
__C.FULL_SET_EVAL_STEP = 50

# Save model
# 'per_epoch': save model when n epochs finished
# 'per_batch': save model when n batches finished
__C.SAVE_MODEL_MODE = 'per_epoch'
# None: not save model
__C.SAVE_MODEL_STEP = 5
# Maximum number of recent checkpoints to keep.
__C.MAX_TO_KEEP_CKP = 5

# Calculate the train loss of full data set, which may take lots of time.
__C.EVAL_WITH_FULL_TRAIN_SET = False

# Show details of training progress
__C.SHOW_TRAINING_DETAILS = False

# ===========================================
# #             Testing Config              #
# ===========================================

# Testing version name
__C.TEST_VERSION = 'with_fc_rec_mse'

# Testing checkpoint index
__C.TEST_CKP_IDX = 29

# Testing with reconstruction
__C.TEST_WITH_RECONSTRUCTION = True

# Saving testing reconstruction images
# None: not save images
__C.TEST_SAVE_IMAGE_STEP = 10  # batches

# Batch size of testing
# should be same as training batch_size
__C.TEST_BATCH_SIZE = 256

# ===========================================
# #                  Others                 #
# ===========================================

# Database name
__C.DATABASE_NAME = 'mnist'

# Source data directory path
__C.SOURCE_DATA_PATH = './data/source_data'

# Path for saving logs
__C.LOG_PATH = os.path.join('./train_logs', __C.VERSION)

# Path for saving summaries
__C.SUMMARY_PATH = os.path.join('./tf_logs', __C.VERSION)

# Path for saving model
__C.CHECKPOINT_PATH = os.path.join('./checkpoints', __C.VERSION)

# Path for saving testing logs
__C.TEST_LOG_PATH = './test_logs'

# ===========================================

# get config by: from config import cfg
config = __C
