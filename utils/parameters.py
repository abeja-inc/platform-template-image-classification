# coding: utf-8
# ML parameters.


import os


# Train params
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
EPOCHS = int(os.environ.get('EPOCHS', '50'))
DROPOUT = float(os.environ.get('DROPOUT', '0.5'))
DROPOUT_SEED = int(os.environ.get('DROPOUT_SEED', '42'))
EARLY_STOPPING_TEST_SIZE = float(os.environ.get('EARLY_STOPPING_TEST_SIZE', '0.2'))
EARLY_STOPPING_PATIENCE = int(os.environ.get('EARLY_STOPPING_PATIENCE', '5'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.0001'))
ADAM_BETA_1 = float(os.environ.get('ADAM_BETA_1', '0.9'))
ADAM_BETA_2 = float(os.environ.get('ADAM_BETA_2', '0.999'))
ADAM_EPSILON = os.environ.get('ADAM_EPSILON')
if ADAM_EPSILON:
    ADAM_EPSILON = float(ADAM_EPSILON)
ADAM_DECAY = float(os.environ.get('ADAM_DECAY', '0.0'))
IMG_ROWS = int(os.environ.get('IMG_ROWS', '224'))
IMG_COLS = int(os.environ.get('IMG_COLS', '224'))
NB_CHANNELS = int(os.environ.get('NB_CHANNELS', '3'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '42'))
USE_CACHE = bool(os.environ.get('USE_CACHE', 'True').lower() == 'true')
USE_ON_MEMORY = bool(os.environ.get('USE_ON_MEMORY', 'True').lower() == 'true')
if USE_ON_MEMORY:
    USE_CACHE = True
NUM_DATA_LOAD_THREAD = int(os.environ.get('NUM_DATA_LOAD_THREAD', '1'))
if NUM_DATA_LOAD_THREAD > BATCH_SIZE:
    NUM_DATA_LOAD_THREAD = BATCH_SIZE

# Augmentation params
ROTATION_RANGE = int(os.environ.get('ROTATION_RANGE', '20'))
WIDTH_SHIFT_RANGE = float(os.environ.get('WIDTH_SHIFT_RANGE', '0.05'))
HEIGHT_SHIFT_RANGE = float(os.environ.get('HEIGHT_SHIFT_RANGE', '0.05'))
BRIGHTNESS_RANGE = os.environ.get('BRIGHTNESS_RANGE')
if BRIGHTNESS_RANGE:
    BRIGHTNESS_RANGE = [float(x) for x in BRIGHTNESS_RANGE.split(",")]
SHEAR_RANGE = float(os.environ.get('SHEAR_RANGE', '0.'))
ZOOM_RANGE = float(os.environ.get('ZOOM_RANGE', '0.'))
CHANNEL_SHIFT_RANGE = float(os.environ.get('CHANNEL_SHIFT_RANGE', '0.'))
FILL_MODE = os.environ.get('FILL_MODE', 'nearest')
CVAL = float(os.environ.get('CVAL', '0.'))
HORIZONTAL_FLIP = bool(os.environ.get('HORIZONTAL_FLIP', 'True').lower() == 'true')
VERTICAL_FLIP = bool(os.environ.get('VERTICAL_FLIP', 'False').lower() == 'true')
RESCALE = float(os.environ.get('RESCALE', '0.'))
DATA_FORMAT = os.environ.get('DATA_FORMAT', 'channels_last')
DTYPE = os.environ.get('DTYPE', 'float32')

# For print
parameters = {
    'BATCH_SIZE': BATCH_SIZE,
    'EPOCHS': EPOCHS,
    'IMG_ROWS': IMG_ROWS,
    'IMG_COLS': IMG_COLS,
    'NB_CHANNELS': NB_CHANNELS,
    'LEARNING_RATE': LEARNING_RATE,
    'ADAM_BETA_1': ADAM_BETA_1,
    'ADAM_BETA_2': ADAM_BETA_2,
    'ADAM_EPSILON': ADAM_EPSILON,
    'ADAM_DECAY': ADAM_DECAY,
    'RANDOM_SEED': RANDOM_SEED,
    'EARLY_STOPPING_TEST_SIZE': EARLY_STOPPING_TEST_SIZE,
    'DROPOUT': DROPOUT,
    'USE_CACHE': USE_CACHE,
    'USE_ON_MEMORY': USE_ON_MEMORY,
    'NUM_DATA_LOAD_THREAD': NUM_DATA_LOAD_THREAD,
    'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
    'DROPOUT_SEED': DROPOUT_SEED,
    'ROTATION_RANGE': ROTATION_RANGE,
    'WIDTH_SHIFT_RANGE': WIDTH_SHIFT_RANGE,
    'HEIGHT_SHIFT_RANGE': HEIGHT_SHIFT_RANGE,
    'BRIGHTNESS_RANGE': BRIGHTNESS_RANGE,
    'SHEAR_RANGE': SHEAR_RANGE,
    'ZOOM_RANGE': ZOOM_RANGE,
    'CHANNEL_SHIFT_RANGE': CHANNEL_SHIFT_RANGE,
    'FILL_MODE': FILL_MODE,
    'CVAL': CVAL,
    'HORIZONTAL_FLIP': HORIZONTAL_FLIP,
    'VERTICAL_FLIP': VERTICAL_FLIP,
    'RESCALE': RESCALE,
    'DATA_FORMAT': DATA_FORMAT,
    'DTYPE': DTYPE
}
