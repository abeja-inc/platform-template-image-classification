# coding: utf-8
# ML parameters.


import os


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
