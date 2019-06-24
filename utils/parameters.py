# coding: utf-8
# ML parameters.


import os


BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
EPOCHS = int(os.environ.get('EPOCHS', '50'))
DROPOUT = float(os.environ.get('DROPOUT', '0.5'))
DROPOUT_SEED = int(os.environ.get('DROPOUT_SEED', '42'))
EARLY_STOPPING_TEST_SIZE = float(os.environ.get('EARLY_STOPPING_TEST_SIZE', '0.2'))
EARLY_STOPPING_PATIENCE = int(os.environ.get('EARLY_STOPPING_PATIENCE', '5'))
IMG_ROWS = int(os.environ.get('IMG_ROWS', '128'))
IMG_COLS = int(os.environ.get('IMG_COLS', '128'))
NB_CHANNELS = int(os.environ.get('NB_CHANNELS', '3'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '42'))
USE_CACHE = bool(os.environ.get('USE_CACHE', 'True').lower() == 'true')
USE_ON_MEMORY = bool(os.environ.get('USE_ON_MEMORY', 'True').lower() == 'true')
if USE_ON_MEMORY:
    USE_CACHE = True
NUM_DATA_LOAD_THREAD = int(os.environ.get('NUM_DATA_LOAD_THREAD', '1'))
if NUM_DATA_LOAD_THREAD > BATCH_SIZE:
    NUM_DATA_LOAD_THREAD = BATCH_SIZE
