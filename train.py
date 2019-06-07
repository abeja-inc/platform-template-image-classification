# coding: utf-8
# Template: image classifier by vgg16 transfer learning


import os
import random

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras import applications
from keras.callbacks import TensorBoard, EarlyStopping
from abeja.contrib.keras.callbacks import Statistics

from utils import (
    set_categories, EPOCHS, IMG_ROWS, IMG_COLS, NB_CHANNELS,
    RANDOM_SEED, EARLY_STOPPING_TEST_SIZE, DROPOUT, DataGenerator, get_dataset_item_ids
)


ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')
random.seed(RANDOM_SEED)


def create_model(num_classes, input_shape):
    """
    Create ML Network.
    :param num_classes: Number of classes.
    :param input_shape: Input shape.
    :return model: ML Network.
    """

    # instantiate vgg without fully-connected layer.
    vgg16 = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=input_shape)

    # create fully-connected layer
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(DROPOUT))
    top_model.add(Dense(num_classes, activation='softmax'))

    # create new model
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

    # fix vgg layers
    for layer in model.layers[:15]:
        layer.trainable = False

    return model


def handler(context):
    dataset_alias = context.datasets
    id2index, _ = set_categories(dataset_alias.values())
    num_classes = len(id2index)
    dataset_item_ids = get_dataset_item_ids(dataset_alias.values())
    random.shuffle(dataset_item_ids)

    test_size = int(len(dataset_item_ids) * EARLY_STOPPING_TEST_SIZE)
    if test_size:
        train_ids, test_ids = dataset_item_ids[test_size:], dataset_item_ids[:test_size]
    else:
        raise Exception("Dataset size is too small. Please add more dataset.")
    input_shape = (IMG_ROWS, IMG_COLS, NB_CHANNELS)
    print('num classes:', num_classes)
    print('input shape:', input_shape)
    print(len(train_ids), 'train samples')
    print(len(test_ids), 'test samples')

    model = create_model(num_classes, input_shape)
    tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0,
                              write_graph=True, write_images=False)
    statistics = Statistics()
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    # Do you want to add `checkpoint` to callback as well?
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # fit_generator
    train_gen = DataGenerator(train_ids, id2index)
    test_gen = DataGenerator(test_ids, id2index)

    # fit_generator
    model.fit_generator(train_gen,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=test_gen,
                        callbacks=[tensorboard, statistics, early])
    score = model.evaluate_generator(test_gen)
    model.save(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.h5'))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    class DebugContext:
        datasets = {"data": os.environ.get('DATASET_ID')}
    handler(DebugContext())
