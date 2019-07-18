# coding: utf-8


import io
from typing import Tuple, Dict, List

from keras.preprocessing.image import load_img
from abeja.datasets import Client

from .parameters import (
    parameters,
    BATCH_SIZE, EPOCHS, IMG_ROWS, IMG_COLS, NB_CHANNELS,
    LEARNING_RATE, ADAM_BETA_1, ADAM_BETA_2, ADAM_EPSILON, ADAM_DECAY,
    RANDOM_SEED, EARLY_STOPPING_TEST_SIZE, DROPOUT, USE_CACHE,
    USE_ON_MEMORY, NUM_DATA_LOAD_THREAD, EARLY_STOPPING_PATIENCE,
    DROPOUT_SEED, ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    BRIGHTNESS_RANGE, SHEAR_RANGE, ZOOM_RANGE, CHANNEL_SHIFT_RANGE,
    FILL_MODE, CVAL, HORIZONTAL_FLIP, VERTICAL_FLIP, RESCALE,
    DATA_FORMAT, DTYPE
)
from .dataset_item_id import DatasetItemId
from .data_generator import DataGenerator


def set_categories(dataset_ids: list) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Set categories from Datasets.
    :param dataset_ids: Dataset IDs. list format.
    :return id2index: Map of label_id to training index.
    :return index2label: Map of training index to label.
    """
    client = Client()
    id2index = dict()
    index2label = dict()
    index = 0
    for dataset_id in dataset_ids:
        dataset = client.get_dataset(dataset_id)
        category_0 = dataset.props['categories'][0]  # FIXME: Allow category selection
        for label in category_0['labels']:
            label_id = label['label_id']
            label_name = label['label']
            if label_id not in id2index:
                id2index[label_id] = index
                index2label[index] = label_name
                index += 1
        break  # FIXME: Allow multiple datasets.
    return id2index, index2label


def get_dataset_item_ids(dataset_ids: List[str]) -> List[DatasetItemId]:
    """
    FIXME: DEPRECATED https://github.com/abeja-inc/platform-planning/issues/2171
    Get dataset item ids.
    :param dataset_ids:
    :return:
    """
    client = Client()
    dataset_item_ids = list()
    for dataset_id in dataset_ids:
        dataset = client.get_dataset(dataset_id)
        for item in dataset.dataset_items.list(prefetch=USE_ON_MEMORY):
            dataset_item_id = DatasetItemId(dataset_id, item.dataset_item_id)
            dataset_item_ids.append(dataset_item_id)
            if USE_ON_MEMORY:
                try:
                    source_data = item.source_data[0]
                    file_content = source_data.get_content(cache=USE_CACHE)
                    file_like_object = io.BytesIO(file_content)
                    img = load_img(file_like_object, target_size=(IMG_ROWS, IMG_COLS))
                    dataset_item_id.data = img
                    label_id = item.attributes['classification'][0]['label_id']  # FIXME: Allow category selection
                    dataset_item_id.label_id = label_id
                except Exception as e:
                    print('Error: Loading', dataset_item_id.item_id)
                    raise e
        break  # FIXME: Allow multiple datasets.
    return dataset_item_ids
