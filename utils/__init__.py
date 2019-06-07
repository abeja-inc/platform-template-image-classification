# coding: utf-8


import io
import math
from typing import Tuple, Dict, List

import numpy as np
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from abeja.datasets import Client

from .parameters import (
    BATCH_SIZE, EPOCHS, IMG_ROWS, IMG_COLS, NB_CHANNELS, RANDOM_SEED, EARLY_STOPPING_TEST_SIZE, DROPOUT, USE_CACHE
)


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


class DatasetItemId:
    def __init__(self, dataset_id: str, dataset_item_id: str):
        self.dataset_id = dataset_id
        self.dataset_item_id = dataset_item_id


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
        for item in dataset.dataset_items.list():
            dataset_item_ids.append(DatasetItemId(dataset_id, item.dataset_item_id))
        break  # FIXME: Allow multiple datasets.
    return dataset_item_ids


class DataGenerator(Sequence):
    """
    Custom Data Generator for ABEJA Datasets
    FIXME: Allow multiple datasets.
    """

    def __init__(self, dataset_item_ids: List[DatasetItemId], id2index: Dict[str, int]):
        self.client = Client()
        self.dataset_item_ids = dataset_item_ids
        self.dataset = self.client.get_dataset(self.dataset_item_ids[0].dataset_id)
        self.id2index = id2index
        self.num_classes = len(id2index)

        # FIXME: https://github.com/abeja-inc/platform-planning/issues/2170
        self.dataset_item_count = len(dataset_item_ids)
        self.num_batches_per_epoch = math.ceil(self.dataset_item_count / BATCH_SIZE)

    def __getitem__(self, idx):
        start_pos = BATCH_SIZE * idx
        imgs = np.empty((BATCH_SIZE, IMG_ROWS, IMG_COLS, NB_CHANNELS), dtype=np.float32)
        labels = [0]*BATCH_SIZE
        for i in range(BATCH_SIZE):
            id_idx = (start_pos + i) % self.dataset_item_count
            dataset_id = self.dataset_item_ids[id_idx].dataset_id
            dataset_item_id = self.dataset_item_ids[id_idx].dataset_item_id
            if self.dataset.dataset_id != dataset_id:
                self.dataset = self.client.get_dataset(dataset_id)
            dataset_item = self.dataset.dataset_items.get(dataset_item_id)
            # FIXME: Allow category selection
            label = self.id2index[dataset_item.attributes['classification'][0]['label_id']]
            source_data = dataset_item.source_data[0]
            file_content = source_data.get_content(cache=USE_CACHE)
            file_like_object = io.BytesIO(file_content)
            img = load_img(file_like_object, target_size=(IMG_ROWS, IMG_COLS))
            img = img_to_array(img)
            img = preprocess_input(img, mode='tf')
            imgs[i, :] = img
            labels[i]=label

        labels = keras.utils.to_categorical(labels, num_classes=self.num_classes)
        return imgs, labels

    def __len__(self):
        return self.num_batches_per_epoch
