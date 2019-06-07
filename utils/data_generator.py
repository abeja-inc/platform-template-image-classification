# coding: utf-8


import io
import math
from typing import Tuple, Dict, List

import numpy as np
import keras
from keras.preprocessing.image import load_img
from keras.utils import Sequence
from abeja.datasets import Client

from preprocessor import preprocessor
from .dataset_item_id import DatasetItemId
from .parameters import (
    BATCH_SIZE, EPOCHS, IMG_ROWS, IMG_COLS, NB_CHANNELS,
    RANDOM_SEED, EARLY_STOPPING_TEST_SIZE, DROPOUT, USE_CACHE,
    USE_ON_MEMORY
)


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
            img = preprocessor(img)
            imgs[i, :] = img
            labels[i]=label

        labels = keras.utils.to_categorical(labels, num_classes=self.num_classes)
        return imgs, labels

    def __len__(self):
        return self.num_batches_per_epoch
