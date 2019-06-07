# coding: utf-8


from typing import Tuple, Dict, List

from abeja.datasets import Client

from .parameters import (
    BATCH_SIZE, EPOCHS, IMG_ROWS, IMG_COLS, NB_CHANNELS, RANDOM_SEED, EARLY_STOPPING_TEST_SIZE, DROPOUT, USE_CACHE
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
        for item in dataset.dataset_items.list():
            dataset_item_id = DatasetItemId(dataset_id, item.dataset_item_id)
            dataset_item_ids.append(dataset_item_id)
        break  # FIXME: Allow multiple datasets.
    return dataset_item_ids
