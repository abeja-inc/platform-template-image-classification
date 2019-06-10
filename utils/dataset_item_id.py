# coding: utf-8


class DatasetItemId:
    def __init__(self, dataset_id: str, item_id: str):
        self.dataset_id = dataset_id
        self.item_id = item_id
        self.data = None
        self.label_id = None
