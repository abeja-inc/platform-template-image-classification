# coding: utf-8


class DatasetItemId:
    def __init__(self, dataset_id: str, dataset_item_id: str):
        self.dataset_id = dataset_id
        self.dataset_item_id = dataset_item_id

    def get_key(self):
        return f'{self.dataset_id}-{self.dataset_item_id}'
