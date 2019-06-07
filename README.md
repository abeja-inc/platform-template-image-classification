# [Template] Image Classification
This is the template of image-classification task for ABEJA Platform.

This template uses transfer-learning from VGG16 ImageNet. Making it easy to try building ML model, this template uses many hard-coding parameters. You can change the parameters by setting environmental variables or editing code directly.

## Requirements
- Python 3.6.x
- [For local] Install [abeja-sdk](https://developers.abeja.io/sdk/)


## Conditions
- Transfer learning from VGG16 ImageNet.
- Allow only 1 category
- Allow only 1 dataset


## Parameters
| env | type | description |
| --- | --- | --- |
| BATCH_SIZE | int | Batch size. Default `32`. |
| EPOCHS | int | Epoch number. This template applies "Early stopping". Default `50`. |
| DROPOUT | float | Dropout of the last layer (Transfer learning). Need to be from `0.0` to `1.0`. Default `0.5`. |
| EARLY_STOPPING_TEST_SIZE | float | Test data size for "Early stopping". Need to be from `0.0` to `1.0`. Default `0.2`. |
| IMG_ROWS | int | Image rows. Automatically resize to this size. Default `128`. |
| IMG_COLS | int | Image cols. Automatically resize to this size. Default `128`. |
| NB_CHANNELS | int | Image channels. If grayscale, then `1`. If color, then `3`. Default `3`. |
| RANDOM_SEED | int | Random seed. Use it for a data shuffling. Default `42`. |
| USE_ON_MEMORY | bool | Load data on memory. If you use a big dataset, set it to `false`. Default `true` |
| USE_CACHE | bool | Image cache. If you use a big dataset, set it to `false`. If `USE_ON_MEMORY=true`, then `USE_CACHE=true` automatically. Default `true` |


## Run on local
Set environment variables.

| env | type | description |
| --- | --- | --- |
| ABEJA_ORGANIZATION_ID | str | Your organization ID. |
| ABEJA_PLATFORM_USER_ID | str | Your user ID. |
| ABEJA_PLATFORM_PERSONAL_ACCESS_TOKEN | str | Your Access Token. |
| DATASET_ID | str | Dataset ID. |

```
$ DATASET_ID='xxx' ABEJA_ORGANIZATION_ID='xxx' ABEJA_PLATFORM_USER_ID='user-xxx' ABEJA_PLATFORM_PERSONAL_ACCESS_TOKEN='xxx' python train.py
```
