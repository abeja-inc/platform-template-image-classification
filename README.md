# [Template] Image Classification
This is the template of image-classification task for ABEJA Platform.

This template uses transfer-learning from VGG16 ImageNet. Making it easy to try building ML model, this template uses many hard-coding parameters. You can change the parameters by setting environmental variables or editing code directly.


### Note
If you want to use **GPU**, you need to edit "requirements-local.txt" like the below.

```
# tensorflow==1.13.1
tensorflow-gpu==1.13.1
```


## Requirements
- Python 3.6.x
- [For local] Install [abeja-sdk](https://developers.abeja.io/sdk/)


## Docker
- abeja/all-cpu:19.04
- abeja/all-gpu:19.04


## Conditions
- Transfer learning from VGG16 ImageNet.
- Allow only 1 category
- Allow only 1 dataset


## Parameters
| env | type | description |
| --- | --- | --- |
| BATCH_SIZE | int | Batch size. Default `32`. |
| EPOCHS | int | Epoch number. This template applies "Early stopping". Default `50`. |
| LEARNING_RATE | float | Learning rate. Need to be from `0.0` to `1.0`. Default `0.0001`. |
| ADAM_BETA_1 | float | Adam parameter "beta_1". Need to be from `0.0` to `1.0`. Default `0.9`. |
| ADAM_BETA_2 | float | Adam parameter "beta_2". Need to be from `0.0` to `1.0`. Default `0.999`. |
| ADAM_EPSILON | float | Adam parameter "epsilon". Need to be from `0.0`. Default `None` = `K.epsilon()`. |
| ADAM_DECAY | float | Adam parameter "decay". Need to be from `0.0`. Default `0.0`. |
| DROPOUT | float | Dropout of the last layer (Transfer learning). Need to be from `0.0` to `1.0`. Default `0.5`. |
| DROPOUT_SEED | int | Random seed for Dropout. Default `42`. |
| EARLY_STOPPING_TEST_SIZE | float | Test data size for "Early stopping". Need to be from `0.0` to `1.0`. Default `0.2`. |
| EARLY_STOPPING_PATIENCE | int | Number of patience for "Early stopping". Default `5`. |
| IMG_ROWS | int | Image rows. Automatically resize to this size. Default `224`. |
| IMG_COLS | int | Image cols. Automatically resize to this size. Default `224`. |
| NB_CHANNELS | int | Image channels. If grayscale, then `1`. If color, then `3`. Default `3`. |
| RANDOM_SEED | int | Random seed. Use it for a data shuffling. Default `42`. |
| USE_ON_MEMORY | bool | Load data on memory. If you use a big dataset, set it to `false`. Default `true` |
| USE_CACHE | bool | Image cache. If you use a big dataset, set it to `false`. If `USE_ON_MEMORY=true`, then `USE_CACHE=true` automatically. Default `true` |
| NUM_DATA_LOAD_THREAD | int | Number of thread image loads. MUST NOT over `BATCH_SIZE`. Default `1` |
| ROTATION_RANGE | int | Degree range for random rotations. Default `20` |
| WIDTH_SHIFT_RANGE | float | Fraction of total width, if < 1, or pixels if >= 1. Default `0.05`. |
| HEIGHT_SHIFT_RANGE | float | Fraction of total height, if < 1, or pixels if >= 1. Default `0.05`. |
| BRIGHTNESS_RANGE | string | CSV format of two floats. Range for picking a brightness shift value from. Default `None`. |
| SHEAR_RANGE | float | Shear Intensity. Default `0.`. |
| ZOOM_RANGE | float | Range for random zoom. `[lower, upper] = [1-zoom_range, 1+zoom_range]`. Default `0.`. |
| CHANNEL_SHIFT_RANGE | float | Range for random channel shifts. Default `0.`. |
| FILL_MODE | string | Points outside the boundaries of the input are filled according to the given mode. One of {"constant", "nearest", "reflect" or "wrap"}. Default `nearest`. |
| CVAL | float | Value used for points outside the boundaries when `fill_mode = "constant"`. Default `0.`. |
| HORIZONTAL_FLIP | bool | Randomly flip inputs horizontally. Default `True`. |
| VERTICAL_FLIP | bool | Randomly flip inputs vertically. Default `False`. |
| RESCALE | float | Rescaling factor. If 0, no rescaling is applied. Default `0.`. |
| DATA_FORMAT | string | Image data format, either "channels_first" or "channels_last". Default `channels_last`. |
| DTYPE | string | Dtype to use for the generated arrays. Default `float32`. |


## Run on local
Use `requirements-local.txt`.

```
$ pip install -r requirements-local.txt
```

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
