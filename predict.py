import os

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image

from utils import set_categories, IMG_ROWS, IMG_COLS


model = load_model(os.path.join(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'), 'model.h5'))
_, index2label = set_categories(os.environ.get('TRAINING_JOB_DATASET_IDS', '').split())


def decode_predictions(result):
    result_with_labels = [{"label": index2label[i], "probability": score} for i, score in enumerate(result)]
    return sorted(result_with_labels, key=lambda x: x['probability'], reverse=True)


def handler(_iter, ctx):
    for img in _iter:
        img = Image.fromarray(img)
        img = img.resize((IMG_ROWS, IMG_COLS))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='tf')

        result = model.predict(x)[0]
        sorted_result = decode_predictions(result.tolist())
        yield {"result": sorted_result}
