import http
import os
import traceback
from io import BytesIO

from keras.models import load_model
import numpy as np
from PIL import Image

from preprocessor import PreProcessor
from utils import set_categories, IMG_ROWS, IMG_COLS


model = load_model(os.path.join(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'), 'model.h5'))
_, index2label = set_categories(os.environ.get('TRAINING_JOB_DATASET_IDS', '').split())
preprocessor = PreProcessor()


def decode_predictions(result):
    result_with_labels = [{"label": index2label[i], "probability": score} for i, score in enumerate(result)]
    return sorted(result_with_labels, key=lambda x: x['probability'], reverse=True)


def handler(request, context):
    print('Start predict handler.')
    if 'http_method' not in request:
        message = 'Error: Support only "abeja/all-cpu:19.04" or "abeja/all-gpu:19.04".'
        print(message)
        return {
            'status_code': http.HTTPStatus.BAD_REQUEST,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': message}
        }

    try:
        data = request.read()
        img = BytesIO(data)
        img = Image.open(img)
        img = np.asarray(img)
        img = Image.fromarray(img)
        img = img.resize((IMG_ROWS, IMG_COLS))

        x = preprocessor.transform(img)
        x = np.expand_dims(x, axis=0)

        result = model.predict(x)[0]
        sorted_result = decode_predictions(result.tolist())

        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': {'result': sorted_result}
        }
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return {
            'status_code': http.HTTPStatus.INTERNAL_SERVER_ERROR,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': str(e)}
        }
