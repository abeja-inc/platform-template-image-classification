import datetime
import http
import json
import os
import traceback
from io import BytesIO

from keras.models import load_model
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

from abeja.datasets import Client
from preprocessor import PreProcessor
from utils import set_categories, IMG_ROWS, IMG_COLS
from utils.image_utils import plot_confusion_matrix


model = load_model(os.path.join(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'), 'model.h5'))
id2index, index2label = set_categories(os.environ.get('TRAINING_JOB_DATASET_IDS', '').split(','))
preprocessor = PreProcessor()

ABEJA_PREDICTION_RESULT_DIR = os.environ.get('ABEJA_PREDICTION_RESULT_DIR', '.')


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
        img = img.convert('RGB')
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


def evaluate(request, context):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_datasets = json.loads(os.environ.get('ABEJA_EVALUATION_DATASETS'))
    client = Client()

    for i, (alias, dataset_id) in enumerate(eval_datasets.items()):
        log_path = os.path.join(ABEJA_PREDICTION_RESULT_DIR, 'logs', 'eval', current_time, alias)
        writer = SummaryWriter(log_dir=log_path)
        dataset = client.get_dataset(dataset_id)
        golds = []
        preds = []
        for item in dataset.dataset_items.list():
            label_id = item.attributes['classification'][0]['label_id']  # FIXME: Allow category selection
            gold_idx = id2index[label_id]
            golds.append(gold_idx)

            source_data = item.source_data[0]
            file_content = source_data.get_content()
            img = BytesIO(file_content)
            img = Image.open(img)
            img = img.convert('RGB')
            img = img.resize((IMG_ROWS, IMG_COLS))
            x = preprocessor.transform(img)
            x = np.expand_dims(x, axis=0)
            result = model.predict(x)
            pred = np.argmax(result, axis=1)
            preds.extend(pred)
            if len(preds) == 5:
                break
        cm = confusion_matrix([0,1,2,3,4], [0,1,2,3,4])
        figure = plot_confusion_matrix(cm, class_names=index2label.values())
        writer.add_figure('Confusion Matrix', figure, i)
        writer.close()

evaluate(None, None)
