import logging
import io
import json

from typing import cast
from teshub.server.app.ml_models import weather_informer_predictor
from PIL import Image


import flask
from werkzeug.datastructures import MultiDict, FileStorage

logger = logging.getLogger(__name__)
bp = flask.Blueprint('preprocess', __name__, url_prefix='/preprocess')


@bp.route('/predict', methods=cast(list[str], ['POST']))
def predict() -> flask.Response:
    request_files: MultiDict[str, FileStorage] = flask.request.files
    image_file: FileStorage = request_files['image']
    image_bytes: bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    seg_mask, labels, color_dict = (
        next(weather_informer_predictor.predict_and_process([image]))
    )

    seg_bytes = io.BytesIO()
    seg_mask.save(seg_bytes, 'png', quality=100)
    seg_bytes.seek(0)

    response = flask.send_file(seg_bytes, mimetype='image/png')

    response.headers['labels'] = json.dumps(labels)
    response.headers['color_map'] = json.dumps(color_dict)

    return response

