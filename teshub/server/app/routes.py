import logging
import io

from typing import cast
from teshub.server.app.ml_models import seg_predictor
from teshub.segmentation.utils import DEFAULT_ID2COLOR
from teshub.visualization.transforms import seg_mask_to_image
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

    predicted_seg = seg_predictor.predict(image)
    seg_image = seg_mask_to_image(predicted_seg[0], DEFAULT_ID2COLOR)

    seg_bytes = io.BytesIO()
    seg_image.save(seg_bytes, 'png', quality=100)
    seg_bytes.seek(0)

    return flask.send_file(seg_bytes, mimetype='image/png')
