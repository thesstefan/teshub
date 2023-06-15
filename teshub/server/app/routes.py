import io
import json
import logging
from typing import cast

import flask
from PIL import Image
from torchvision.transforms.functional import (gaussian_blur, normalize,
                                               pil_to_tensor, to_tensor, to_pil_image)
from werkzeug.datastructures import FileStorage, MultiDict

from teshub.server.app.ml_models import (morph_add_clouds, morph_add_fog,
                                         morph_add_snow,
                                         weather_informer_predictor)
from teshub.translation.config.add_fog_config import AddFogConfig
from teshub.translation.config.add_snow_config import AddSnowConfig
from teshub.translation.config.clear2cloudy_config import Clear2CloudyConfig
from teshub.visualization.transforms import rgb_pixels_to_1d
from teshub.recognition.utils import DEFAULT_SEG_COLOR2ID
import torch

logger = logging.getLogger(__name__)
bp = flask.Blueprint('predict', __name__, url_prefix='/predict')


@bp.route('/segmentation', methods=cast(list[str], ['POST']))
def predict_segmentation() -> flask.Response:
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


def choose_translation(translation_type: str):
    if translation_type == "add_snow":
        return morph_add_snow, AddSnowConfig.attention_map

    if translation_type == "add_fog":
        return morph_add_fog, AddFogConfig.attention_map

    if translation_type == "add_clouds":
        return morph_add_clouds, Clear2CloudyConfig.attention_map

    raise RuntimeError(f"Translation type unsupported: {translation_type}")


@bp.route('/translation/<method>', methods=cast(list[str], ['POST']))
def predict_translation(method: str) -> flask.Response:
    request_files: MultiDict[str, FileStorage] = flask.request.files

    image_file: FileStorage = request_files['image']
    seg_file: FileStorage = request_files['seg']

    image_bytes: bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).resize(
        (512, 512), Image.Resampling.BILINEAR)

    seg_bytes: bytes = seg_file.read()
    seg = Image.open(io.BytesIO(seg_bytes)).resize((512, 512),
                     Image.Resampling.NEAREST)

    source = to_tensor(image)
    seg = rgb_pixels_to_1d(pil_to_tensor(
        seg), rgb_pixel_to_value=DEFAULT_SEG_COLOR2ID)

    model, attention_map = choose_translation(method)

    attention = attention_map(seg)
    attention = torch.maximum(attention, gaussian_blur(
        attention, kernel_size=101, sigma=50.0))

    source = normalize(source, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    gen = model(torch.unsqueeze(source, dim=0),
                torch.unsqueeze(attention, dim=0))[0]
    gen= normalize(gen, (0.0, 0.0, 0.0), (2, 2, 2))
    gen= normalize(gen, (-0.5, -0.5, -0.5), (1, 1, 1))


    gen = to_pil_image(gen).convert('RGB')
    gen_bytes = io.BytesIO()
    gen.save(gen_bytes, 'png', quality=100)
    gen_bytes.seek(0)

    response = flask.send_file(gen_bytes, mimetype='image/png')

    return response
