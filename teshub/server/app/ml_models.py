import torch

from teshub.recognition.predictor import WeatherInFormerPredictor
from teshub.server.config import Config
from teshub.translation.weather_morph import WeatherMorph

config = Config()

assert config.SEG_MODEL_CKPT
assert config.MORPH_ADD_SNOW_CKPT
assert config.MORPH_ADD_FOG_CKPT
assert config.MORPH_ADD_CLOUDS_CKPT

weather_informer_predictor = WeatherInFormerPredictor(
    model_checkpoint_path=config.SEG_MODEL_CKPT,
)

morph_add_snow = WeatherMorph.load_from_checkpoint(
    weather_informer_model="INFORMER_XL",
    weather_informer_ckpt_path=config.SEG_MODEL_CKPT,
    checkpoint_path=config.MORPH_ADD_SNOW_CKPT,
    map_location=torch.device('cpu'),
)

morph_add_fog = WeatherMorph.load_from_checkpoint(
    weather_informer_model="INFORMER_XL",
    weather_informer_ckpt_path=config.SEG_MODEL_CKPT,
    checkpoint_path=config.MORPH_ADD_FOG_CKPT,
    map_location=torch.device('cpu'),
)

morph_add_clouds = WeatherMorph.load_from_checkpoint(
    weather_informer_model="INFORMER_XL",
    weather_informer_ckpt_path=config.SEG_MODEL_CKPT,
    checkpoint_path=config.MORPH_ADD_CLOUDS_CKPT,
    map_location=torch.device('cpu')
)
