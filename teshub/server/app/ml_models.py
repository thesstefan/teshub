from teshub.server.config import Config
from teshub.recognition.predictor import WeatherInFormerPredictor

config = Config()

assert config.SEG_MODEL_CKPT

weather_informer_predictor = WeatherInFormerPredictor(
    model_checkpoint_path=config.SEG_MODEL_CKPT,
)
