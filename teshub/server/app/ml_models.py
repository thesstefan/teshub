from teshub.server.config import Config
from teshub.segmentation.predictor import SegmentationPredictor

config = Config()

assert config.SEG_MODEL_CKPT

seg_predictor = SegmentationPredictor(
    model_checkpoint_path=config.SEG_MODEL_CKPT,
)
