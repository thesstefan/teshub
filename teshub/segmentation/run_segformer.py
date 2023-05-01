from transformers import SegformerFeatureExtractor

pretrained_model_name = "nvidia/mit-b0"
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",

)
