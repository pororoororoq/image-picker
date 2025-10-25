from transformers import pipeline, CLIPModel, CLIPProcessor

aesthetic_pipeline = pipeline(
        "image-classification",
        model="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
        device=0 
    )