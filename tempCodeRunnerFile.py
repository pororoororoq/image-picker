#import gradio as gr
import torch
#import numpy as np
import cv2
import json
from PIL import Image
from io import BytesIO

# =============== Device & Globals ===============
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Optional/available flags
HAS_LAION = False
HAS_CLIP = False
HAS_FER = False

# =============== Model Loads ===============
# 1) LAION Aesthetic predictor (1-10 scale)
try:
    from transformers import pipeline, CLIPModel, CLIPProcessor
    aesthetic_pipeline = pipeline(
        "image-classification",
        model="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
        device=0 if device == "cuda" else -1
    )
    HAS_LAION = True
    print("✓ LAION Aesthetics V2 loaded")
except Exception as e:
    print(f"✗ Failed to load LAION Aesthetics V2: {e}")
    aesthetic_pipeline = None