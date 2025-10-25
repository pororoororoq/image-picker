import gradio as gr
import torch
import numpy as np
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

# 2) CLIP for composition scoring (semantic prompts)
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    HAS_CLIP = True
    print("✓ CLIP (ViT-B/32) loaded for composition prompts")
except Exception as e:
    print(f"✗ Failed to load CLIP: {e}")
    clip_model, clip_processor = None, None

# 3) FER (emotion recognition) – optional
try:
    # pip install fer
    from fer import FER
    fer_detector = FER(mtcnn=True)
    HAS_FER = True
    print("✓ FER loaded (emotion detection)")
except Exception as e:
    print(f"⚠ Emotion model not available (FER): {e}")
    fer_detector = None

# 4) OpenCV face detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# =============== Utilities ===============
def pil_to_np_rgb(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)

def detect_faces_np(img_rgb: np.ndarray):
    """Return list of (x, y, w, h). Try frontal then profile."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    return faces

def face_coverage(img_rgb: np.ndarray, faces) -> float:
    if len(faces) == 0:
        return 0.0
    H, W = img_rgb.shape[:2]
    total = float(H * W)
    area = sum(int(w*h) for (_, _, w, h) in faces)
    return float(area) / total if total > 0 else 0.0

def face_position_score(img_rgb: np.ndarray, faces) -> float:
    """Rule-of-thirds alignment; return ~[2, 9]. Multiple faces avg."""
    if len(faces) == 0:
        return 5.0
    H, W = img_rgb.shape[:2]
    thirds = (W/3, 2*W/3, H/3, 2*H/3)

    scores = []
    for (x, y, w, h) in faces:
        cx, cy = x + w/2, y + h/2
        x_dist = min(abs(cx - thirds[0]), abs(cx - thirds[1]))
        y_dist = min(abs(cy - thirds[2]), abs(cy - thirds[3]))
        # normalized to [0..1], 0 best
        x_norm = max(0.0, 1.0 - x_dist / (W/6))
        y_norm = max(0.0, 1.0 - y_dist / (H/6))
        s = 5.0 + 2.0*(x_norm + y_norm)  # 5..9
        scores.append(s)
    return float(np.mean(scores)) if scores else 5.0

def face_sharpness(img_rgb: np.ndarray, faces) -> float:
    """
    Laplacian/gradient sharpness only on face regions. If multiple, average.
    Return a raw number; later normalized.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if len(faces) == 0:
        # center patch fallback (40%)
        H, W = gray.shape
        y1, y2 = int(H*0.3), int(H*0.7)
        x1, x2 = int(W*0.3), int(W*0.7)
        patch = gray[y1:y2, x1:x2]
        return _sharpness_metrics(patch)

    vals = []
    for (x, y, w, h) in faces:
        pad = int(max(w, h) * 0.15)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        patch = gray[y1:y2, x1:x2]
        vals.append(_sharpness_metrics(patch))
    return float(np.mean(vals)) if vals else 0.0

def _sharpness_metrics(gray_patch: np.ndarray) -> float:
    lap_var = cv2.Laplacian(gray_patch, cv2.CV_64F).var()
    sobelx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.sqrt(sobelx**2 + sobely**2).mean())
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    mod_lap = cv2.filter2D(gray_patch, cv2.CV_64F, kernel)
    focus_var = float(np.var(mod_lap))
    # Weighted sum (scaled down)
    return 0.5*lap_var + 0.3*grad_mag + 0.2*focus_var

def lighting_exposure_score(img_rgb: np.ndarray) -> float:
    """
    Histogram-based lighting. Penalize under/over exposure, low contrast.
    Return ~[2..9]
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-6)

    # Brightness: mean intensity (0..255)
    bins = np.arange(256)
    mean_intensity = (hist_norm * bins).sum()

    # Contrast: std
    contrast = np.sqrt(((bins - mean_intensity)**2 * hist_norm).sum())

    # Clip ratios: % near black/white
    dark_ratio = hist_norm[:8].sum()
    bright_ratio = hist_norm[-8:].sum()

    score = 5.5
    # ideal brightness around 110..150
    if mean_intensity < 90 or mean_intensity > 170:
        score -= 1.0
    # contrast boost
    score += min(2.0, max(0.0, (contrast - 40) / 20.0))  # contrast > ~40 is good
    # penalize clipping
    score -= min(2.0, (dark_ratio + bright_ratio) * 4.0)

    return float(np.clip(score, 2.0, 9.0))

def emotion_score(img_rgb: np.ndarray, faces) -> (float, str):
    """
    Use FER if available; otherwise return neutral 5.0.
    Returns (score[0..10], top_emotion_str)
    """
    if not HAS_FER or fer_detector is None:
        return 5.0, "unknown"

    # FER prefers BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    try:
        results = fer_detector.detect_emotions(img_bgr)
        if not results:
            return 5.0, "unknown"

        # Average happiness/positive affect across faces
        emotions = []
        for r in results:
            em = r.get("emotions", {})
            # happiness as main proxy; surprise can be positive; fear/angry/sad negative
            pos = em.get("happy", 0.0) + 0.5*em.get("surprise", 0.0)
            neg = 0.7*em.get("angry", 0.0) + 0.7*em.get("sad", 0.0) + 0.5*em.get("fear", 0.0) + 0.5*em.get("disgust", 0.0)
            score = 5.0 + 5.0*(pos - neg)  # center 5, +/- up to ~5
            emotions.append(np.clip(score, 0.0, 10.0))
        # simple top emotion by first face
        top_emotion = max(results[0].get("emotions", {}), key=results[0].get("emotions", {}).get, default="unknown")
        return float(np.mean(emotions)), top_emotion
    except Exception:
        return 5.0, "unknown"

def laion_aesthetic_score(image: Image.Image) -> float:
    if not HAS_LAION or aesthetic_pipeline is None:
        return 5.0
    try:
        # shunk031 model returns a single score in [1..10]
        out = aesthetic_pipeline(image)
        # Some versions return [{'label': '...', 'score': float}] or direct score
        if isinstance(out, list) and len(out) > 0 and 'score' in out[0]:
            return float(out[0]['score'])
        # Fallback to average if structure differs
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            vals = [float(d.get('score', 0)) for d in out]
            vals = [v for v in vals if v > 0]
            return float(np.mean(vals)) if vals else 5.0
        return 5.0
    except Exception as e:
        print(f"LAION score error: {e}")
        return 5.0

def clip_composition_score(image: Image.Image) -> float:
    """
    Use CLIP similarity against prompts describing good vs poor composition.
    Return score on ~[2..9].
    """
    if not HAS_CLIP or clip_model is None or clip_processor is None:
        return 5.0

    prompts_good = [
        "a photo with excellent composition",
        "a well-balanced portrait",
        "a professional yearbook photo",
        "a portrait with clear subject and clean background",
    ]
    prompts_bad = [
        "a poorly composed photo",
        "an off-center unbalanced portrait",
        "an amateur snapshot with cluttered background",
    ]
    prompts = prompts_good + prompts_bad

    try:
        inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # [1, len(prompts)]
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        good = float(np.sum(probs[:len(prompts_good)]))
        bad  = float(np.sum(probs[len(prompts_good):]))
        raw = good - bad  # [-1..1]
        # map to ~[2..9]
        score = 5.5 + 3.5*raw
        return float(np.clip(score, 2.0, 9.0))
    except Exception as e:
        print(f"CLIP composition error: {e}")
        return 5.0

def normalize_face_sharpness(raw_value: float) -> float:
    """
    Map raw sharpness to 0..10 with soft cap.
    Typical Laplacian-var-derived combined values ~[20..250+].
    """
    # gentle S-shaped mapping
    x = max(0.0, raw_value)
    score = 10.0 * (1.0 - np.exp(-x / 120.0))
    return float(np.clip(score, 0.0, 10.0))

def face_count_score(count: int) -> float:
    """
    Reward 1–5 faces (typical yearbook groups); beyond that mildly penalize.
    """
    if count == 0:
        return 3.5
    if 1 <= count <= 5:
        return 6.0 + 0.8*(count-1)  # 1->6.0, 5->9.2
    # 6+ faces: diminishing returns
    return 7.5 - 0.2*(count - 5)

def face_coverage_score(coverage: float) -> float:
    """
    Encourage larger visible faces (e.g., 10%–50% total coverage).
    """
    # convert fraction to percent
    p = 100.0 * max(0.0, min(0.7, coverage))
    # 0% -> 3.5, 10% -> ~6.5, 25% -> ~8.0, >=50% -> ~9.0
    score = 3.5 + 0.11 * p - 0.0006 * (p**2)  # concave
    return float(np.clip(score, 2.0, 9.5))

def recommend(final_score: float, faces_len: int, sharp_score: float, laion: float) -> (str, str):
    if final_score >= 8.2 and faces_len > 0 and sharp_score >= 6.5:
        return "use", "Ready to use – sharp faces, engaging composition"
    if final_score >= 7.3 and sharp_score >= 5.5:
        return "enhance", "Strong candidate – slight sharpening or exposure tweaks"
    if final_score >= 6.0:
        return "maybe", "Acceptable if needed – check expression/lighting"
    return "skip", "Not recommended – subject clarity or composition below goal"

# =============== Core Processing ===============
def analyze_single_image(image, debug_enhance_flag):
    if image is None:
        return {"status": "error", "error": "No image provided"}

    # Handle gradio filepath, np array, file obj
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    elif hasattr(image, "name"):
        img = Image.open(image.name).convert("RGB")
    elif hasattr(image, "read"):
        img = Image.open(image).convert("RGB")
    else:
        return {"status": "error", "error": f"Unsupported image type: {type(image)}"}

    img_rgb = pil_to_np_rgb(img)
    H, W = img_rgb.shape[:2]
    mpix = round((W*H)/1_000_000, 2)

    # --- Face analysis
    faces = detect_faces_np(img_rgb)
    fcov = face_coverage(img_rgb, faces)
    fpos = face_position_score(img_rgb, faces)
    fsharp_raw = face_sharpness(img_rgb, faces)
    fsharp = normalize_face_sharpness(fsharp_raw)
    fcount = len(faces)

    # --- Emotion
    emo_score, emo_label = emotion_score(img_rgb, faces)

    # --- Lighting
    light_score = lighting_exposure_score(img_rgb)

    # --- CLIP composition
    comp_score = clip_composition_score(img)

    # --- LAION aesthetic
    laion_score_val = laion_aesthetic_score(img)

    # --- Weighted final score (yearbook-optimized)
    # Weights sum to 1.0
    w = {
        "laion": 0.30,
        "clip_comp": 0.20,
        "face_cov": 0.12,
        "face_count": 0.08,
        "face_sharp": 0.15,
        "emotion": 0.08,
        "face_pos": 0.04,
        "lighting": 0.03
    }

    score_face_cov = face_coverage_score(fcov)
    score_face_cnt = face_count_score(fcount)

    final_score = (
        laion_score_val * w["laion"] +
        comp_score * w["clip_comp"] +
        score_face_cov * w["face_cov"] +
        score_face_cnt * w["face_count"] +
        fsharp * w["face_sharp"] +
        emo_score * w["emotion"] +
        fpos * w["face_pos"] +
        light_score * w["lighting"]
    )

    final_score = float(np.clip(final_score, 0.0, 10.0))

    reco, action = recommend(final_score, fcount, fsharp, laion_score_val)

    return {
        "status": "success",
        "image_info": {
            "width": W,
            "height": H,
            "megapixels": mpix
        },
        "models": {
            "laion_loaded": HAS_LAION,
            "clip_loaded": HAS_CLIP,
            "fer_loaded": HAS_FER
        },
        "scores": {
            "aesthetic_laion": round(laion_score_val, 2),
            "composition_clip": round(comp_score, 2),
            "face_coverage_score": round(score_face_cov, 2),
            "face_count_score": round(score_face_cnt, 2),
            "face_sharpness_score": round(fsharp, 2),
            "emotion_score": round(emo_score, 2),
            "face_position_score": round(fpos, 2),
            "lighting_score": round(light_score, 2),
            "final_score": round(final_score, 2)
        },
        "face_metrics": {
            "face_count": fcount,
            "face_coverage_ratio": round(fcov, 4),
            "face_sharpness_raw": round(fsharp_raw, 2),
            "top_emotion": emo_label
        },
        "analysis": {
            "recommendation": reco,
            "action": action
        }
    }

def analyze_batch(files, debug_enhance_flag):
    if not files:
        return {"status": "error", "error": "No images provided"}
    results = []
    for i, f in enumerate(files):
        try:
            res = analyze_single_image(f, debug_enhance_flag)
            res["image_index"] = i
            results.append(res)
        except Exception as e:
            results.append({"status":"error", "error":str(e), "image_index": i})
    return {"status": "success", "total_images": len(files), "results": results}

# =============== Gradio UI ===============
with gr.Blocks(title="Yearbook Photo ML Evaluator (LAION + CLIP + Face)") as demo:
    gr.Markdown("# 📚 Yearbook Photo Evaluator\n**LAION aesthetics + CLIP composition + face clarity/emotion + lighting**")

    with gr.Tab("Single Image"):
        with gr.Row():
            image_input = gr.Image(type="filepath", label="Upload Image")
            enhance_check = gr.Checkbox(label="(Debug toggle – no enhancement applied)", value=False)
        analyze_btn = gr.Button("Analyze Photo", variant="primary")
        output = gr.JSON(label="Analysis Results")
        analyze_btn.click(analyze_single_image, inputs=[image_input, enhance_check], outputs=output, api_name="predict")

    with gr.Tab("Batch"):
        batch_input = gr.File(file_count="multiple", file_types=["image"], label="Upload Multiple Images")
        batch_enhance = gr.Checkbox(label="(Debug toggle – no enhancement applied)", value=False)
        batch_btn = gr.Button("Analyze All Photos", variant="primary")
        batch_output = gr.JSON(label="Batch Results")
        batch_btn.click(analyze_batch, inputs=[batch_input, batch_enhance], outputs=batch_output)

    with gr.Tab("Info"):
        gr.Markdown("""
### What this does
- **Overall quality**: LAION Aesthetics V2 (1–10)
- **Composition**: CLIP similarity with semantic prompts (good vs poor)
- **Portrait rules**: face count, coverage, positioning
- **Clarity**: face-only sharpness (Laplacian/gradient)
- **Emotion**: FER (happy/engagement bonus) if available
- **Lighting**: exposure/contrast/clipping heuristics

### Final Score Weights (Yearbook-Optimized)
- LAION aesthetics **30%**
- CLIP composition **20%**
- Face coverage **12%**
- Face count **8%**
- Face sharpness **15%**
- Emotion **8%**
- Face positioning **4%**
- Lighting **3%**

> Recommendation thresholds:
- **use**: final ≥ 8.2, faces present, sharpness ≥ 6.5  
- **enhance**: final ≥ 7.3  
- **maybe**: final ≥ 6.0  
- **skip**: otherwise
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)