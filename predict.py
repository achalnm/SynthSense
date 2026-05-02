import json
import numpy as np
import torch
import joblib
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoModel

from config import DEVICE, MODEL_DTYPE, MODELS_DIR, MLP_SEEDS
from models import MLP
from features import clip_features_single, dino_features_single, compute_handcrafted

_clip_model = None
_clip_proc  = None
_dino_model = None
_infer_pack = None


def _load_vision_models():
    global _clip_model, _clip_proc, _dino_model
    if _clip_model is None:
        print("[predict] Loading CLIP ViT-L/14...")
        _clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=MODEL_DTYPE
        ).to(DEVICE).eval()
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    if _dino_model is None:
        print("[predict] Loading DINOv2 ViT-L/14...")
        _dino_model = AutoModel.from_pretrained(
            "facebook/dinov2-large", torch_dtype=MODEL_DTYPE
        ).to(DEVICE).eval()


def _load_inference_models():
    global _infer_pack
    if _infer_pack is not None:
        return _infer_pack

    cfg_path = MODELS_DIR / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No trained models found at {MODELS_DIR}.\n"
            "Run  python train.py  first."
        )

    with open(cfg_path) as f:
        cfg = json.load(f)

    pca    = joblib.load(MODELS_DIR / "pca.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    lr     = joblib.load(MODELS_DIR / "lr.joblib")
    meta   = joblib.load(MODELS_DIR / "meta.joblib")

    mlps = []
    for seed in MLP_SEEDS:
        m = MLP(cfg["in_dim"]).to(DEVICE)
        m.load_state_dict(torch.load(MODELS_DIR / f"mlp_{seed}.pt", map_location=DEVICE))
        m.eval()
        mlps.append(m)

    _infer_pack = (pca, scaler, lr, mlps, meta, cfg)
    return _infer_pack


def extract_features(img_pil: Image.Image) -> np.ndarray:
    _load_vision_models()
    clip_f = clip_features_single(img_pil, _clip_model, _clip_proc)
    dino_f = dino_features_single(img_pil, _dino_model)
    hand_f = compute_handcrafted(img_pil).reshape(1, -1)
    return np.concatenate([clip_f, dino_f, hand_f], axis=1)


def predict(img_pil: Image.Image) -> dict:
    pca, scaler, lr, mlps, meta, cfg = _load_inference_models()
    threshold = cfg["threshold"]

    feats  = extract_features(img_pil)
    feats  = pca.transform(feats).astype(np.float32)
    scaled = scaler.transform(feats)

    lr_prob = float(lr.predict_proba(scaled)[0, 1])

    mlp_prob = 0.0
    with torch.no_grad():
        xt = torch.FloatTensor(scaled).to(DEVICE)
        for m in mlps:
            mlp_prob += torch.sigmoid(m(xt)).item() / len(mlps)

    final_prob = float(meta.predict_proba(np.array([[lr_prob, mlp_prob]]))[0, 1])
    is_ai = final_prob > threshold

    return {
        "label":       "AI-Generated" if is_ai else "Authentic",
        "probability": final_prob,
        "confidence":  final_prob if is_ai else (1.0 - final_prob),
        "threshold":   threshold,
        "lr_prob":     lr_prob,
        "mlp_prob":    mlp_prob,
    }
