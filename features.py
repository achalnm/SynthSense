import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from scipy import stats as sp_stats
from scipy import signal as sp_signal
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, AutoModel

from config import (
    DEVICE, MODEL_DTYPE, CACHE_DIR,
    CLIP_BATCH, DINO_BATCH,
    TTA_TRANSFORMS, DINO_MEAN, DINO_STD, DINO_SIZE,
)


def _clip_batch(imgs, model, processor):
    inp = processor(images=imgs, return_tensors="pt")
    pv  = inp["pixel_values"].to(DEVICE).to(MODEL_DTYPE)
    out = model.vision_model(pixel_values=pv, output_hidden_states=True)
    # last 3 hidden layers concatenated → 3072-dim
    layers = [out.hidden_states[i][:, 1:, :].mean(dim=1) for i in [-4, -3, -2]]
    return torch.cat(layers, dim=-1).float().cpu().numpy()


def extract_clip_tta(paths, model, processor, batch_size=CLIP_BATCH):
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP"):
        batch = paths[i : i + batch_size]
        with torch.no_grad():
            view_feats = []
            for aug in TTA_TRANSFORMS:
                imgs = []
                for p in batch:
                    try:
                        img = Image.open(p).convert("RGB")
                    except Exception:
                        img = Image.new("RGB", (224, 224), (128, 128, 128))
                    imgs.append(aug(img))
                view_feats.append(_clip_batch(imgs, model, processor))
            all_feats.append(np.mean(view_feats, axis=0))
        if DEVICE == "cuda" and (i // batch_size) % 100 == 0:
            torch.cuda.empty_cache()
    return np.vstack(all_feats)


def clip_features_single(img_pil: Image.Image, model, processor) -> np.ndarray:
    with torch.no_grad():
        view_feats = [_clip_batch([aug(img_pil)], model, processor) for aug in TTA_TRANSFORMS]
    return np.mean(view_feats, axis=0)


def _dino_batch(img_tensors, model):
    _mean = torch.tensor(DINO_MEAN).view(1, 3, 1, 1).to(DEVICE).to(MODEL_DTYPE)
    _std  = torch.tensor(DINO_STD).view(1, 3, 1, 1).to(DEVICE).to(MODEL_DTYPE)
    pixels = torch.stack(img_tensors).to(DEVICE).to(MODEL_DTYPE)
    pixels = (pixels - _mean) / _std
    out = model(pixel_values=pixels)
    cls     = out.last_hidden_state[:, 0, :]
    patches = out.last_hidden_state[:, 1:, :].mean(dim=1)
    return torch.cat([cls, patches], dim=1).float().cpu().numpy()


def extract_dino_tta(paths, model, batch_size=DINO_BATCH):
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="DINOv2"):
        batch = paths[i : i + batch_size]
        with torch.no_grad():
            view_feats = []
            for aug in TTA_TRANSFORMS:
                imgs = []
                for p in batch:
                    try:
                        img = Image.open(p).convert("RGB").resize((DINO_SIZE, DINO_SIZE))
                    except Exception:
                        img = Image.new("RGB", (DINO_SIZE, DINO_SIZE), (128, 128, 128))
                    imgs.append(T.ToTensor()(aug(img)))
                view_feats.append(_dino_batch(imgs, model))
            all_feats.append(np.mean(view_feats, axis=0))
        if DEVICE == "cuda" and (i // batch_size) % 100 == 0:
            torch.cuda.empty_cache()
    return np.vstack(all_feats)


def dino_features_single(img_pil: Image.Image, model) -> np.ndarray:
    img_r = img_pil.resize((DINO_SIZE, DINO_SIZE))
    with torch.no_grad():
        view_feats = [_dino_batch([T.ToTensor()(aug(img_r))], model) for aug in TTA_TRANSFORMS]
    return np.mean(view_feats, axis=0)


def compute_handcrafted(path_or_pil, size=256, fft_bins=64) -> np.ndarray:
    n_feats = 12 + fft_bins + 12 + 4
    try:
        if isinstance(path_or_pil, (str, Path)):
            img_pil = Image.open(path_or_pil).convert("RGB").resize((size, size))
        else:
            img_pil = path_or_pil.convert("RGB").resize((size, size))
    except Exception:
        return np.zeros(n_feats, dtype=np.float32)

    img_np = np.array(img_pil, dtype=np.float32) / 255.0

    t    = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    down = F.interpolate(t, scale_factor=0.5, mode="nearest", recompute_scale_factor=True)
    up   = F.interpolate(down, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
    residual = (t - up).squeeze(0).numpy()
    npr_feats = []
    for c in range(3):
        r = residual[c].ravel()
        npr_feats.extend([float(r.mean()), float(r.std()),
                          float(sp_stats.skew(r)), float(sp_stats.kurtosis(r))])
    npr_feats = np.array(npr_feats, dtype=np.float32)

    gray  = np.array(img_pil.convert("L"), dtype=np.float32)
    fft   = np.fft.fft2(gray)
    mag   = np.log(np.abs(np.fft.fftshift(fft)) + 1e-8)
    cy, cx = size // 2, size // 2
    y, x   = np.ogrid[:size, :size]
    r_map  = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    counts = np.bincount(r_map.ravel(), minlength=size)
    sums   = np.bincount(r_map.ravel(), weights=mag.ravel(), minlength=size)
    fft_feats = (sums / (counts + 1e-8))[:fft_bins].astype(np.float32)

    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    img_recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    ela_diff = np.abs(img_np - img_recomp)
    ela_feats = []
    for c in range(3):
        d = ela_diff[:, :, c].ravel()
        ela_feats.extend([float(d.mean()), float(d.std()),
                          float(np.percentile(d, 90)), float(d.max())])
    ela_feats = np.array(ela_feats, dtype=np.float32)

    gray_norm = gray / 255.0
    denoised  = sp_signal.wiener(gray_norm, mysize=5)
    noise     = (gray_norm - denoised).ravel()
    prnu_feats = np.array([float(noise.mean()), float(noise.std()),
                           float(sp_stats.skew(noise)), float(sp_stats.kurtosis(noise))],
                          dtype=np.float32)

    return np.concatenate([npr_feats, fft_feats, ela_feats, prnu_feats])


def build_handcrafted(paths) -> np.ndarray:
    return np.vstack([compute_handcrafted(p) for p in tqdm(paths, desc="Handcrafted")])


def get_features(train_paths, test_paths, force_recompute=False):
    cache = {
        "clip_tr":  CACHE_DIR / "clip_train.npy",
        "clip_te":  CACHE_DIR / "clip_test.npy",
        "dino_tr":  CACHE_DIR / "dino_train.npy",
        "dino_te":  CACHE_DIR / "dino_test.npy",
        "hand_tr":  CACHE_DIR / "handcrafted_train.npy",
        "hand_te":  CACHE_DIR / "handcrafted_test.npy",
    }

    if not force_recompute and cache["clip_tr"].exists() and cache["clip_te"].exists():
        print("  Loading cached CLIP features...")
        X_clip_tr = np.load(cache["clip_tr"])
        X_clip_te = np.load(cache["clip_te"])
    else:
        print(f"  Extracting CLIP features ({DEVICE.upper()})...")
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=MODEL_DTYPE
        ).to(DEVICE).eval()
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        X_clip_tr = extract_clip_tta(train_paths, clip_model, clip_proc)
        X_clip_te = extract_clip_tta(test_paths,  clip_model, clip_proc)
        np.save(cache["clip_tr"], X_clip_tr)
        np.save(cache["clip_te"], X_clip_te)
        del clip_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print(f"  CLIP: {X_clip_tr.shape}")

    if not force_recompute and cache["dino_tr"].exists() and cache["dino_te"].exists():
        print("  Loading cached DINOv2 features...")
        X_dino_tr = np.load(cache["dino_tr"])
        X_dino_te = np.load(cache["dino_te"])
    else:
        print(f"  Extracting DINOv2 features ({DEVICE.upper()})...")
        dino_model = AutoModel.from_pretrained(
            "facebook/dinov2-large", torch_dtype=MODEL_DTYPE
        ).to(DEVICE).eval()
        X_dino_tr = extract_dino_tta(train_paths, dino_model)
        X_dino_te = extract_dino_tta(test_paths,  dino_model)
        np.save(cache["dino_tr"], X_dino_tr)
        np.save(cache["dino_te"], X_dino_te)
        del dino_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print(f"  DINOv2: {X_dino_tr.shape}")

    if not force_recompute and cache["hand_tr"].exists() and cache["hand_te"].exists():
        print("  Loading cached handcrafted features...")
        X_hand_tr = np.load(cache["hand_tr"])
        X_hand_te = np.load(cache["hand_te"])
    else:
        print("  Computing handcrafted features...")
        X_hand_tr = build_handcrafted(train_paths)
        X_hand_te = build_handcrafted(test_paths)
        np.save(cache["hand_tr"], X_hand_tr)
        np.save(cache["hand_te"], X_hand_te)
    print(f"  Handcrafted: {X_hand_tr.shape}")

    X_tr = np.concatenate([X_clip_tr, X_dino_tr, X_hand_tr], axis=1)
    X_te = np.concatenate([X_clip_te, X_dino_te, X_hand_te], axis=1)
    print(f"  Combined: {X_tr.shape}  (CLIP 3072 + DINOv2 2048 + handcrafted 92)")
    return X_tr, X_te
