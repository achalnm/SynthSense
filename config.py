from pathlib import Path
import torch
from PIL import Image

DATA_DIR   = Path("data")        # point this at your dataset root
CACHE_DIR  = Path("cache")
MODELS_DIR = Path("saved_models")

QUICK_TEST    = False
QUICK_TRAIN_N = 5000
QUICK_TEST_N  = 1000

for _d in (CACHE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

CLIP_BATCH = 8
DINO_BATCH = 8

N_FOLDS   = 5
SEED      = 42
MLP_SEEDS = [42, 123, 456, 789, 999]

DINO_SIZE = 224
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]

TTA_TRANSFORMS = [lambda img: img]
