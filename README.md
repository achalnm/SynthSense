# AI Image Detector

A plug-and-play pipeline for detecting AI-generated images. Bring your own dataset, run `train.py`, and get a production-ready ensemble classifier with a Gradio web interface.

No pretrained weights are included, and that's intentional. Different AI generators (StyleGAN, Midjourney, DALL-E, Stable Diffusion) leave different statistical fingerprints. Training on your target distribution consistently outperforms generic detectors.

---

## How it works

Features are extracted from three complementary sources and fused into a single vector:

| Source | What it captures | Dims |
|---|---|---|
| CLIP ViT-L/14 (last 3 hidden layers) | Semantic inconsistencies, unnatural textures | 3072 |
| DINOv2 ViT-L/14 (CLS + patch mean) | Structural and spatial artifacts | 2048 |
| Forensic signals (NPR + FFT + ELA + PRNU) | Compression artifacts, frequency anomalies, sensor noise | 92 |

Combined 5212-dim vector -> IncrementalPCA -> 256-dim -> stacking ensemble:
- **LogisticRegression** (saga, balanced class weights)
- **5-seed MLP** (4-layer residual network, focal loss, cosine LR, early stopping)
- **Meta-learner** (LogisticRegression on OOF predictions)

Training uses 5-fold stratified cross-validation with out-of-fold stacking. The decision threshold is tuned on OOF F1.

---

## Dataset format

```
data/
├── train/
│   ├── real/    <- authentic images (.jpg / .jpeg / .png / .webp)
│   └── fake/    <- AI-generated images
└── test/
    ├── real/
    └── fake/
```

Any binary real-vs-fake image dataset works. The pipeline was originally built and tested on the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset (Flickr faces vs StyleGAN).

---

## Setup

```bash
pip install -r requirements.txt
```

Point `DATA_DIR` in `config.py` at your dataset root.

---

## Training

```bash
python train.py
```

Features are extracted and cached to `cache/` so interrupted runs resume from the last completed phase. Trained models are saved to `saved_models/`.

**Quick test mode** (~30 min on GPU): set `QUICK_TEST = True` in `config.py` to train on 5k images per class.

---

## Inference

```bash
python app.py
```

Opens a Gradio web UI at `http://localhost:7860`. Upload any image to get a verdict with confidence score and model internals.

Requires trained models in `saved_models/`. Run `train.py` first.

---

## Hardware

Tested on an NVIDIA RTX 3050 (4 GB VRAM). The pipeline uses float16 on CUDA to fit within 4 GB. CPU and Apple MPS are supported but significantly slower for feature extraction.

Full training on 100k images takes approximately 2–3 hours on a mid-range GPU.

---

## Project structure

```
├── config.py        - paths, device, hyperparameters
├── data_loader.py   - dataset scanning and split loading
├── features.py      - CLIP, DINOv2, and forensic feature extraction + caching
├── models.py        - MLP architecture and training routines
├── train.py         - end-to-end training pipeline
├── predict.py       - single-image inference
└── app.py           - Gradio web interface
```
