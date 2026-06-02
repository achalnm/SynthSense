# SynthSense

Detects whether an image is real or AI-generated. Instead of shipping fixed weights, SynthSense is built to be trained on the distribution you care about. Different generators (StyleGAN, Midjourney, Stable Diffusion, and others) leave different statistical traces, so a detector trained on the target distribution holds up better than a generic one.

## Approach

SynthSense does not rely on a single signal. It pulls features from three sources that capture different kinds of evidence, then combines them.

Semantic features come from CLIP ViT-L/14, using the last three hidden layers (3072 dimensions). These pick up high-level inconsistencies and unnatural texture that a vision-language model is sensitive to.

Structural features come from DINOv2 ViT-L/14, using the CLS token and the mean of the patch tokens (2048 dimensions). These capture spatial and structural artifacts.

Forensic features are computed directly from the pixels (92 dimensions) and target the low-level traces generators tend to leave: an up- and down-sampling residual (NPR) that exposes local pixel-correlation artifacts, a radially averaged FFT power spectrum for frequency anomalies, Error Level Analysis for compression inconsistencies, and a PRNU estimate for sensor-noise patterns.

The three sources concatenate into a single vector of roughly 5,200 dimensions. That vector is reduced with IncrementalPCA to 256 dimensions, then classified by a stacked ensemble: a logistic regression (saga solver, balanced class weights) and a five-seed MLP, which is a four-layer residual network trained with focal loss, a cosine learning-rate schedule, and early stopping. A logistic-regression meta-learner combines the two using out-of-fold predictions.

## Evaluation

The training pipeline is built around avoiding the usual ways a detector can look better than it really is.

Cross-validation uses 5-fold stratified splits. The ensemble is trained with out-of-fold stacking, so the meta-learner never sees predictions from data it was fit on. That keeps the stacking honest and avoids leakage. The decision threshold is tuned on the out-of-fold F1 rather than left at the default of 0.5.

Beyond accuracy and F1, the pipeline reports precision-recall AUC, a Brier score to check whether the predicted probabilities are calibrated, and a McNemar test comparing the full ensemble against the logistic-regression baseline. The McNemar test checks whether the added complexity is doing statistically meaningful work rather than just adding moving parts. The pipeline also reports the gap between training and validation performance as an overfitting check.

## Dataset format

Point the pipeline at a folder split into train and test, each with real and fake subfolders:

```
data/
  train/
    real/
    fake/
  test/
    real/
    fake/
```

Any binary real-vs-fake image set works (.jpg, .jpeg, .png, .webp). It was originally built and tested on the 140k Real and Fake Faces dataset, which pairs real Flickr faces against StyleGAN-generated faces.

No trained weights are included in the repository. You supply a dataset and train, which is the intended workflow given the point above about target distributions.

## Setup

```
pip install -r requirements.txt
```

Set `DATA_DIR` in `config.py` to your dataset root.

## Training

```
python train.py
```

Extracted features are cached to disk, so an interrupted run resumes from the last completed stage. Trained models are written to `saved_models`. For a fast check, set `QUICK_TEST` in `config.py` to train on a small subset per class.

## Inference

```
python app.py
```

This opens a local Gradio interface. Upload an image to get a verdict with a confidence score, along with the individual contributions of the logistic regression and the MLP, so you can see what each part of the ensemble concluded rather than only the final answer. Trained models need to be present in `saved_models` first.

## Hardware

Feature extraction runs the CLIP and DINOv2 vision transformers, so a GPU helps considerably. The pipeline uses float16 on CUDA to keep memory down. CPU and Apple MPS work but are slower for feature extraction. Running full feature extraction over a large dataset is memory-heavy, which is the main practical constraint on a modest machine.

## Project structure

```
config.py        paths, device, hyperparameters
data_loader.py   dataset scanning and split loading
features.py      CLIP, DINOv2, and forensic feature extraction with caching
models.py        MLP architecture and training routines
train.py         end-to-end training, cross-validation, and evaluation
predict.py       single-image inference
app.py           Gradio interface
```
