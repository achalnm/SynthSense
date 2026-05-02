import gc
import json
import warnings
import numpy as np
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, classification_report,
    roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss,
)
from scipy.stats import chi2 as _chi2

warnings.filterwarnings("ignore")

from config import DEVICE, MODELS_DIR, N_FOLDS, SEED, MLP_SEEDS
from data_loader import load_dataset
from features import get_features
from models import MLP, train_mlp, train_final_mlp


def main():
    print("=" * 60)
    print("CLIP + DINOv2 + forensics  →  LR + MLP×5 stacking ensemble")
    print("=" * 60)
    _lbl = torch.cuda.get_device_name(0) if DEVICE == "cuda" else ("Apple MPS" if DEVICE == "mps" else "CPU")
    print(f"Device : {DEVICE}  ({_lbl})")

    print("\n[1] Loading dataset...")
    train_df, test_df = load_dataset()
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    print("\n[2] Extracting features...")
    X_tr, X_te = get_features(train_df["path"].tolist(), test_df["path"].tolist())

    y_all  = train_df["label"].values
    y_test = test_df["label"].values

    print("\n[3] IncrementalPCA (→ 256 dims)...")
    pca = IncrementalPCA(n_components=256, batch_size=2000)
    pca.fit(X_tr)
    X_tr = pca.transform(X_tr).astype(np.float32)
    X_te = pca.transform(X_te).astype(np.float32)
    gc.collect()
    in_dim = X_tr.shape[1]
    print(f"  Train {X_tr.shape}  Test {X_te.shape}")

    print(f"\n[4] {N_FOLDS}-fold CV...")
    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr_oof  = np.zeros(len(train_df));  lr_test  = np.zeros(len(test_df))
    mlp_oof = np.zeros(len(train_df));  mlp_test = np.zeros(len(test_df))
    fold_f1s       = {"LR": [], "MLP": []}
    fold_train_f1s = {"LR": []}

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_all)):
        print(f"\n-- Fold {fold+1}/{N_FOLDS} --")
        X_f_tr, X_f_vl = X_tr[trn_idx], X_tr[val_idx]
        y_f_tr, y_f_vl = y_all[trn_idx], y_all[val_idx]

        sc       = StandardScaler()
        X_f_tr_s = sc.fit_transform(X_f_tr).astype(np.float32)
        X_f_vl_s = sc.transform(X_f_vl).astype(np.float32)
        X_f_te_s = sc.transform(X_te).astype(np.float32)

        lreg = LogisticRegression(C=0.0001, max_iter=2000, random_state=SEED,
                                  class_weight="balanced", solver="saga", n_jobs=1)
        lreg.fit(X_f_tr_s, y_f_tr)
        lr_oof[val_idx] = lreg.predict_proba(X_f_vl_s)[:, 1]
        lr_test        += lreg.predict_proba(X_f_te_s)[:, 1] / N_FOLDS
        _lr_val_f1   = f1_score(y_f_vl, (lr_oof[val_idx] > 0.5).astype(int))
        _lr_train_f1 = f1_score(y_f_tr, (lreg.predict_proba(X_f_tr_s)[:, 1] > 0.5).astype(int))
        fold_f1s["LR"].append(_lr_val_f1)
        fold_train_f1s["LR"].append(_lr_train_f1)
        print(f"  LR  F1: {_lr_val_f1:.4f}  (train {_lr_train_f1:.4f})")

        fold_vl = np.zeros(len(val_idx))
        fold_te = np.zeros(len(test_df))
        for seed in MLP_SEEDS:
            vl_p, te_p = train_mlp(X_f_tr_s, y_f_tr, X_f_vl_s, y_f_vl,
                                   X_f_te_s, in_dim, seed=seed)
            fold_vl += vl_p / len(MLP_SEEDS)
            fold_te += te_p / len(MLP_SEEDS)
        mlp_oof[val_idx] = fold_vl
        mlp_test        += fold_te / N_FOLDS
        print(f"  MLP F1: {f1_score(y_f_vl, (mlp_oof[val_idx] > 0.5).astype(int)):.4f}")
        fold_f1s["MLP"].append(f1_score(y_f_vl, (mlp_oof[val_idx] > 0.5).astype(int)))

        del X_f_tr, X_f_vl, X_f_tr_s, X_f_vl_s, X_f_te_s
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n[5] Stacking meta-learner...")
    oof_stack  = np.column_stack([lr_oof,  mlp_oof])
    test_stack = np.column_stack([lr_test, mlp_test])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(oof_stack, y_all)
    ens_oof     = meta.predict_proba(oof_stack)[:, 1]
    final_probs = meta.predict_proba(test_stack)[:, 1]
    print(f"  Meta weights — LR: {meta.coef_[0][0]:.3f}  MLP: {meta.coef_[0][1]:.3f}")

    thresholds  = np.linspace(0.2, 0.8, 121)
    oof_f1s     = [f1_score(y_all, (ens_oof > t).astype(int)) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(oof_f1s))])
    best_f1     = float(max(oof_f1s))
    oof_preds   = (ens_oof > best_thresh).astype(int)

    print(f"\n[6] OOF evaluation (threshold={best_thresh:.3f})...")
    print(f"  F1: {best_f1:.4f}  ROC-AUC: {roc_auc_score(y_all, ens_oof):.4f}"
          f"  PR-AUC: {average_precision_score(y_all, ens_oof):.4f}"
          f"  Brier: {brier_score_loss(y_all, ens_oof):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_all, oof_preds).ravel()
    print(f"  FPR: {fp/(fp+tn):.4f}  FNR: {fn/(fn+tp):.4f}")

    for clf in ["LR"]:
        tr_m = np.mean(fold_train_f1s[clf])
        vl_m = np.mean(fold_f1s[clf])
        gap  = tr_m - vl_m
        flag = "  ← overfit" if gap > 0.05 else ""
        print(f"  {clf} overfitting: train {tr_m:.4f}  OOF {vl_m:.4f}  gap {gap:+.4f}{flag}")

    print(classification_report(y_all, oof_preds, target_names=["Authentic", "AI-Generated"]))

    # McNemar's test
    lr_preds = (lr_oof > 0.5).astype(int)
    b = int(np.sum((lr_preds == y_all) & ~(oof_preds == y_all)))
    c = int(np.sum(~(lr_preds == y_all) & (oof_preds == y_all)))
    if (b + c) > 0:
        stat = (b - c) ** 2 / (b + c)
        p    = 1.0 - _chi2.cdf(stat, df=1)
        print(f"  McNemar (LR vs Ensemble): chi2={stat:.4f}  p={p:.4f}"
              + ("  ← ensemble significantly better" if p <= 0.05 else ""))

    print("\n[7] Test set evaluation...")
    final_preds = (final_probs > best_thresh).astype(int)
    test_f1     = f1_score(y_test, final_preds)
    print(f"  F1: {test_f1:.4f}  ROC-AUC: {roc_auc_score(y_test, final_probs):.4f}")
    print(classification_report(y_test, final_preds, target_names=["Authentic", "AI-Generated"]))

    print("\n[8] Training final models on all data...")
    sc_final = StandardScaler()
    X_tr_s   = sc_final.fit_transform(X_tr).astype(np.float32)

    lr_final = LogisticRegression(C=0.0001, max_iter=2000, random_state=SEED,
                                  class_weight="balanced", solver="saga", n_jobs=1)
    lr_final.fit(X_tr_s, y_all)

    mlp_finals = []
    for seed in MLP_SEEDS:
        print(f"  MLP seed={seed}...")
        m = train_final_mlp(X_tr_s, y_all.astype(np.float32), in_dim, seed=seed)
        mlp_finals.append((seed, m))

    joblib.dump(pca,      MODELS_DIR / "pca.joblib")
    joblib.dump(sc_final, MODELS_DIR / "scaler.joblib")
    joblib.dump(lr_final, MODELS_DIR / "lr.joblib")
    joblib.dump(meta,     MODELS_DIR / "meta.joblib")
    for seed, m in mlp_finals:
        torch.save(m.state_dict(), MODELS_DIR / f"mlp_{seed}.pt")

    cfg = {"threshold": best_thresh, "in_dim": in_dim,
           "oof_f1": best_f1, "test_f1": test_f1}
    with open(MODELS_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n  Saved to {MODELS_DIR}")
    print("=" * 60)
    print(f"  OOF F1: {best_f1:.4f}  |  Test F1: {test_f1:.4f}")
    print("  Run  python app.py  to start the web interface.")
    print("=" * 60)


if __name__ == "__main__":
    main()
