import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR, QUICK_TEST, QUICK_TRAIN_N, QUICK_TEST_N, SEED

# Expected layout:
#   DATA_DIR/train/real/  and  DATA_DIR/train/fake/
#   DATA_DIR/test/real/   and  DATA_DIR/test/fake/

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def load_split(split_dir: Path, max_per_class: int = None) -> pd.DataFrame:
    rows = []
    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = split_dir / label_name
        if not folder.exists():
            print(f"  WARNING: {folder} not found, skipping.")
            continue
        files = []
        for ext in _IMAGE_EXTS:
            files.extend(folder.glob(ext))
        if max_per_class is not None:
            rng = np.random.default_rng(SEED)
            files = list(rng.choice(files, size=min(max_per_class, len(files)), replace=False))
        for p in files:
            rows.append({"path": str(p), "image_id": p.name, "label": label_val})
    return pd.DataFrame(rows).reset_index(drop=True)


def load_dataset():
    if QUICK_TEST:
        print(f"  [QUICK TEST] {QUICK_TRAIN_N} train + {QUICK_TEST_N} test per class")
        train_df = load_split(DATA_DIR / "train", max_per_class=QUICK_TRAIN_N)
        test_df  = load_split(DATA_DIR / "test",  max_per_class=QUICK_TEST_N)
    else:
        train_df = load_split(DATA_DIR / "train")
        test_df  = load_split(DATA_DIR / "test")
    return train_df, test_df


if __name__ == "__main__":
    from PIL import Image

    train_df, test_df = load_dataset()
    for name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        n_real = (df["label"] == 0).sum()
        n_fake = (df["label"] == 1).sum()
        print(f"{name}:  real={n_real:,}  fake={n_fake:,}  total={len(df):,}")

    print("\nSanity-checking images...")
    for _, row in train_df.sample(min(6, len(train_df)), random_state=0).iterrows():
        try:
            img = Image.open(row["path"]).convert("RGB")
            print(f"  OK  {img.size}  {row['image_id']}")
        except Exception as e:
            print(f"  BAD {row['image_id']} — {e}")
