import glob
import os
import re

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LABEL2ID = {"real": 0, "fake": 1}


def _find(keyword):
    files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
             if keyword in os.path.basename(f).lower()]
    if not files:
        raise FileNotFoundError(f"Không thấy file csv chứa '{keyword}' trong {DATA_DIR}")
    return sorted(files)[0]


def _read(path):
    df = pd.read_csv(path)
    text_col = next((c for c in ("tweet", "post", "text") if c in df.columns), None)
    if text_col is None or "label" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} có các cột {list(df.columns)}; "
                         "cần cột chữ (tweet/post/text) và cột label.")
    df = df.rename(columns={text_col: "text"})
    raw = df["label"].astype(str).str.strip().str.lower()
    df["label"] = raw.map(LABEL2ID)
    if df["label"].isna().any():
        raise ValueError(f"{os.path.basename(path)}: cột label có giá trị {sorted(raw.unique())[:6]}, "
                         "không phải real/fake (file test có thể chưa có nhãn, hoặc nhãn là 0/1).")
    return df[["text", "label"]].reset_index(drop=True)


def load_data():
    """Trả về (train, val, test), mỗi bảng có cột text và label (real=0, fake=1)."""
    return [_read(_find(k)) for k in ("train", "val", "test")]


def clean_for_transformer(t):
    return re.sub(r"http\S+", "URL", t)  # giữ nội dung, thay link bằng token URL


if __name__ == "__main__":
    tr, va, te = load_data()
    print("train/val/test:", len(tr), len(va), len(te))
    print(tr["label"].value_counts())
    print(tr.sample(3, random_state=0))