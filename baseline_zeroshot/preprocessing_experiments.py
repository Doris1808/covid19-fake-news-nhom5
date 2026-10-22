"""So sánh các bước tiền xử lý văn bản. Chạy:  python preprocessing_experiments.py

Mỗi biến thể áp dụng MỘT bước (hoặc một tổ hợp) lên văn bản gốc, rồi huấn luyện
TF-IDF (bigram, min_df = 2) + SVM / hồi quy logistic và đo F1 weighted trên val và test.
Kết quả lưu ở results/preprocessing_compare.csv.

Cách đọc: chọn theo val; chênh lệch dưới khoảng 1 điểm F1 chưa đủ căn cứ để nói bước đó có ích.
"""
import html
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from data import load_data

os.makedirs("results", exist_ok=True)
tr, va, te = load_data()
URL = re.compile(r"https?://\S+|www\.\S+")

STEPS = {
    "html": lambda t: html.unescape(t),
    "url_token": lambda t: URL.sub(" URL ", t),
    "url_remove": lambda t: URL.sub(" ", t),
    "rt_remove": lambda t: re.sub(r"^RT\s+@\w+:?\s*", "", t),
    "mention_token": lambda t: re.sub(r"@\w+", " USER ", t),
    "hashtag_split": lambda t: re.sub(r"#(\w+)", lambda m: " " + re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", m.group(1)) + " ", t),
    "non_ascii_remove": lambda t: re.sub(r"[^\x00-\x7F]+", " ", t),
    "squash_ws": lambda t: re.sub(r"\s+", " ", t).strip(),
}
COMBO = ["html", "url_token", "rt_remove", "mention_token", "hashtag_split", "squash_ws"]
VARIANTS = [
    ("0. Gốc (mặc định của TfidfVectorizer)", [], {}),
    ("1. Giữ nguyên chữ hoa/thường", [], {"lowercase": False}),
    ("2. Giải mã thực thể HTML (&amp; ...)", ["html"], {}),
    ("3. Link thay bằng token URL", ["url_token"], {}),
    ("4. Bỏ hẳn link", ["url_remove"], {}),
    ("5. Bỏ tiền tố RT @user:", ["rt_remove"], {}),
    ("6. @mention thay bằng token USER", ["mention_token"], {}),
    ("7. Tách hashtag ghép (#StayAtHome)", ["hashtag_split"], {}),
    ("8. Bỏ ký tự non-ASCII (emoji, ngoặc cong...)", ["non_ascii_remove"], {}),
    ("9. Bỏ stop-word tiếng Anh", [], {"stop_words": "english"}),
    ("10. Tổ hợp bước 2, 3, 5, 6, 7", COMBO, {}),
    ("11. Tổ hợp bước 10 + bỏ stop-word", COMBO, {"stop_words": "english"}),
]
MODELS = {"SVM (C=1)": lambda: LinearSVC(C=1.0),
          "LogReg (C=100)": lambda: LogisticRegression(C=100, max_iter=2000)}


def clean(texts, steps):
    out = []
    for t in texts:
        for s in steps:
            t = STEPS[s](t)
        out.append(t)
    return out


rows = []
for name, steps, vkw in VARIANTS:
    X = [clean(d.text, steps) for d in (tr, va, te)]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, **vkw)
    Xtr = vec.fit_transform(X[0])
    Xva, Xte = vec.transform(X[1]), vec.transform(X[2])
    for mname, make in MODELS.items():
        clf = make().fit(Xtr, tr.label)
        r = {"biến thể": name, "mô hình": mname, "số đặc trưng": Xtr.shape[1]}
        for split, Xs, y in (("val", Xva, va.label), ("test", Xte, te.label)):
            p = clf.predict(Xs)
            r[f"{split}_acc"] = accuracy_score(y, p)
            r[f"{split}_f1w"] = f1_score(y, p, average="weighted")
        rows.append(r)
        print(name, "|", mname, f"val_f1w={r['val_f1w']:.4f}", flush=True)

df = pd.DataFrame(rows)
base = df[df["biến thể"].str.startswith("0.")].set_index("mô hình")
df["Δ val (điểm)"] = [(r.val_f1w - base.loc[r["mô hình"], "val_f1w"]) * 100 for _, r in df.iterrows()]
df["Δ test (điểm)"] = [(r.test_f1w - base.loc[r["mô hình"], "test_f1w"]) * 100 for _, r in df.iterrows()]
df.to_csv("results/preprocessing_compare.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== Ảnh hưởng của từng bước tiền xử lý (so với bản gốc) ===")
print(df[["biến thể", "mô hình", "số đặc trưng", "val_f1w", "test_f1w", "Δ val (điểm)", "Δ test (điểm)"]].round(4).to_string(index=False))
print("\nGhi nhớ: chênh dưới khoảng 1 điểm F1 nằm trong sai số đo lường, chưa đủ để kết luận bước đó có ích.")
