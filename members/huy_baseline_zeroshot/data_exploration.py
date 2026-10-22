"""Khảo sát dữ liệu trước khi tiền xử lý. Chạy:  python -m members.huy_baseline_zeroshot.data_exploration

In ra: giá trị thiếu, bài trùng lặp (trong từng tập và giữa các tập), đặc điểm văn bản
(link, mention, hashtag, emoji, độ dài...) và tỉ lệ của chúng theo nhãn real/fake.
Lưu kết quả vào thư mục outputs/baseline_zeroshot/.
"""
import os
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_data, RESULTS_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
tr, va, te = load_data()
tr["split"], va["split"], te["split"] = "train", "val", "test"
al = pd.concat([tr, va, te], ignore_index=True)

URL = re.compile(r"https?://\S+|www\.\S+")


def norm(s):
    """Chuẩn hóa để phát hiện bài trùng: chữ thường, bỏ link, chỉ giữ chữ và số."""
    s = URL.sub(" ", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# 1. Chất lượng cơ bản 
print("1. Chất lượng cơ bản")
print("Số mẫu mỗi tập:", al.split.value_counts().to_dict())
print("Giá trị thiếu (text):", int(al.text.isna().sum()))
print("Bài rỗng sau khi strip:", int((al.text.fillna("").str.strip() == "").sum()))
print("Phân bố nhãn theo tập (tỉ lệ fake):")
print(al.groupby("split").label.agg(["count", "mean"]).round(4))

# 2. Trùng lặp và rò rỉ 
print("\n 2. Trùng lặp và rò rỉ giữa các tập")
al["norm"] = al.text.fillna("").map(norm)
for n in ("train", "val", "test"):
    d = al[al.split == n]
    print(f"Bài trùng hệt trong {n}: {int(d.text.duplicated().sum())};  trùng sau chuẩn hóa: {int(d.norm.duplicated().sum())}")
sets = {n: set(al[al.split == n].norm) for n in ("train", "val", "test")}
print("Giao nhau sau chuẩn hóa: traingiaoval =", len(sets["train"] & sets["val"]),
      "| traingiaotest =", len(sets["train"] & sets["test"]), "| valgiaotest =", len(sets["val"] & sets["test"]))
conf = al.groupby("norm").label.nunique()
print("Nội dung trùng nhưng khác nhãn:", int((conf > 1).sum()))

# 3. Đặc điểm văn bản 
t = al.text.fillna("")
al["n_chars"] = t.str.len()
al["n_words"] = t.str.split().str.len()
al["n_url"] = t.str.count(r"https?://\S+")
al["n_mention"] = t.str.count(r"@\w+")
al["n_hashtag"] = t.str.count(r"#\w+")
letters = t.str.count(r"[A-Za-z]").clip(lower=1)
al["upper_ratio"] = t.str.count(r"[A-Z]") / letters
FEATS = {
    "có link": al.n_url > 0,
    "có @mention": al.n_mention > 0,
    "có hashtag": al.n_hashtag > 0,
    "bắt đầu bằng RT": t.str.startswith("RT "),
    "có dấu !": t.str.contains("!", regex=False),
    "có dấu ?": t.str.contains("?", regex=False),
    "có ký tự non-ASCII (emoji, ngoặc cong...)": t.map(lambda s: any(ord(c) > 127 for c in s)),
    "có dấu … hoặc ... (bị cắt cụt?)": t.str.contains("…|\\.\\.\\.", regex=True),
    "có thực thể HTML (&amp; ...)": t.str.contains(r"&(?:amp|lt|gt|quot|#\d+);", regex=True),
    "có xuống dòng": t.str.contains("\n", regex=False),
    "có khoảng trắng thừa đầu/cuối hoặc kép": (t != t.str.strip()) | t.str.contains("  ", regex=False),
    "có từ viết hoa toàn bộ (>=3 chữ)": t.str.contains(r"\b[A-Z]{3,}\b", regex=True),
    "có số": t.str.contains(r"\d", regex=True),
}
rows = []
for name, mask in FEATS.items():
    rows.append({"đặc điểm": name, "tỉ lệ chung": mask.mean(),
                 "tỉ lệ trong real": mask[al.label == 0].mean(), "tỉ lệ trong fake": mask[al.label == 1].mean()})
ft = pd.DataFrame(rows)
ft["chênh (fake - real)"] = ft["tỉ lệ trong fake"] - ft["tỉ lệ trong real"]
print("\n3. Đặc điểm văn bản (tỉ lệ bài có đặc điểm đó)")
print(ft.round(3).to_string(index=False))
ft.to_csv(str(RESULTS_DIR / "eda_features_by_label.csv"), index=False)

print("\n 4. Độ dài theo nhãn (số từ) ")
ln = al.groupby("label").n_words.describe(percentiles=[0.5, 0.95]).round(1)
ln.index = ["real", "fake"]
print(ln)
ln.to_csv(str(RESULTS_DIR / "eda_length_by_label.csv"))
print("Số link trung bình: real =", round(al[al.label == 0].n_url.mean(), 3),
      "| fake =", round(al[al.label == 1].n_url.mean(), 3))

# 5. Hình 
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for lab, name, col in ((0, "real", "#1d9e75"), (1, "fake", "#e07a3f")):
    ax[0].hist(al[al.label == lab].n_words.clip(upper=80), bins=40, alpha=0.6, label=name, color=col)
ax[0].set_xlabel("Số từ (cắt ở 80)")
ax[0].set_ylabel("Số bài")
ax[0].legend()
ft2 = ft.sort_values("chênh (fake - real)")
ax[1].barh(ft2["đặc điểm"].str.slice(0, 28), ft2["chênh (fake - real)"],
           color=["#e07a3f" if v > 0 else "#1d9e75" for v in ft2["chênh (fake - real)"]])
ax[1].set_xlabel("Chênh tỉ lệ (fake − real)")
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_overview.png"), dpi=150)
print("\nĐã lưu outputs/baseline_zeroshot/eda_features_by_label.csv, eda_length_by_label.csv, eda_overview.png")
