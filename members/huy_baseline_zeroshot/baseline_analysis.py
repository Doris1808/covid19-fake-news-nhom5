"""Phân tích sâu mô hình baseline tốt nhất.

Sửa phần CẤU HÌNH theo dòng tốt nhất trong outputs/baseline_zeroshot/baseline_best_per_model.csv, rồi chạy:
    python -m members.huy_baseline_zeroshot.baseline_analysis
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC

from .data import RESULTS_DIR
from .baseline_experiments import build, VEC_CFGS, STAT_NAMES, te, tr

# ===== CẤU HÌNH (sửa cho khớp kết quả tốt nhất của bạn) =====
VEC_NAME = "bi_min2_nosub"
USE_STATS = True
MODEL = "LinearSVC"
C = 1
# =============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(Xtr, Xva, Xte), vec = build(VEC_CFGS[VEC_NAME], USE_STATS)
clf = (LogisticRegression(C=C, max_iter=2000) if MODEL == "LogReg" else LinearSVC(C=C)).fit(Xtr, tr.label)
pred = clf.predict(Xte)
y = te.label.values

# 1) Báo cáo và confusion matrix
print(classification_report(y, pred, target_names=["real", "fake"], digits=4))
cm = confusion_matrix(y, pred)
tn, fp, fn, tp = cm.ravel()
print(f"TP={tp}  FN={fn} (tin giả bị bỏ sót)  FP={fp} (tin thật bị báo oan)  TN={tn}")
ConfusionMatrixDisplay(cm, display_labels=["real", "fake"]).plot(cmap="Blues", values_format="d")
import matplotlib.pyplot as plt
plt.title(f"{MODEL} C={C} ({VEC_NAME}, stats={USE_STATS})")
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "confusion_matrix.png"), dpi=150)

# 2) Từ có trọng số lớn nhất (chỉ ý nghĩa với mô hình tuyến tính)
names = list(vec.get_feature_names_out()) + (STAT_NAMES if USE_STATS else [])
w = clf.coef_[0]
order = np.argsort(w)
print("\nTừ đẩy về REAL:", [names[i] for i in order[:20]])
print("Từ đẩy về FAKE:", [names[i] for i in order[::-1][:20]])
pd.DataFrame({"feature": names, "weight": w}).sort_values("weight").to_csv(str(RESULTS_DIR / "feature_weights.csv"), index=False)

# 3) Phân tích lỗi: tỉ lệ sai theo độ dài, theo việc có link, theo dấu '!'
d = te.copy()
d["pred"] = pred
d["wrong"] = d.pred != d.label
d["n_words"] = d.text.str.split().str.len()
d["has_link"] = d.text.str.lower().str.contains("http")
d["has_exclaim"] = d.text.str.contains("!")
d["len_bucket"] = pd.cut(d.n_words, [0, 10, 20, 40, 80, 100000], labels=["<=10", "11-20", "21-40", "41-80", ">80"])
print("\nTỉ lệ sai theo độ dài (số mẫu, tỉ lệ sai):")
print(d.groupby("len_bucket", observed=True).wrong.agg(["count", "mean"]).round(4))
print("\nTỉ lệ sai theo có link:")
print(d.groupby("has_link").wrong.agg(["count", "mean"]).round(4))
print("\nTỉ lệ sai theo có dấu '!':")
print(d.groupby("has_exclaim").wrong.agg(["count", "mean"]).round(4))

err = d[d.wrong].copy()
err["error_type"] = np.where(err.label == 1, "FN (fake bị đoán real)", "FP (real bị đoán fake)")
err = err[[c for c in ["id", "error_type", "label", "pred", "n_words", "has_link", "has_exclaim", "text"] if c in err.columns]]
err.to_csv(str(RESULTS_DIR / "baseline_errors.csv"), index=False)
print(f"\nĐã lưu {len(err)} mẫu sai vào outputs/baseline_zeroshot/baseline_errors.csv. "
      "Hãy đọc tay khoảng 50 mẫu, gom thành các nhóm lỗi và đếm số mẫu mỗi nhóm.")
