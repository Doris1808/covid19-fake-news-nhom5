"""Thí nghiệm zero-shot: so sánh mô hình NLI, cách đặt tên nhãn, template, và chỉnh ngưỡng.

Chạy trên Colab/Kaggle có GPU:  python src/zeroshot_experiments.py
Quy trình: chọn cấu hình bằng 500 mẫu của tập VAL, rồi mới đánh giá trên TEST.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline

from data import load_data, clean_for_transformer

os.makedirs("results", exist_ok=True)
_, va, te = load_data()
va_s = va.sample(500, random_state=0).reset_index(drop=True)

MODELS = ["facebook/bart-large-mnli", "valhalla/distilbart-mnli-12-3"]
# Mỗi bộ nhãn là [nhãn real, nhãn fake]. Phần tử thứ 2 luôn là nhãn fake.
LABELSETS = {
    "real_fake": ["real news", "fake news"],
    "true_false": ["true information", "false information"],
    "reliable_misinfo": ["reliable information", "misinformation"],
}
TEMPLATES = ["This text is {}.", "This is {}.", "The following post is {}."]
DEV = 0 if torch.cuda.is_available() else -1


def fake_scores(clf, texts, labels, template):
    """Điểm score_fake = softmax qua các nhãn của logit entailment (multi_label=False)."""
    out = clf([clean_for_transformer(t) for t in texts], candidate_labels=labels,
              hypothesis_template=template, batch_size=16)
    return np.array([dict(zip(o["labels"], o["scores"]))[labels[1]] for o in out])


def metrics(y, s, tau=0.5):
    p = (s >= tau).astype(int)
    return accuracy_score(y, p), f1_score(y, p, average="weighted")


# 1) Lưới: mô hình x bộ nhãn x template, đánh giá trên val (500 mẫu)
rows = []
for m in MODELS:
    clf = pipeline("zero-shot-classification", model=m, device=DEV)
    for ln, labels in LABELSETS.items():
        for tpl in TEMPLATES:
            s = fake_scores(clf, va_s.text, labels, tpl)
            acc, f1 = metrics(va_s.label.values, s)
            rows.append(dict(model=m, labelset=ln, template=tpl, val_acc=acc, val_f1w=f1))
            print(rows[-1], flush=True)
    del clf
grid = pd.DataFrame(rows).sort_values("val_f1w", ascending=False)
grid.to_csv("results/zeroshot_grid.csv", index=False)
print("\n=== Xếp hạng cấu hình zero-shot (val, 500 mẫu) ===")
print(grid.round(4).to_string(index=False))

# 2) Lấy cấu hình tốt nhất, chỉnh ngưỡng tau trên val, rồi đánh giá trên test
best = grid.iloc[0]
labels = LABELSETS[best.labelset]
clf = pipeline("zero-shot-classification", model=best.model, device=DEV)
s_val = fake_scores(clf, va_s.text, labels, best.template)
taus = np.linspace(0.05, 0.95, 19)
tau_best = max(taus, key=lambda t: metrics(va_s.label.values, s_val, t)[1])

s_test = fake_scores(clf, te.text, labels, best.template)
acc05, f105 = metrics(te.label.values, s_test, 0.5)
acc_t, f1_t = metrics(te.label.values, s_test, tau_best)
res = {
    "model": best.model, "labelset": best.labelset, "template": best.template,
    "test_acc_tau0.5": acc05, "test_f1w_tau0.5": f105,
    "tau_tuned_on_val": float(tau_best), "test_acc_tuned": acc_t, "test_f1w_tuned": f1_t,
}
print("\n=== Kết quả test của cấu hình zero-shot tốt nhất ===")
print(json.dumps(res, indent=2, ensure_ascii=False))
json.dump(res, open("results/zeroshot_best.json", "w"), indent=2, ensure_ascii=False)
np.save("results/zeroshot_test_scores.npy", s_test)
