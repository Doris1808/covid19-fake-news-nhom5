"""Gom kết quả baseline + zero-shot thành bảng và biểu đồ để làm slide.

Chạy sau baseline_experiments.py:  python make_summary.py
Số zero-shot lấy từ kết quả Colab (sửa lại nếu bạn chạy lại và ra số khác).
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

best = pd.read_csv("results/baseline_best_per_model.csv")
names = {
    "LinearSVC": "TF-IDF + SVM",
    "LogReg": "TF-IDF + Hồi quy logistic",
    "NaiveBayes": "TF-IDF + Naive Bayes",
    "DecisionTree": "TF-IDF + Cây quyết định",
}

rows = []
for _, r in best.iterrows():
    rows.append(dict(method=names.get(r.model, r.model),
                     config=f"{r.vec}, stats={r.stats}, {r.params}",
                     test_acc=r.test_acc, test_f1w=r.test_f1w))

# ===== Số zero-shot =====
# Nếu bạn đã tải results/zeroshot_best.json từ Colab về thư mục results thì script tự đọc.
# Nếu chưa có file, dùng số điền tay bên dưới (lấy từ kết quả Colab, sửa nếu bạn chạy lại).
import json
import os

ZS = dict(acc05=0.6986, f105=0.6812, acc_t=0.6972, f1_t=0.6967, tau=0.3)
if os.path.exists("results/zeroshot_best.json"):
    j = json.load(open("results/zeroshot_best.json", encoding="utf-8"))
    ZS = dict(acc05=j["test_acc_tau0.5"], f105=j["test_f1w_tau0.5"],
              acc_t=j["test_acc_tuned"], f1_t=j["test_f1w_tuned"], tau=j["tau_tuned_on_val"])

rows.append(dict(method="Zero-shot (BART-MNLI, ngưỡng 0.5)",
                 config="reliable/misinformation, 'The following post is {}.'",
                 test_acc=ZS["acc05"], test_f1w=ZS["f105"]))
rows.append(dict(method=f"Zero-shot (ngưỡng chỉnh {ZS['tau']})",
                 config="như trên, ngưỡng chọn trên val",
                 test_acc=ZS["acc_t"], test_f1w=ZS["f1_t"]))
# ========================
rows.append(dict(method="Bài báo gốc (SVM)", config="Patwa et al., 2021",
                 test_acc=None, test_f1w=0.9346))

df = pd.DataFrame(rows).sort_values("test_f1w", ascending=False).reset_index(drop=True)
df.to_csv("results/summary_table.csv", index=False)
print(df[["method", "test_acc", "test_f1w"]].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.5))
d = df.sort_values("test_f1w")
colors = ["#888888" if "Bài báo" in m else ("#D85A30" if "Zero-shot" in m else "#1D9E75") for m in d.method]
ax.barh(d.method, d.test_f1w, color=colors)
for y, v in enumerate(d.test_f1w):
    ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1.05)
ax.set_xlabel("F1 weighted trên tập test")
ax.set_title("So sánh baseline và zero-shot")
plt.tight_layout()
plt.savefig("results/summary_chart.png", dpi=150)
print("\nĐã lưu results/summary_table.csv và results/summary_chart.png")
