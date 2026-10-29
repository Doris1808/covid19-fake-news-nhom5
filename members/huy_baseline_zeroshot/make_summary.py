"""Gom kết quả baseline + zero-shot thành bảng và biểu đồ để làm slide.

Chạy sau baseline_experiments.py:  python -m members.huy_baseline_zeroshot.make_summary
Số zero-shot đọc trực tiếp từ JSON kết quả đã lưu.
"""
from .data import RESULTS_DIR
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

best = pd.read_csv(str(RESULTS_DIR / "baseline_best_per_model.csv"))
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

# Chỉ tổng hợp kết quả thực nghiệm đã lưu, không dùng số điền sẵn.
import json

zs_path = RESULTS_DIR / "zeroshot_best.json"
if not zs_path.is_file():
    raise FileNotFoundError(f"Thiếu {zs_path}. Chạy zero-shot hoặc chép JSON kết quả gốc vào đây.")
j = json.loads(zs_path.read_text(encoding="utf-8"))
config = f"{j['labelset']}, {j['template']}"
rows.append(dict(method=f"Zero-shot ({j['model']}, ngưỡng 0.5)",
                 config=config, test_acc=j["test_acc_tau0.5"],
                 test_f1w=j["test_f1w_tau0.5"]))
rows.append(dict(method=f"Zero-shot ({j['model']}, ngưỡng {j['tau_tuned_on_val']})",
                 config=config, test_acc=j["test_acc_tuned"],
                 test_f1w=j["test_f1w_tuned"]))
# Điểm bài báo để trong báo cáo tham khảo riêng, không trộn với Test của nhóm.

df = pd.DataFrame(rows).sort_values("test_f1w", ascending=False).reset_index(drop=True)
df.to_csv(str(RESULTS_DIR / "summary_table.csv"), index=False)
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
plt.savefig(str(RESULTS_DIR / "summary_chart.png"), dpi=150)
print("\nĐã lưu outputs/baseline_zeroshot/summary_table.csv và outputs/baseline_zeroshot/summary_chart.png")
