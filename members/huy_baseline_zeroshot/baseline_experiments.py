"""Thí nghiệm baseline: nhiều mô hình x nhiều cấu hình TF-IDF x nhiều tham số.

Chạy từ thư mục gốc dự án:  python -m members.huy_baseline_zeroshot.baseline_experiments
Quy trình: chọn tham số theo tập VAL, rồi mới báo cáo kết quả TEST của cấu hình đã chọn.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .data import load_data, RESULTS_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
tr, va, te = load_data()

STAT_NAMES = ["stat_n_words", "stat_n_exclaim", "stat_n_link"]

# Cấu hình vector hóa TF-IDF để so sánh
VEC_CFGS = {
    "uni_min2": dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True),
    "bi_min2": dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    "bi_min2_nosub": dict(ngram_range=(1, 2), min_df=2, sublinear_tf=False),
}

# (tên, hàm tạo mô hình, lưới tham số, danh sách seed)
# Mô hình có tính ngẫu nhiên (cây quyết định) chạy 3 seed để có mean +- std
MODELS = [
    ("NaiveBayes", lambda p, s: MultinomialNB(**p), [{"alpha": a} for a in (0.1, 0.5, 1.0)], [0]),
    ("LogReg", lambda p, s: LogisticRegression(max_iter=2000, **p), [{"C": c} for c in (0.1, 1, 10, 100)], [0]),
    ("LinearSVC", lambda p, s: LinearSVC(**p), [{"C": c} for c in (0.1, 0.5, 1, 5)], [0]),
    ("DecisionTree", lambda p, s: DecisionTreeClassifier(random_state=s, **p),
     [{"max_depth": d} for d in (10, 50)], [42, 1, 2]),
]
# Tùy chọn: thêm GradientBoostingClassifier (chậm trên TF-IDF nhiều chiều, nên giới hạn max_features=5000).


def stat_feats(texts):
    """Độ dài (số từ), số dấu '!', số link. Dùng log1p để giảm ảnh hưởng của giá trị rất lớn."""
    f = [[len(t.split()), t.count("!"), t.lower().count("http")] for t in texts]
    return np.log1p(np.array(f, dtype=float))


def build(vcfg, use_stats):
    """Trả về ([Xtr, Xva, Xte], vectorizer)."""
    vec = TfidfVectorizer(**vcfg)
    X = [vec.fit_transform(tr.text), vec.transform(va.text), vec.transform(te.text)]
    if use_stats:
        sc = MinMaxScaler().fit(stat_feats(tr.text))  # fit trên train để không rò rỉ
        S = [np.clip(sc.transform(stat_feats(d.text)), 0, 1) for d in (tr, va, te)]  # clip để NB không bị số âm
        X = [hstack([x, csr_matrix(s)]).tocsr() for x, s in zip(X, S)]
    return X, vec


def run_all():
    rows = []
    for vname, vcfg in VEC_CFGS.items():
        for use_stats in (False, True):
            (Xtr, Xva, Xte), _ = build(vcfg, use_stats)
            for mname, make, grid, seeds in MODELS:
                for params in grid:
                    for seed in seeds:
                        t0 = time.time()
                        clf = make(params, seed).fit(Xtr, tr.label)
                        r = dict(vec=vname, stats=use_stats, model=mname,
                                 params=json.dumps(params), seed=seed,
                                 fit_s=round(time.time() - t0, 2))
                        for split, X, y in (("train", Xtr, tr.label), ("val", Xva, va.label), ("test", Xte, te.label)):
                            p = clf.predict(X)
                            r[f"{split}_acc"] = accuracy_score(y, p)
                            r[f"{split}_f1w"] = f1_score(y, p, average="weighted")
                            if split == "test":
                                r["test_prec_fake"] = precision_score(y, p, pos_label=1)
                                r["test_rec_fake"] = recall_score(y, p, pos_label=1)
                        rows.append(r)
                        print(vname, use_stats, mname, params, seed,
                              f"val_f1w={r['val_f1w']:.4f}", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    keys = ["vec", "stats", "model", "params"]
    agg = df.groupby(keys).agg(
        train_f1w=("train_f1w", "mean"),
        val_f1w=("val_f1w", "mean"),
        test_acc=("test_acc", "mean"),
        test_f1w=("test_f1w", "mean"),
        test_f1w_std=("test_f1w", "std"),
        prec_fake=("test_prec_fake", "mean"),
        rec_fake=("test_rec_fake", "mean"),
        fit_s=("fit_s", "mean"),
    ).reset_index()
    agg["test_f1w_std"] = agg["test_f1w_std"].fillna(0.0)
    agg["overfit_gap"] = agg["train_f1w"] - agg["val_f1w"]  # chênh lệch train - val
    return agg


def plot_c_curve(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = agg[(~agg.stats) & (agg.vec == "bi_min2") & (agg.model.isin(["LogReg", "LinearSVC"]))].copy()
    sub["C"] = sub.params.apply(lambda s: json.loads(s)["C"])
    plt.figure(figsize=(6, 4))
    for m, g in sub.groupby("model"):
        g = g.sort_values("C")
        plt.plot(g.C, g.train_f1w, "--", marker="o", label=f"{m} train")
        plt.plot(g.C, g.val_f1w, marker="o", label=f"{m} val")
    plt.xscale("log")
    plt.xlabel("C (log)")
    plt.ylabel("F1 weighted")
    plt.title("Ảnh hưởng của C: khoảng cách train-val là mức overfit")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "c_curve.png"), dpi=150)


if __name__ == "__main__":
    df = run_all()
    df.to_csv(str(RESULTS_DIR / "baseline_all_runs.csv"), index=False)
    agg = summarize(df)
    agg.to_csv(str(RESULTS_DIR / "baseline_all_configs.csv"), index=False)

    # Mỗi mô hình lấy cấu hình có F1 val cao nhất, rồi mới đọc kết quả test
    best = agg.loc[agg.groupby("model")["val_f1w"].idxmax()].sort_values("val_f1w", ascending=False)
    best.to_csv(str(RESULTS_DIR / "baseline_best_per_model.csv"), index=False)
    print("\n=== Cấu hình tốt nhất của mỗi mô hình (chọn theo val) ===")
    print(best[["model", "vec", "stats", "params", "val_f1w", "test_acc", "test_f1w",
                "test_f1w_std", "prec_fake", "rec_fake", "overfit_gap", "fit_s"]].round(4).to_string(index=False))

    # So sánh có / không có đặc trưng thống kê (yêu cầu nâng cao)
    cmp_ = (agg.loc[agg.groupby(["model", "stats"])["val_f1w"].idxmax()]
            [["model", "stats", "vec", "params", "val_f1w", "test_f1w"]].round(4))
    cmp_.to_csv(str(RESULTS_DIR / "baseline_stats_compare.csv"), index=False)
    print("\n=== Có / không có đặc trưng thống kê ===")
    print(cmp_.to_string(index=False))

    plot_c_curve(agg)
    print("\nĐã lưu: outputs/baseline_zeroshot/baseline_*.csv và outputs/baseline_zeroshot/c_curve.png")
