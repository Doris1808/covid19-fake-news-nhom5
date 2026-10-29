"""Chạy toàn bộ phần baseline + zero-shot bằng MỘT lệnh.

    python run_all.py                  # chạy đầy đủ (trừ zero-shot, dùng kết quả có sẵn nếu có)
    python run_all.py --quick          # chế độ demo: lưới nhỏ, bỏ thí nghiệm tiền xử lý
    python run_all.py --zeroshot       # chạy cả zero-shot (cần GPU, rất chậm trên CPU)
    python run_all.py --only baseline,analysis,summary
    python run_all.py --skip preprocessing,eda
    python run_all.py --list           # xem danh sách các bước

Mỗi bước chạy bằng một tiến trình riêng, log lưu ở thư mục logs/. Bước lỗi sẽ được ghi nhận
và chương trình chạy tiếp (dùng --stop-on-error để dừng ngay).
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

STEPS = [
    ("data", "data.py", "Kiểm tra dữ liệu (số mẫu, phân bố nhãn)"),
    ("eda", "data_exploration.py", "Khảo sát dữ liệu: trùng lặp, đặc điểm văn bản"),
    ("preprocessing", "preprocessing_experiments.py", "Thí nghiệm các bước tiền xử lý"),
    ("baseline", "baseline_experiments.py", "Baseline: 4 mô hình x tham số, chọn theo val"),
    ("analysis", "baseline_analysis.py", "Confusion matrix và phân tích lỗi"),
    ("zeroshot", "zeroshot_experiments.py", "Zero-shot: 18 cấu hình (cần GPU)"),
    ("summary", "make_summary.py", "Gom bảng và biểu đồ so sánh"),
]
NAMES = [s[0] for s in STEPS]


def ensure_data():
    """Nếu thiếu train/val/test thì thử tải từ Hugging Face."""
    need = ["train.csv", "val.csv", "test.csv"]
    os.makedirs("data", exist_ok=True)
    if all(os.path.exists(os.path.join("data", f)) for f in need):
        return True
    print("Thiếu dữ liệu trong thư mục data/, đang thử tải từ Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        for f in need:
            hf_hub_download("nanyy1025/covid_fake_news", f, repo_type="dataset", local_dir="data")
        return True
    except Exception as e:  # noqa: BLE001
        print("Không tải được dữ liệu:", e)
        print("Hãy tải train.csv, val.csv, test.csv vào thư mục data/ rồi chạy lại.")
        return False


def run_step(key, script, title, env, log_dir):
    print(f"\n{'=' * 78}\n[{key}] {title}\n{'=' * 78}", flush=True)
    t0 = time.time()
    log_path = os.path.join(log_dir, f"{key}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
    return code, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="chế độ demo: lưới nhỏ, bỏ tiền xử lý")
    ap.add_argument("--zeroshot", action="store_true", help="chạy cả zero-shot (cần GPU)")
    ap.add_argument("--only", default="", help="chỉ chạy các bước này, cách nhau bằng dấu phẩy")
    ap.add_argument("--skip", default="", help="bỏ qua các bước này, cách nhau bằng dấu phẩy")
    ap.add_argument("--stop-on-error", action="store_true", help="dừng ngay khi có bước lỗi")
    ap.add_argument("--list", action="store_true", help="in danh sách các bước rồi thoát")
    a = ap.parse_args()

    if a.list:
        for k, s, t in STEPS:
            print(f"{k:14s} {s:32s} {t}")
        return 0

    only = [x.strip() for x in a.only.split(",") if x.strip()]
    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    bad = [x for x in only + list(skip) if x not in NAMES]
    if bad:
        print("Tên bước không hợp lệ:", bad, "| hợp lệ:", NAMES)
        return 2
    if a.quick:
        skip.add("preprocessing")

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg")
    if a.quick:
        env["QUICK"] = "1"
    os.makedirs("results", exist_ok=True)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    if not ensure_data():
        return 1

    report = []
    t_all = time.time()
    for key, script, title in STEPS:
        if (only and key not in only) or key in skip:
            report.append((key, "bỏ qua", 0.0))
            continue
        if key == "zeroshot" and not a.zeroshot and "zeroshot" not in only:
            have = os.path.exists("results/zeroshot_best.json")
            msg = "dùng kết quả có sẵn (results/zeroshot_best.json)" if have else "bỏ qua (chạy trên Colab: --zeroshot)"
            print(f"\n[zeroshot] {msg}")
            report.append((key, "có sẵn" if have else "bỏ qua", 0.0))
            continue
        if key == "zeroshot":
            miss = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
            if miss:
                print(f"\n[zeroshot] thiếu thư viện {miss}. Cài: pip install torch transformers accelerate")
                report.append((key, "LỖI", 0.0))
                if a.stop_on_error:
                    break
                continue
        try:
            code, sec = run_step(key, script, title, env, log_dir)
        except KeyboardInterrupt:
            print("\nĐã dừng theo yêu cầu (Ctrl+C).")
            report.append((key, "dừng", time.time() - t_all))
            break
        report.append((key, "OK" if code == 0 else "LỖI", sec))
        if code != 0 and a.stop_on_error:
            break

    print(f"\n{'=' * 78}\nTÓM TẮT\n{'=' * 78}")
    for key, st, sec in report:
        print(f"{key:14s} {st:10s} {sec:8.1f} s")
    print(f"Tổng thời gian: {time.time() - t_all:.1f} s   (log từng bước: thư mục {log_dir}/)")

    if os.path.exists("results/summary_table.csv"):
        df = pd.read_csv("results/summary_table.csv")
        print("\nBẢNG SO SÁNH CUỐI CÙNG (F1 weighted trên test):")
        print(df[["method", "test_acc", "test_f1w"]].round(4).to_string(index=False))
    return 1 if any(st == "LỖI" for _, st, _ in report) else 0


if __name__ == "__main__":
    sys.exit(main())
