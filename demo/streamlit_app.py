"""Streamlit presentation for the four HUFLIT NLP notebooks."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import streamlit as st

REPO = "Doris1808/covid19-fake-news-nhom5"
DATA_REV = "f88834da7f9914519e2589b2b4bcc89210b66587"
DEMO_REF = "streamlit-demo"
MODEL_SHA256 = "ba3afb11feaa990ddbe06b8202698a2c6b4161a01bfea705bc70e1430dff8225"
BROWSER = sys.platform == "emscripten"
ROOT = Path(__file__).resolve().parents[1] if not BROWSER else Path("/")
NOTEBOOKS = {
    "Huy · Baseline và zero-shot": "members/huy_baseline_zeroshot/01_Baseline_ZeroShot.ipynb",
    "Quân · DistilBERT": "members/quan_distilbert/02_DistilBERT.ipynb",
    "Ly · BERT-base": "members/ly_bert/03_BERT.ipynb",
    "Bình · RoBERTa-base": "members/binh_roberta/04_RoBERTa.ipynb",
}
DEMO_HEADER = "Phát hiện tin giả COVID-19"
EXAMPLES = {
    "Bản tin thống kê · Test #1": "Our daily update is published. States reported 734k tests 39k new cases and 532 deaths. Current hospitalizations fell below 30k for the first time since June 22. https://t.co/wzSYMe0Sht",
    "Tin đồn về 5G · câu ngoài #7": "5G mobile towers spread the coronavirus, so they must be shut down immediately.",
}
from collections import Counter
from html import unescape
from math import log, sqrt
import re


def preprocess(text):
    text = re.sub(r"https?://\S+", " urltoken ", unescape(text))
    text = re.sub(r"@\w+", " usertoken ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def char_ngrams(text):
    for token in text.split():
        padded = f" {token} "
        for n in range(2, min(5, len(padded)) + 1):
            for start in range(len(padded) - n + 1):
                yield padded[start:start + n]


def weighted_features(counts, features):
    values = []
    for term, frequency in counts.items():
        if term in features:
            idf, *coefficients = features[term]
            values.append(((1 + log(frequency)) * idf, coefficients))
    norm = sqrt(sum(value * value for value, _ in values)) or 1.0
    return [(value / norm, coefs) for value, coefs in values]


def predict_scores(text, model):
    clean = preprocess(text)
    tokens = re.findall(r"(?u)\b\w\w+\b", clean)
    word_counts = Counter(tokens)
    word_counts.update(a + " " + b for a, b in zip(tokens, tokens[1:]))
    word_features = weighted_features(word_counts, model["words"])
    char_features = weighted_features(Counter(char_ngrams(clean)), model["chars"])
    word_score, joint_score = model["intercepts"]
    for value, (baseline, joint) in word_features:
        word_score += value * baseline
        joint_score += value * joint
    joint_score += sum(value * coefs[0] for value, coefs in char_features)
    return word_score, joint_score


def fetch_bytes(path, revision=DATA_REV):
    if not BROWSER:
        return (ROOT / path).read_bytes()
    from js import XMLHttpRequest, Uint8Array
    # Stlite executes Python in a web worker, where synchronous binary XHR is supported.
    urls = [f"https://raw.githubusercontent.com/{REPO}/{revision}/{path}",
            f"https://cdn.jsdelivr.net/gh/{REPO}@{revision}/{path}"]
    for url in urls:
        try:
            req = XMLHttpRequest.new()
            req.open("GET", url, False)
            req.responseType = "arraybuffer"
            req.send()
            if req.status == 200:
                return Uint8Array.new(req.response).to_py().tobytes()
        except Exception:
            continue
    raise OSError(f"Chưa tải được {path}. Kiểm tra kết nối rồi thử lại.")


@st.cache_resource(show_spinner=False)
def fitted_svm():
    data = fetch_bytes("demo/svm.json.gz", DEMO_REF)
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("Tệp mô hình chưa tải đủ hoặc khác phiên bản.")
    return json.loads(gzip.decompress(data))


@st.cache_data(show_spinner=False)
def rows(path):
    return list(csv.DictReader(io.StringIO(fetch_bytes(path).decode("utf-8-sig"))))


def predict_panel():
    names = ["SVM · từ và ký tự", "SVM · chỉ dùng từ"]
    if not BROWSER:
        from deep_models import native_options
        names += native_options()
    model = st.selectbox("Mô hình dự đoán", names)
    def fill_example():
        st.session_state.tweet = EXAMPLES.get(st.session_state.sample, "")
    st.selectbox("Chọn bài mẫu hoặc tự nhập", ["Tự nhập", *EXAMPLES],
                 key="sample", on_change=fill_example)
    tweet = st.text_area("Bài đăng tiếng Anh", key="tweet", height=140, max_chars=6000,
                         placeholder="Dán một bài đăng về COVID-19 hoặc chọn bài mẫu phía trên…")
    if st.button("Dự đoán", type="primary"):
        if not tweet.strip():
            st.warning("Bạn hãy nhập một bài đăng hoặc chọn bài mẫu.")
            return
        with st.spinner("Đang nạp mô hình và phân loại…"):
            if model in names[:2]:
                scores = predict_scores(tweet, fitted_svm())
                selected = scores[1] if model == names[0] else scores[0]
                label = "real" if selected > 0 else "fake"
                result_label(label)
                st.caption(f"Điểm quyết định SVM: {selected:+.3f}. Điểm dương nghiêng về REAL, điểm âm nghiêng về FAKE.")
                with st.expander("Xem văn bản sau tiền xử lý"):
                    st.code(preprocess(tweet), language=None)
            else:
                from deep_models import classify
                label, score = classify(model, tweet)
                result_label(label)
                st.caption(f"Điểm mô hình cho nhãn đã chọn: {score:.1%}.")
        for i, (example, value) in enumerate(EXAMPLES.items()):
            if value == tweet.strip():
                ref = ["REAL", "FAKE"][i]
                st.write(f"Nhãn tham chiếu của nhóm cho mẫu này: **{ref}**.")
    st.caption("Nhãn REAL/FAKE thể hiện cách mô hình phân loại văn bản. Để xác minh một thông tin y tế, cần đối chiếu nguồn.")


def result_label(label):
    if label == "real":
        st.success("Mô hình dự đoán **REAL · Tin thật**")
    else:
        st.warning("Mô hình dự đoán **FAKE · Tin giả**")


def summary_panel():
    st.subheader("So sánh trên 2.140 bài Test")
    metrics = rows("reports/so_sanh_test.csv")
    table = [{"Phương pháp": r["method"], "Số lượt": r["runs"],
              "Accuracy": f"{float(r['accuracy']):.4f}", "F1 macro": f"{float(r['f1_macro']):.4f}",
              "F1 weighted": f"{float(r['f1_weighted']):.4f}" if r["f1_weighted"] else "—"} for r in metrics]
    st.table(table)
    st.caption("Cả sáu phương án dùng cùng tập Test. Điểm Transformer là trung bình qua các seed. Dấu — là chỉ số nhóm chưa báo cáo.")
    st.markdown(f"[Xem bảng gốc và cách tính](https://github.com/{REPO}/blob/{DATA_REV}/reports/README.md)")


def examples_panel():
    sample_set = st.radio("Bộ ví dụ", ["10 câu chung", "2 câu của DistilBERT"], horizontal=True)
    st.caption("Các cột Transformer dưới đây lấy từ kết quả nhóm đã lưu ở seed 123.")
    if sample_set == "2 câu của DistilBERT":
        group = rows("reports/distilbert/external_examples_seed123.csv")
        for row in group:
            with st.container(border=True):
                st.write(row["text"])
                st.markdown(f"Nhãn tham chiếu **{row['reference_label'].upper()}** · DistilBERT **{row['predicted_label'].upper()}**")
        return
    bert = rows("members/ly_bert/results/external_10_results.csv")
    roberta = {r["sentence"]: r for r in rows("members/binh_roberta/results/external_10_sentences_comparison.csv")}
    index = st.selectbox("Chọn câu để đối chiếu", range(len(bert)),
                         format_func=lambda i: f"Câu {bert[i]['id']} · {bert[i]['sentence']}")
    example = bert[index]
    st.write(example["sentence"])
    reference = example["true_label"].split()[0].upper()
    st.markdown(f"Nhãn tham chiếu của nhóm: **{reference}**")
    rob = roberta[example["sentence"]]
    word, combined = predict_scores(example["sentence"], fitted_svm())
    models = [
        ("SVM · chỉ dùng từ", "REAL" if word > 0 else "FAKE", "Tính trực tiếp"),
        ("SVM · từ và ký tự", "REAL" if combined > 0 else "FAKE", "Tính trực tiếp"),
        ("BERT-base", example["pred_label"].split()[0].upper(), "Đã lưu"),
        ("RoBERTa gốc", rob["orig_pred"].upper(), "Đã lưu"),
        ("RoBERTa cải tiến", rob["improved_pred"].upper(), "Đã lưu"),
    ]
    st.table([{"Mô hình": name, "Nhãn dự đoán": lab, "Đối chiếu": "Đúng" if lab == reference else "Sai",
               "Nguồn kết quả": source} for name, lab, source in models])


def source_panel():
    descriptions = [
        "Khảo sát dữ liệu, tiền xử lý, chọn C cho SVM, thử n-gram ký tự và so sánh với BART-MNLI.",
        "Ba seed fine-tune, cách chọn checkpoint bằng Validation và phân tích kết quả Test.",
        "Fine-tune BERT-base, ma trận nhầm lẫn và thử 10 câu ngoài bộ dữ liệu.",
        "Năm seed, kiểm tra trùng dữ liệu, phân tích lỗi và so sánh biến thể cải tiến.",
    ]
    for (title, path), description in zip(NOTEBOOKS.items(), descriptions):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)
            st.markdown(f"[Mở notebook trên GitHub](https://github.com/{REPO}/blob/{DATA_REV}/{path})　·　"
                        f"[Chạy trên Colab](https://colab.research.google.com/github/{REPO}/blob/{DATA_REV}/{path})")


def main():
    st.set_page_config(page_title="Phát hiện tin giả COVID-19 · Nhóm 5", page_icon="🔎", layout="centered")
    st.caption("HUFLIT   /   Xử lý ngôn ngữ tự nhiên   /   Nhóm 5")
    st.title(DEMO_HEADER)
    pages = {"Dự đoán": predict_panel, "Kết quả Test": summary_panel,
             "Câu ví dụ": examples_panel, "Mã nguồn": source_panel}
    choice = st.radio("Nội dung demo", list(pages), horizontal=True, label_visibility="collapsed")
    st.divider()
    try:
        pages[choice]()
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        st.error(f"Không thể hoàn tất thao tác. {error}")


if __name__ == "__main__":
    main()
