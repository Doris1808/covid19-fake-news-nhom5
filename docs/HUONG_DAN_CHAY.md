# Cách mở và chạy lại bài nhóm 5

Để đọc bài, mở [trang chính](../README.md), chọn notebook rồi cuộn đến các bảng kết quả. Các output đã lưu hiển thị ngay trên GitHub. Gói [ZIP mã nguồn](https://github.com/Doris1808/covid19-fake-news-nhom5/archive/refs/heads/main.zip) chứa cả ba tập dữ liệu tại [data/raw](../data/raw) và [10 câu minh họa](../data/external_10_sentences.csv).

| Phần | Môi trường và dữ liệu đầu vào | Khi chạy lại |
|---|---|---|
| [Baseline + zero-shot](https://colab.research.google.com/github/Doris1808/covid19-fake-news-nhom5/blob/main/members/huy_baseline_zeroshot/01_Baseline_ZeroShot.ipynb) | Colab. Trong ô nạp dữ liệu, chọn cùng lúc `train.csv`, `val.csv`, `test.csv` từ ZIP. | SVM chạy trên CPU. Dùng GPU và Internet cho BART-MNLI ở phần 8. |
| [DistilBERT](https://colab.research.google.com/github/Doris1808/covid19-fake-news-nhom5/blob/main/members/quan_distilbert/02_DistilBERT.ipynb) | Colab, Kaggle hoặc Jupyter. `MODE="xem"` đọc repo và các số liệu có sẵn. | Xem hướng dẫn checkpoint bên dưới nếu muốn suy luận hoặc train. |
| [BERT-base](https://colab.research.google.com/github/Doris1808/covid19-fake-news-nhom5/blob/main/members/ly_bert/03_BERT.ipynb) | Colab GPU. Tạo một thư mục `MyDrive/BERT_FakeNews` cho phiên thực nghiệm, chép ba CSV cùng `external_10_sentences.csv` vào đó. | Kết nối Drive rồi chạy từ trên xuống. Phần train ghi `best_model` của seed được chọn vào thư mục này. |
| [RoBERTa-base](https://colab.research.google.com/github/Doris1808/covid19-fake-news-nhom5/blob/main/members/binh_roberta/04_RoBERTa.ipynb) | Colab GPU. Đặt ba CSV và `external_10_sentences.csv` trực tiếp trong `/content`. | Chạy các ô theo thứ tự. Tệp dự đoán, log và checkpoint mới nằm trong phiên làm việc. |

## DistilBERT trên Colab, Kaggle hoặc máy cá nhân

1. Mở [notebook của Quân](../members/quan_distilbert/02_DistilBERT.ipynb). Để xem số liệu ngoại tuyến, giải nén repo và mở notebook từ bên trong thư mục dự án. Để Colab tự tải code từ GitHub, bật Internet.
2. Giữ `MODE="xem"` rồi chạy toàn bộ. Notebook kiểm tra ba CSV bằng SHA-256, dựng lại bảng ba seed và ma trận nhầm lẫn. Phần này chỉ cần thư viện chuẩn Python.
3. Để suy luận, chuyển `MODE` thành `"du_doan"`, chọn đúng `SEED` và gắn checkpoint đã fine-tune. Ví dụ, ở Kaggle đặt `MODEL_DIR` tới thư mục `model/`, cùng cấp với `run_config.json`; tại Colab có thể trỏ tới thư mục trên Drive đã mount. Trọng số phải đi cùng cấu hình, tokenizer và đúng dữ liệu của seed đã chọn. Bật `INSTALL_DEPS` nếu cần cài thư viện. Ô cuối xuất dự đoán cho 10 câu minh họa.
4. Có thể tự tạo checkpoint với `MODE="train"`. Chọn một GPU, bật Internet để tải DistilBERT pretrained và cài thư viện. Notebook tạo một thư mục mới dưới `distilbert_runs/`, train đủ bốn epoch, đánh giá Test và lưu mô hình. `MODE="smoke"` chỉ chạy thử bằng ít mẫu.

### Demo Streamlit có sẵn

Dùng Python 3.11 trong môi trường riêng. Từ thư mục gốc của repo, chạy:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Nhập đường dẫn tới thư mục `model/` của **DistilBERT** trong thanh bên. Kết quả demo là nhãn dự đoán của mô hình nghiên cứu. Đường dẫn mặc định tương ứng với cách chạy `python -m src.finetune --seed 42` từ terminal.

[Trang bài nộp](../README.md) · [So sánh kết quả](../reports/README.md)
