# XỬ LÝ NGÔN NGỮ TỰ NHIÊN
## PHÁT HIỆN TIN GIẢ VỀ COVID-19 (CONSTRAINT@AAAI2021)

**Nhóm 5 · HUFLIT · Giảng viên: ThS. Trần Thị Thanh Thảo**

Đồ án phân loại bài đăng tiếng Anh về COVID-19 thành `fake` hoặc `real`. Nhóm khảo sát baseline TF-IDF, zero-shot và fine-tune ba mô hình Transformer để so sánh kết quả cùng chi phí tính toán.

### Thành viên

| Sinh viên | MSSV | Phần phụ trách |
|---|---|---|
| Vũ Hồng Huy | 24DH113458 | Bài báo, EDA, baseline A, zero-shot B, tổng hợp số liệu |
| Lương Hoàng Bảo Quân | 24DH111520 | DistilBERT, 3 seed, thời gian/GPU |
| Đặng Thị Trúc Ly | 24DH112737 | BERT-base, 3 seed, phân tích lỗi |
| Lê Nguyễn Ngọc Bình | 24DH113442 | RoBERTa-base, 3 seed, demo |

### Tình trạng mã nguồn

Bản này triển khai trước phần DistilBERT của Quân: chuẩn bị dữ liệu, train một seed, lưu checkpoint, đánh giá và tổng hợp kết quả. Có giao diện Streamlit tối thiểu để gắn checkpoint sau khi train. Các thư mục trong `members/` dành cho phần bổ sung của các thành viên; chưa có kết quả A/B hoặc BERT/RoBERTa trong bản này.

Chưa kèm dữ liệu thô, trọng số hoặc kết quả fine-tune. Có code không đồng nghĩa đã hoàn thành thực nghiệm. Xem [tiến độ](docs/TIEN_DO.md).

### Bắt đầu

1. Mã nguồn của nhóm: https://github.com/Doris1808/covid19-fake-news-nhom5. Xem [hướng dẫn GitHub](docs/GITHUB.md) để tải về hoặc cập nhật.
2. Mở [notebook DistilBERT](notebooks/01_DistilBERT_Kaggle.ipynb) trên Kaggle; link repository đã được điền sẵn.
3. Chạy thử, rồi train lần lượt seed `42`, `123`, `2024`.
4. Tải kết quả về và chạy giao diện trên Mac theo [hướng dẫn](docs/KAGGLE_VA_MAC.md).

### Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `configs/distilbert.json` | Cấu hình chung cho ba seed |
| `notebooks/01_DistilBERT_Kaggle.ipynb` | Các bước chạy trên Kaggle |
| `src/download_data.py`, `src/eda.py` | Tải dữ liệu và thống kê hỗ trợ |
| `src/finetune.py` | Train DistilBERT, chọn checkpoint bằng Validation |
| `src/evaluate.py`, `src/summarize.py` | Đánh giá Test riêng và tổng hợp Validation |
| `src/predict.py`, `app.py` | Nạp checkpoint và giao diện thử nghiệm |
| `members/` | Nơi các bạn đưa phần code riêng lên sau |
| `docs/` | Bài báo, hướng dẫn, phân công, quy trình thực nghiệm |
| `outputs/` | Kết quả tạo sau khi thực thi |
| `reports/` | Báo cáo và minh chứng nhóm bổ sung |

### Chạy bằng terminal

Python 3.11 được đề xuất. Chạy các lệnh tại thư mục chứa README này.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.download_data
python -m src.eda
python -m src.finetune --seed 42 --smoke
python -m src.finetune --seed 42
python -m src.finetune --seed 123
python -m src.finetune --seed 2024
python -m src.summarize
python -m streamlit run app.py
```

Cấu hình mặc định hướng tới một GPU trên Kaggle. Mac bộ nhớ ít nên dùng cấu hình riêng theo hướng dẫn, không mặc định chạy toàn bộ các lệnh train trên Mac.

### Quy tắc thực nghiệm

- Giữ nguyên Train/Validation/Test; không chia lại ngẫu nhiên.
- Nhãn thống nhất: `fake=0`, `real=1`.
- Chọn checkpoint theo **Validation F1 weighted**; ghi thêm Accuracy, F1 macro, F1 fake.
- Không dùng điểm Test để chọn epoch, seed hay siêu tham số.
- Ba seed dùng cùng cấu hình. Chỉ tính mean ± std khi đủ ba seed hợp lệ.
- Lượt `--smoke` dùng ít mẫu để kiểm tra phần mềm, không đưa vào bảng kết quả nghiên cứu.
- Giữ nguồn dữ liệu, SHA256, log và cấu hình cùng kết quả. Không commit token, môi trường ảo hay trọng số lớn.

### Nguồn tham khảo

- Patwa và cộng sự, *Fighting an Infodemic: COVID-19 Fake News Dataset*: https://arxiv.org/abs/2011.03327
- Kho của tác giả: https://github.com/parthpatwa/covid19-fake-news-detection
- Bản sao CSV được code sử dụng: https://huggingface.co/datasets/nanyy1025/covid_fake_news/tree/main
- Sanh và cộng sự, *DistilBERT, a distilled version of BERT*: https://arxiv.org/abs/1910.01108
- API Trainer của phiên bản được chỉ định: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer

Các mô hình và dữ liệu có điều khoản sử dụng riêng từ nguồn phát hành. Dự đoán của hệ thống chỉ phản ánh mô hình phân loại văn bản, không thay thế xác minh thông tin thực tế.
