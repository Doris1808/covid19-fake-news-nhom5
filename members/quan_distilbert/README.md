# 02 · DistilBERT
**Lương Hoàng Bảo Quân · 24DH111520**

Mô hình `distilbert-base-uncased` được fine-tune trên 6.420 bài. Ba lượt dùng 4 epoch, learning rate 2 × 10⁻⁵, giới hạn 128 token, batch 8 và tích lũy gradient 4. Mỗi lượt chọn checkpoint theo Validation F1 weighted. Thời gian ghi trong báo cáo khoảng 247–251 giây mỗi seed trên một Tesla T4.

| Seed | Validation F1 weighted | Test Accuracy | Test F1 weighted |
|---:|---:|---:|---:|
| 42 | 0,97195 | 0,96963 | 0,96962 |
| 123 | **0,97522** | 0,97290 | 0,97289 |
| 2024 | 0,97150 | 0,97243 | 0,97244 |

**Test F1 weighted trung bình bằng 0,97165 ± 0,00177** (độ lệch chuẩn mẫu). Seed 123 được chọn theo Validation. Phiên Kaggle có trọng số đã gắn và output từng ô của seed 42. Trong Test của seed 42 có 39 fake → real và 26 real → fake.

[Notebook để trình bày](02_DistilBERT.ipynb) · [Mở Colab](https://colab.research.google.com/github/Doris1808/covid19-fake-news-nhom5/blob/main/members/quan_distilbert/02_DistilBERT.ipynb) · [Số liệu CSV](../../reports/distilbert/metrics_3_seeds.csv) · [Phiên Kaggle có output gốc](minh_chung/Phien_Kaggle_seed42.ipynb)

**Chạy lại.** `MODE = "xem"` đọc JSON và CSV bằng CPU. Chọn `du_doan` cùng `MODEL_DIR` để thử 10 câu bằng đúng checkpoint, hoặc chọn `train` để tạo checkpoint mới của seed đã chọn. Notebook kiểm tra seed, cấu hình, dấu vân tay dữ liệu trước khi dùng checkpoint có sẵn. Trong các tệp nộp có hai ví dụ ngoài kèm nguồn cho seed 123. Muốn bổ sung bảng 10 câu, chạy ô cuối và lưu CSV mới.

[Cấu hình](../../configs/distilbert.json) · [Hướng dẫn chạy trên Colab/Kaggle](../../docs/HUONG_DAN_CHAY.md) · [Hồ sơ gốc theo seed](../../reports/distilbert)
