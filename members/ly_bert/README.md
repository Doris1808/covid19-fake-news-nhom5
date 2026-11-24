# 03 · BERT-base
**Đặng Thị Trúc Ly · 24DH112737**

Ly fine-tune `bert-base-uncased` trên ba seed 42, 123 và 2024. Mô hình seed **123** đạt Validation F1 weighted cao nhất (**0,97428**). Qua ba seed, Test F1 weighted là **0,97444 ± 0,00150**. Notebook có kết quả từng seed, ma trận nhầm lẫn, phân tích lỗi Validation và 10 câu ngoài.

[Xem notebook và output](03_BERT.ipynb) · [Mở Colab](https://colab.research.google.com/github/Doris1808/covid19-fake-news-nhom5/blob/main/members/ly_bert/03_BERT.ipynb) · [Điểm từng seed](results/metrics_3_seeds.csv)

| Minh chứng | Nội dung |
|---|---|
| [Các bài Validation dự đoán sai](results/val_errors.csv) | 55 bài của checkpoint được chọn |
| [Dự đoán ngoài Test](results/external_10_results.csv) | Đúng 5/10 câu |
| [Cấu hình và trạng thái huấn luyện](seed123) | Tokenizer, config và trainer state của seed 123 |

Trên Colab, notebook đọc ba CSV và `external_10_sentences.csv` từ `MyDrive/BERT_FakeNews`. Khi train, notebook cập nhật thư mục `best_model` trong đó. Dùng thư mục Drive riêng cho lần chạy mới. Gói `Ly_Bert_Full (1).zip` của thành viên có tệp trọng số `best_model/model.safetensors`. Có thể tạo lại checkpoint bằng phần huấn luyện trong notebook.

[Chuẩn bị dữ liệu và mở notebook](../../docs/HUONG_DAN_CHAY.md) · [So sánh với các phần khác](../../reports/README.md)
