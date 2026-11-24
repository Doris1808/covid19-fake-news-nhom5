# So sánh kết quả trên tập Test gốc

Cả sáu phương án được đánh giá trên cùng 2.140 bài, gồm 1.020 fake và 1.120 real. Accuracy là tỉ lệ dự đoán đúng. F1 macro cho hai lớp trọng số bằng nhau. F1 weighted lấy trung bình theo số mẫu mỗi lớp.

| Phương pháp | Lượt | Accuracy | F1 macro | F1 weighted |
|---|---:|---:|---:|---:|
| TF-IDF từ + LinearSVC | 1 | 0,9505 | 0,9503 | — |
| TF-IDF từ và ký tự + LinearSVC | 1 | 0,9612 | 0,9611 | — |
| Zero-shot BART-MNLI | 1 | 0,5710 | 0,4500 | — |
| DistilBERT | 3 | 0,9717 | 0,9716 ± 0,0018 | 0,9716 ± 0,0018 |
| BERT-base | 3 | 0,9745 | 0,9744 ± 0,0015 | 0,9744 ± 0,0015 |
| RoBERTa-base | 5 | 0,9750 | 0,9749 ± 0,0017 | 0,9749 ± 0,0017 |

Với mô hình fine-tune, bảng ghi trung bình qua các seed và độ lệch chuẩn mẫu của F1 (n − 1). DistilBERT và BERT dùng 42, 123, 2024. RoBERTa có thêm 2025 và 2026. Khoảng ± thể hiện mức dao động giữa những lượt đã chạy. Với SVM và BART, nhóm ghi một lượt và giữ độ chính xác làm tròn của notebook. Dấu — là chỉ số phần đó chưa báo cáo.

## Đối chiếu với output

- [Baseline và zero-shot](../members/huy_baseline_zeroshot/01_Baseline_ZeroShot.ipynb) · SVM gốc và bản thêm n-gram ký tự ở phần 4 và 7, BART-MNLI ở phần 8. Trích các dòng Test Accuracy và F1 macro mà notebook đã in.
- [DistilBERT](distilbert) · Mỗi thư mục seed có `metrics.json` cho Validation và `test_metrics.json` cho Test. Các ma trận theo thứ tự fake, real cho cùng điểm với các JSON.
- [BERT-base](../members/ly_bert/results/metrics_3_seeds.csv) · Bảng lấy Validation từ output của từng seed. Trên Test, tính Accuracy, F1 macro và F1 weighted từ bốn ô ma trận nhầm lẫn rồi đối chiếu Accuracy và F1 weighted với bảng in trong notebook.
- [RoBERTa-base](../members/binh_roberta/results/roberta_test_5seeds_summary.csv) · Bảng dùng cột của Test gốc. F1 macro được tính từ ma trận của từng seed rồi mới lấy trung bình. Chỉ số trên Test sau lọc trùng nằm riêng trong cùng CSV.

## Kiểm tra ngoài tập Test

Bản SVM gốc đúng 7/10 câu minh họa của Huy. BERT seed 123 đúng 5/10. RoBERTa seed 123 đúng 5/10 với mô hình gốc, 7/10 sau biến thể làm sạch và tăng cường dữ liệu. Với DistilBERT, bảng seed 123 lưu **hai** ví dụ WHO, cả hai bị gán nhầm. Notebook có ô thử bộ 10 câu chung khi gắn checkpoint. Các mẫu ngoài chỉ minh họa lỗi, quy mô nhỏ và có cách thu thập riêng.

[Mẫu ngoài của DistilBERT](distilbert/external_examples_seed123.csv) · [BERT](../members/ly_bert/results/external_10_results.csv) · [RoBERTa](../members/binh_roberta/results/external_10_sentences_comparison.csv)

## Giới hạn của phép so sánh

Theo cách chuẩn hóa của Bình, 75 dòng Test có văn bản trùng với Train. Phép kiểm tra bổ sung còn 2.065 bài. Bộ Test gốc được giữ để đối chiếu các phương pháp. Quy tắc tiền xử lý của Huy nhận diện 34 dòng trùng, nên hai số đếm mô tả hai phép kiểm tra khác nhau. Điểm số đo khả năng khớp nhãn của bộ dữ liệu nghiên cứu.

[Biên bản kiểm tra trùng](../members/binh_roberta/results/data_leakage_report.json) · [Số liệu để tải](so_sanh_test.csv) · [Trang bài nộp](../README.md)
