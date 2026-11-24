# 04 · RoBERTa-base
**Lê Nguyễn Ngọc Bình · 24DH113442**

Bình fine-tune `roberta-base` với năm seed 42, 123, 2024, 2025 và 2026. Seed 123 đạt **0,97849 Validation F1 weighted**. Năm lượt đạt **0,97494 ± 0,00174 Test F1 weighted** trên 2.140 bài gốc. Notebook có thêm thí nghiệm điều chỉnh ngưỡng, làm sạch HTML và giảm phụ thuộc vào dấu hiệu bề mặt trong văn bản.

[Xem notebook và output](04_RoBERTa.ipynb) · [Mở Colab](https://colab.research.google.com/github/Doris1808/covid19-fake-news-nhom5/blob/main/members/binh_roberta/04_RoBERTa.ipynb) · [Validation từng seed](results/roberta_val_5seeds_summary.csv) · [Test từng seed](results/roberta_test_5seeds_summary.csv)

| Minh chứng | Mở nhanh |
|---|---|
| Xu hướng học và loại lỗi | [Đường học 5 seed](figures/roberta_learning_curves_5seeds.png) · [Ma trận nhầm lẫn](figures/roberta_confusion_matrices_5seeds.png) |
| Các thử nghiệm cải tiến | [So sánh trước và sau](results/improvement_ablation_comparison.csv) · [10 câu ngoài](results/external_10_sentences_comparison.csv) |
| Đối chiếu từng dự đoán | [Test seed 123](results/test_predictions_best_seed123.csv) · [Validation seed 123](results/val_predictions_best_seed123.csv) |
| Cấu hình và log | [Nhật ký các seed](logs) · [Kiểm tra dữ liệu trùng](results/data_leakage_report.json) |

Xem kết quả trực tiếp trên GitHub. Để train lại, mở phiên Colab có GPU và để các CSV ở `/content` như hướng dẫn. Riêng Test sau lọc trùng của phần này có 2.065 bài.

[Hướng dẫn chạy](../../docs/HUONG_DAN_CHAY.md) · [So sánh nhóm](../../reports/README.md)
