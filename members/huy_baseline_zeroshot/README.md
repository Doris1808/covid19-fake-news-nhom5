# 01 · Baseline và zero-shot
**Vũ Hồng Huy · 24DH113458**

Notebook khảo sát phân bố nhãn, chuẩn hóa văn bản và so sánh ba phương án trên Test gốc. Với LinearSVC, Huy chọn `C=3` theo Validation. SVM gốc đạt **0,9503 F1 macro** trên Test, biến thể bổ sung n-gram ký tự đạt **0,9611**. Phần zero-shot dùng `facebook/bart-large-mnli` và mẫu giả thuyết `This is {}.`. F1 macro Test là **0,4500**, recall lớp fake là **0,1069**.

[Xem notebook và output](01_Baseline_ZeroShot.ipynb) · [Mở Colab](https://colab.research.google.com/github/Doris1808/covid19-fake-news-nhom5/blob/main/members/huy_baseline_zeroshot/01_Baseline_ZeroShot.ipynb) · [Bảng điểm](results/metrics.csv)

Để chạy lại trên Colab, tải cùng lúc `train.csv`, `val.csv`, `test.csv` ở ô đầu. Các phần SVM chạy được trên CPU. Phần 8 tải BART-MNLI và nên dùng GPU. Xem phần 5 để đối chiếu mẫu trùng và 10 câu minh họa.

[Dữ liệu dùng chung](../../data/raw) · [Bảng so sánh nhóm](../../reports/README.md)
