# Phần Huy — EDA, baseline A, zero-shot B

Vũ Hồng Huy — 24DH113458 — Nhóm 5.

Chạy các lệnh sau từ thư mục gốc repo, sau khi cài `requirements.txt`:

```bash
python -m members.huy_baseline_zeroshot.data
python -m members.huy_baseline_zeroshot.data_exploration
python -m members.huy_baseline_zeroshot.baseline_experiments
python -m members.huy_baseline_zeroshot.baseline_analysis
python -m members.huy_baseline_zeroshot.zeroshot_experiments
python -m members.huy_baseline_zeroshot.make_summary
```

- Dữ liệu chung: `data/raw/train.csv`, `val.csv`, `test.csv`; giữ nguyên split.
- Toàn bộ kết quả phần Huy: `outputs/baseline_zeroshot/`. Không trộn với `outputs/distilbert/`.
- Mã nhãn phần Huy: `real=0`, `fake=1`; khi ghép với DistilBERT phải giải mã về chữ. Cột `id` được giữ lại để đối chiếu.
- Trước khi chạy phân tích baseline, chọn cấu hình trong `baseline_analysis.py` theo F1 Validation của bảng `baseline_best_per_model.csv`.
- Zero-shot hiện chọn cấu hình và ngưỡng trên **200** mẫu Val, random_state=0. Đây là giá trị code cũ; báo cáo trước ghi 500 nên cần đối chiếu lượt chạy thực tế. Không đổi số mẫu để hợp thức hóa kết quả cũ.
- `make_summary` yêu cầu cả kết quả baseline và `zeroshot_best.json`; không sinh số zero-shot nếu thiếu file. Model, template và ngưỡng lấy từ JSON.
- Baseline hiện tính Test cho từng cấu hình nhưng chọn theo Val. Cần tách đánh giá Test sau khi chốt cấu hình trước khi coi đây là pipeline thực nghiệm cuối cùng; không mô tả code hiện tại là chỉ đánh giá Test một lần.
- Không coi thay đổi cấu trúc này là một lượt chạy thực nghiệm mới. Giữ log, cấu hình và metric gốc khi bổ sung kết quả.

Thư mục `outputs/` đang được Git bỏ qua. Chỉ đưa các bảng/JSON/log cần báo cáo vào `reports/baseline_zeroshot/` sau khi kiểm tra; model/checkpoint lưu riêng.
