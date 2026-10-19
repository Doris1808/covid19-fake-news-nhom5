# Chạy phần DistilBERT

## 1. Repository đã có mã nguồn

Repository nhóm 5: https://github.com/Doris1808/covid19-fake-news-nhom5. Notebook đã dùng sẵn địa chỉ này. Mac của Quân đã cài thư viện và mở được giao diện Streamlit.

## 2. Mở notebook trên Kaggle

1. Tạo notebook rồi import `notebooks/01_DistilBERT_Kaggle.ipynb`.
2. Bật Internet để tải code, dữ liệu và mô hình. Chọn GPU nếu tài khoản được cấp; xác minh tài khoản nếu Kaggle yêu cầu. Nếu không có GPU, có thể chuyển notebook sang Colab và đổi đường dẫn làm việc.
3. Chạy ô clone code; `REPO_URL` đã trỏ tới repo của nhóm 5. Hướng dẫn clone này dành cho repository public. Với private repo, tải code lên dạng Kaggle Dataset riêng rồi copy sang `/kaggle/working`; không ghi token trong notebook.
4. Cài thư viện. Nếu notebook yêu cầu khởi động lại kernel, restart rồi chạy lại ô thiết lập đường dẫn trước khi đi tiếp.
5. Chạy kiểm tra GPU, tải CSV và EDA. Xem nhãn, số dòng và cảnh báo dữ liệu trùng.
6. Chạy smoke một lần. Kiểm tra có `model/`, `metrics.json` và `completed.flag` trong `outputs/smoke/seed_42/`.
7. Chạy seed 42 trên toàn bộ Train. Xem output và file kết quả, sau đó thay seed bằng 123 rồi 2024. Mỗi lần chỉ chạy một seed.
8. Chạy ô tổng hợp và nén output. Lưu phiên bản notebook/output trên Kaggle và tải ZIP về trước khi kết thúc phiên.

Notebook giới hạn còn một GPU nhìn thấy để tránh vô tình thay đổi effective batch khi máy được cấp nhiều GPU. Mặc định fp32 cho dễ đối chiếu; chưa bật mixed precision.

## 3. File phải giữ sau mỗi seed

- `model/`: trọng số và tokenizer dùng cho demo.
- `run_config.json`: cấu hình, seed, SHA256 Train/Val.
- `environment.json`, `pip_freeze.txt`: môi trường thực chạy và GPU.
- `metrics.json`: metric Validation, thời gian, bộ nhớ CUDA peak.
- `log_history.json`: loss và metric theo tiến trình.
- `val_predictions.csv`, `val_errors.csv`: dự đoán và mẫu sai.
- `completed.flag`: chỉ có khi toàn bộ lượt chạy đã kết thúc.

`train_seconds_this_session` tính phần gọi train trong phiên hiện tại (bao gồm đánh giá/lưu checkpoint trong train); không gồm tải model hay đánh giá cuối. Nếu resume, không xem thời gian này là tổng toàn bộ các phiên. Bộ nhớ peak là tensor CUDA được cấp phát trên thiết bị chính, không phải toàn bộ VRAM máy; MPS/CPU để null.

## 4. Thiếu bộ nhớ hoặc phiên bị ngắt

Giảm `train_batch_size` từ 8 xuống 4, tăng `gradient_accumulation_steps` từ 4 lên 8 để giữ effective batch 32 trên một GPU. Giảm `eval_batch_size` nếu lỗi lúc đánh giá. Chốt cấu hình rồi chạy đủ ba seed với cùng cấu hình, không trộn kết quả khác cấu hình.

Checkpoint mặc định lưu cuối mỗi epoch. Phiên ngắt giữa epoch có thể mất tiến độ từ lần lưu gần nhất. Nếu còn checkpoint, dùng:

```bash
python -m src.finetune --seed 42 --resume outputs/distilbert/seed_42/checkpoints/checkpoint-N
```

Thay `checkpoint-N` bằng thư mục thật. Resume cần giữ nguyên config, dữ liệu, seed và output. File `failure.json` có thể vẫn còn từ lần lỗi trước; xem `completed.flag` để biết trạng thái cuối. Nếu muốn chạy lại từ đầu, dùng `--out-dir outputs/retry/seed_42`, không ghi đè kết quả cũ.

## 5. Chạy giao diện trên Mac

Giải nén output tải từ Kaggle và đặt thư mục `seed_42` vào `outputs/distilbert/`. Cần có đường dẫn `outputs/distilbert/seed_42/model/config.json` và file trọng số trong cùng thư mục.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mở địa chỉ hiện trong terminal. Chưa có checkpoint thì giao diện vẫn hiển thị hướng dẫn nhưng chưa cho dự đoán. Suy luận mặc định dùng CPU; không cần train lại trên Mac. Chọn một checkpoint theo kế hoạch Validation của nhóm để demo, không chọn bằng điểm Test.

Muốn thử train trên M1: copy config sang `configs/distilbert_m1.json`, đặt train batch 2, gradient accumulation 16, eval batch 4, rồi chạy `--config configs/distilbert_m1.json --smoke`. Kiểm tra `environment.json` có device `mps`. Khả năng chạy toàn bộ phụ thuộc RAM; chưa được đo trên máy của bạn.

## 6. Đánh giá Test sau khi chốt

Train không tự đọc Test. Sau khi cố định thiết kế thí nghiệm, đánh giá từng seed theo cùng quy trình:

```bash
python -m src.evaluate --run-dir outputs/distilbert/seed_42 --confirm-config-frozen
```

Lặp lại với 123 và 2024. Lệnh lưu `test_metrics.json` riêng; không lấy seed có điểm Test cao nhất để báo cáo thay cho cả ba seed. Bản đánh giá này dùng CPU và xử lý từng mẫu để giảm bộ nhớ, có thể chậm.
