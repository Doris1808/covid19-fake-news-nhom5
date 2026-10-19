# Kiểm tra trước khi chạy thực nghiệm

Đã kiểm tra khi đóng gói:

- Cú pháp các module Python và các ô code notebook.
- Kiểm tra nhãn không hợp lệ, văn bản rỗng, thứ tự ma trận nhầm lẫn và F1 fake.
- Loại lượt smoke/lượt chưa hoàn tất khỏi tổng hợp.
- Xuất EDA và tính mean ± std ba seed trên dữ liệu kiểm thử tạm thời.

Các kiểm tra phần mềm dùng dữ liệu giả; dữ liệu và metric đó không được lưu vào kết quả nghiên cứu của dự án.

Chưa kiểm tra end-to-end: tải model/dataset từ mạng, fine-tune GPU, resume checkpoint và chạy Streamlit trong trình duyệt. Môi trường đóng gói chưa có PyTorch/Transformers/Streamlit; cần thực hiện notebook smoke trên Kaggle rồi nạp lại model trước khi chạy ba seed đầy đủ. File requirements chỉ định Transformers 4.57.1; các thư viện còn lại có khoảng phiên bản, vì vậy mỗi lượt chạy lưu pip_freeze.txt để giữ môi trường thực tế.

Lệnh kiểm tra nhẹ:

```bash
python -m compileall -q src app.py
python -m unittest discover -s tests -v
```
