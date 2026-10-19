# Mô hình DistilBERT của nhóm 5

Chưa có checkpoint đã train được xác nhận. Giao diện Streamlit trên Mac đã mở được; chức năng dự đoán sẽ dùng checkpoint sau khi chạy Kaggle.

Notebook tạo gói `/kaggle/working/distilbert_outputs.zip`. Tải gói này về Mac rồi giải nén sao cho thư mục dự án chứa:

```text
outputs/distilbert/seed_42/model/
outputs/distilbert/seed_42/run_config.json
outputs/distilbert/seed_42/metrics.json
```

Nếu chọn seed 123 hoặc 2024 để demo theo Validation, đổi đường dẫn checkpoint ở thanh bên của giao diện. Không cần nhập email hay token vào mã nguồn để nạp mô hình cục bộ.

Thông tin GPU, seed, cấu hình, thời gian và metric được chương trình ghi tự động trong các file output. Link tải checkpoint chia sẻ cho cô và các bạn sẽ được bổ sung sau khi nhóm lưu output Kaggle; hiện chưa có link mô hình để công bố.
