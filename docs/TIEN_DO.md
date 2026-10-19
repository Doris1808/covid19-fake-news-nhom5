# Tiến độ báo cáo lần 1

Cập nhật theo minh chứng thực tế trước buổi báo cáo. Tình trạng dưới đây mô tả gói mã nguồn, chưa xác nhận việc thực nghiệm trên máy của từng thành viên.

| Thành viên | Đầu việc | Tình trạng trong gói | Minh chứng cần bổ sung |
|---|---|---|---|
| Vũ Hồng Huy | Bài báo, EDA, baseline A, zero-shot B | Có mã EDA hỗ trợ; chưa có code A/B và số liệu thực chạy | Notebook A/B, JSON metric, thống kê CSV |
| Lương Hoàng Bảo Quân | DistilBERT, 3 seed, thời gian/GPU | Đã push GitHub và mở giao diện Streamlit trên Mac; đã import notebook Kaggle; chưa có seed hoàn tất được xác nhận | Checkpoint, metrics, environment, log |
| Đặng Thị Trúc Ly | BERT-base, 3 seed, phân tích lỗi | Chừa thư mục để tích hợp; chưa có kết quả | Code, config, log, checkpoint |
| Lê Nguyễn Ngọc Bình | RoBERTa-base, 3 seed, demo | Chừa thư mục; có khung demo chung dùng DistilBERT | Code, log, checkpoint, demo hoàn chỉnh |

Không suy ra phần trăm hoàn thành từ việc có sẵn file code. Nhóm tự cập nhật phần trăm kèm tiêu chí: setup, chạy thử, các seed, đánh giá và phân tích.

## Checklist theo yêu cầu của cô

- [ ] Tóm tắt bài báo và mục tiêu bằng lời nhóm; rà lại tài liệu tham khảo.
- [ ] EDA có số mẫu, phân bố nhãn và ví dụ dữ liệu thực.
- [ ] Baseline A có số liệu và notebook chạy được.
- [ ] Zero-shot B có số liệu và cấu hình nhãn/template.
- [ ] Mỗi người báo cáo môi trường, số seed đã xong và vướng mắc.
- [ ] Bảng phân công được nhóm xác nhận, kèm mức hoàn thành có căn cứ.

## Nhật ký Quân tự cập nhật

| Ngày | Môi trường/GPU | Seed | Công việc hoặc lỗi | Cách xử lý | Đường dẫn minh chứng |
|---|---|---|---|---|---|
| 19/10/2026 | Mac M1, môi trường `.venv`; Kaggle đang thiết lập GPU | Chưa train | Đã cài thư viện, mở giao diện và push mã nguồn | Tiếp tục xác minh quyền GPU trên Kaggle | Repository GitHub và notebook; chưa có log fine-tune |
