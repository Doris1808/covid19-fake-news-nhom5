# Đặng Thị Trúc Ly

Phần phụ trách: BERT-base, ba seed và phân tích lỗi.

Thư mục này dành cho code được tích hợp sau. Thêm README hướng dẫn chạy, notebook hoặc module, config, kết quả thật và nhật ký. Dùng chung split với nhóm, nhãn fake=0/real=1, chọn checkpoint bằng Validation F1 weighted. Ghi metric trên thang 0–1 và ghi rõ split. Không ghi đè thư mục kết quả của thành viên khác.
## 1. Kết quả thực nghiệm (Đánh giá trên tập Validation)
Quá trình huấn luyện sử dụng mô hình `bert-base-uncased`, nhãn phân loại được map theo chuẩn chung của nhóm (`fake=0`, `real=1`). Tiêu chí chọn mô hình là Validation F1 weighted trên thang điểm 0-1. Dữ liệu sử dụng chung split chuẩn của nhóm.

*   **Seed 42:** F1 Weighted = 0.977565
*   **Seed 123:** F1 Weighted = 0.979435
*   **Seed 2024:** F1 Weighted = **0.980838** (Tốt nhất)

 **Lựa chọn Checkpoint:** Checkpoint từ **Seed 2024** được chọn làm kết quả chính thức để tích hợp vì đạt Validation F1 cao nhất.

## 2. Nhật ký Phân tích lỗi (Error Analysis)
Sử dụng checkpoint Seed 2024 (Tốt nhất) để dự đoán trên tập Validation, mô hình có 41 câu dự đoán sai. Rút ra 2 nguyên nhân cốt lõi từ các mẫu phân tích:
*   **Tin giả "đội lốt" đính chính sự thật (False Positives):** BERT-base rất dễ bị lừa bởi các tin giả được viết dưới dạng bác bỏ tin đồn (ví dụ: "There is no evidence...", "The government has not..."). Việc sử dụng văn phong báo chí nghiêm túc, trích dẫn số liệu giả và đính kèm link URL khiến mô hình "tin sái cổ" đây là tin Real.
*   **Tin thật mang văn phong đời thường (False Negatives):** Các thông điệp y tế cộng đồng (nhắc nhở rửa tay, ở nhà), tin tức về người nổi tiếng (Noel Gallagher), hoặc các bài đăng dùng nhiều emoji thường bị mô hình gán nhãn Fake. Nguyên nhân do chúng mang sắc thái quá thân mật, kịch tính hoặc thiếu đường link (URL) tham chiếu mà mô hình vốn quen thấy ở các bản tin chính thống.

## 3. Hướng dẫn chạy code và Minh chứng
*   **Mã nguồn:** Code thực thi trên Google Colab với GPU T4. Chi tiết quá trình huấn luyện và dự đoán vui lòng xem tại file notebook `Ly_Bert.ipynb` đính kèm trong thư mục này.
*   **Cài đặt thư viện:** `pip install transformers datasets scikit-learn pandas numpy torch`
*   **Trọng số mô hình (Checkpoint):** Tuân thủ quy tắc nhóm (không commit trọng số lớn), file mô hình nặng (`pytorch_model.bin` / `model.safetensors`) không được tải lên kho chứa chung. Các file `config.json`, `trainer_state.json` và log đính kèm đã minh chứng cho toàn bộ quá trình thực nghiệm này.
