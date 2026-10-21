# Đặng Thị Trúc Ly

Phần phụ trách: BERT-base, ba seed và phân tích lỗi.

Thư mục này dành cho code được tích hợp sau. Thêm README hướng dẫn chạy, notebook hoặc module, config, kết quả thật và nhật ký. Dùng chung split với nhóm, nhãn fake=0/real=1, chọn checkpoint bằng Validation F1 weighted. Ghi metric trên thang 0–1 và ghi rõ split. Không ghi đè thư mục kết quả của thành viên khác.
# 1. Kết quả thực nghiệm (Đánh giá trên tập Validation)
Quá trình huấn luyện sử dụng mô hình `bert-base-uncased`, nhãn phân loại được map theo chuẩn chung của nhóm (`fake=0`, `real=1`). Tiêu chí chọn mô hình là Validation F1 weighted trên thang điểm 0-1.

*   **Seed 42:** F1 Weighted = 0.968385
*   **Seed 123:** F1 Weighted = **0.970074** (Tốt nhất)
*   **Seed 2024:** F1 Weighted = 0.966042

 **Lựa chọn Checkpoint:** Checkpoint từ **Seed 123** được chọn làm kết quả chính thức để tích hợp vì đạt Validation F1 cao nhất.

## 2. Nhật ký Phân tích lỗi (Error Analysis)
Sử dụng checkpoint Seed 123 để dự đoán trên tập Validation, mô hình có 218 câu dự đoán sai. Rút ra 2 nguyên nhân cốt lõi:
*   **Tin giả mang văn phong học thuật (False Positives):** BERT-base dễ bị qua mặt bởi các tin tức giả mạo nhưng được viết theo văn phong báo chí nghiêm túc, có chèn số liệu chi tiết hoặc đính kèm các đường link (URL) giả mạo.
*   **Tin thật chứa từ khóa nhạy cảm (False Negatives):** Các tin chính thống nhưng vô tình chứa các từ khóa từng là tâm điểm của tin đồn nhảm (ví dụ: "tái sử dụng khẩu trang N95", "tia UV") hoặc mang nội dung kịch tính cao thường bị mô hình mang "định kiến" và gán nhãn Fake.

## 3. Hướng dẫn chạy code và Minh chứng
*   **Mã nguồn:** Code thực thi trên Google Colab với GPU T4. Chi tiết quá trình huấn luyện và dự đoán vui lòng xem tại file notebook `Ly_Bert.ipynb` đính kèm trong thư mục này.
*   **Cài đặt thư viện:** `pip install transformers datasets scikit-learn pandas numpy torch`
*   **Trọng số mô hình (Checkpoint):** Tuân thủ quy tắc nhóm (không commit trọng số lớn), file mô hình nặng (`pytorch_model.bin`) không được tải lên kho chứa chung. Quá trình thực nghiệm và thông số cấu hình đã được ghi nhận trực tiếp trong file notebook.
