# Bài báo nền tảng và mục tiêu nghiên cứu

Bài báo: Patwa và cộng sự, *Fighting an Infodemic: COVID-19 Fake News Dataset*. Bản tiền ấn phẩm: https://arxiv.org/abs/2011.03327. Kho mã nguồn tác giả: https://github.com/parthpatwa/covid19-fake-news-detection.

## Tóm tắt để nhóm thảo luận

Nghiên cứu đặt vấn đề rằng lượng thông tin COVID-19 lan truyền nhanh khiến việc nhận diện nội dung sai lệch trở nên khó khăn. Tác giả xây dựng dữ liệu tiếng Anh có hai nhãn real/fake và đưa ra các mô hình cơ sở để các nghiên cứu sau có thể so sánh trên cùng bài toán. Đóng góp chính là nguồn dữ liệu có nhãn và thiết lập thực nghiệm chung; một mô hình có điểm cao trên dữ liệu này chưa thể được xem là hệ thống kiểm chứng mọi tin tức ngoài thực tế.

Nhóm sử dụng bài toán này để tìm hiểu quá trình chuyển văn bản thành đầu vào mô hình, huấn luyện bộ phân loại và đánh giá trên dữ liệu chưa dùng để học. Phần của Quân tập trung vào DistilBERT: mô hình có thể đạt kết quả như thế nào sau fine-tune, kết quả thay đổi bao nhiêu giữa ba seed và cần thời gian/tài nguyên ra sao?

## Các câu hỏi cần trả lời bằng thực nghiệm

1. DistilBERT, baseline A và zero-shot B khác nhau thế nào trên cùng tập Test khi đo bằng F1 macro?
2. Kết quả giữa seed 42, 123, 2024 có ổn định không?
3. Những mẫu dự đoán sai có đặc điểm gì: văn bản dài, phủ định, trích dẫn hay thiếu ngữ cảnh?
4. Chi phí chạy có phù hợp với tài nguyên nhóm được cấp không?

## Dữ liệu của bài nộp

Ba tệp trong [data/raw](../data/raw) có 6.420 dòng Train, 2.140 dòng Validation và 2.140 dòng Test. [Notebook DistilBERT](../members/quan_distilbert/02_DistilBERT.ipynb) đếm nhãn trực tiếp và đối chiếu SHA-256 với hồ sơ thực nghiệm. Mã tải dữ liệu trước đây dùng [bản sao CSV trên Hugging Face](https://huggingface.co/datasets/nanyy1025/covid_fake_news).

## Đọc số liệu

[Bảng tổng hợp](../reports/README.md) dùng cùng tập Test gốc và cùng loại F1. Phần ngoài bộ Test và phép kiểm tra sau lọc trùng có số mẫu riêng. Các đầu ra là dự đoán theo nhãn của bộ dữ liệu nghiên cứu.
