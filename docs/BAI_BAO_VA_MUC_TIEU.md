# Bài báo nền tảng và mục tiêu nghiên cứu

Bài báo: Patwa và cộng sự, *Fighting an Infodemic: COVID-19 Fake News Dataset*. Bản tiền ấn phẩm: https://arxiv.org/abs/2011.03327. Kho mã nguồn tác giả: https://github.com/parthpatwa/covid19-fake-news-detection.

## Tóm tắt để nhóm thảo luận

Nghiên cứu đặt vấn đề rằng lượng thông tin COVID-19 lan truyền nhanh khiến việc nhận diện nội dung sai lệch trở nên khó khăn. Tác giả xây dựng dữ liệu tiếng Anh có hai nhãn real/fake và đưa ra các mô hình cơ sở để các nghiên cứu sau có thể so sánh trên cùng bài toán. Đóng góp chính là nguồn dữ liệu có nhãn và thiết lập thực nghiệm chung; một mô hình có điểm cao trên dữ liệu này chưa thể được xem là hệ thống kiểm chứng mọi tin tức ngoài thực tế.

Nhóm sử dụng bài toán này để tìm hiểu quá trình chuyển văn bản thành đầu vào mô hình, huấn luyện bộ phân loại và đánh giá trên dữ liệu chưa dùng để học. Phần của Quân tập trung vào DistilBERT: mô hình có thể đạt kết quả như thế nào sau fine-tune, kết quả thay đổi bao nhiêu giữa ba seed và cần thời gian/tài nguyên ra sao?

## Các câu hỏi cần trả lời bằng thực nghiệm

1. DistilBERT có cải thiện F1 weighted so với baseline A và zero-shot B của nhóm không?
2. Kết quả giữa seed 42, 123, 2024 có ổn định không?
3. Những mẫu dự đoán sai có đặc điểm gì: văn bản dài, phủ định, trích dẫn hay thiếu ngữ cảnh?
4. Chi phí chạy có phù hợp với tài nguyên nhóm được cấp không?

## Dữ liệu

Dùng các tập Train, Validation, Test có sẵn của CONSTRAINT@AAAI2021. Thống kê tham khảo từ bài báo: 6.420 / 2.140 / 2.140 mẫu. Đây không phải kết quả EDA của lần chạy hiện tại. Sau khi tải, dùng `python -m src.eda` để tạo số liệu từ CSV và giữ file nguồn/checksum.

Code tải bản sao cộng đồng trên Hugging Face, không gọi đó là kho phát hành chính thức. Cần đối chiếu với dữ liệu nhóm thống nhất trước khi train. Dữ liệu không được đóng gói trong repository này.

## Cách diễn giải kết quả

So sánh các mô hình trên cùng split và cùng metric. Báo cáo F1 weighted, Accuracy, F1 macro và F1 fake; đọc thêm ma trận nhầm lẫn để biết lỗi tập trung vào nhãn nào. Thảo luận giới hạn về nguồn dữ liệu, thời điểm thu thập và khả năng mô hình học dấu hiệu phong cách thay vì xác minh sự kiện. Nhóm bổ sung nhận xét riêng sau khi đọc bài báo đầy đủ và xem kết quả thực chạy.
