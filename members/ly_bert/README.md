# Đặng Thị Trúc Ly

Phần phụ trách: BERT-base, huấn luyện 3 seed, đánh giá Test (Precision/Recall, Confusion Matrix), phân tích lỗi, thử nghiệm câu ngoài và chuẩn bị demo (tích hợp id2label).

Thư mục này dành cho code được tích hợp sau. Chứa mã nguồn (`Ly_Bert_2.ipynb`), các file cấu hình (`config.json`, `training_args.bin`) và nhật ký kết quả của mô hình BERT. Tuân thủ nghiêm ngặt quy định dùng chung split của nhóm, nhãn `fake=0`/`real=1`. Tiêu chí chọn checkpoint tốt nhất là Validation F1 weighted. Các metric được ghi nhận trên thang 0–1. Không ghi đè thư mục kết quả của thành viên khác và tuyệt đối không commit file trọng số nặng (`model.safetensors`).

---

## 1. Kết quả thực nghiệm (Đánh giá trên tập Validation và Test)

Quá trình huấn luyện sử dụng mô hình `bert-base-uncased`, nhãn phân loại được cấu hình cứng vào metadata của mô hình thông qua `id2label` (`fake=0`, `real=1`) để đảm bảo tính đồng nhất. Mô hình được tinh chỉnh với `learning_rate=2e-5` và `weight_decay=0.01`. Dữ liệu sử dụng chung split chuẩn của nhóm (6.420 mẫu Train / 2.140 mẫu Validation / 2.140 mẫu Test).

### 1.1. Kết quả trên tập Validation (2.140 mẫu: 1.020 Fake, 1.120 Real)

- **Seed 42:** Validation F1 Weighted = `0.979432` (Accuracy = `0.979439`)
- **Seed 123:** Validation F1 Weighted = `0.978503` (Accuracy = `0.978505`)
- **Seed 2024:** Validation F1 Weighted = **`0.980368`** (Accuracy = **`0.980374`**) — *(Tốt nhất)*
- **Trung bình (Mean ± Std trên tập Validation):** F1 Weighted = `0.979434 ± 0.000933`

**Lựa chọn Checkpoint:** Checkpoint tại **Epoch 3 của Seed 2024 (checkpoint-1206)** được chọn làm kết quả chính thức để đánh giá Test, tích hợp và chạy Demo vì đạt Validation F1 weighted cao nhất.

### 1.2. Kết quả trên tập Test cả 3 Seed (2.140 mẫu: 1.020 Fake, 1.120 Real)

Đánh giá độc lập cả 3 seed trên tập Test (thang 0–1). Mã nguồn đã được cấu hình để trích xuất đầy đủ Báo cáo phân loại (Classification Report) bao gồm Precision, Recall, F1 cho từng lớp và Ma trận nhầm lẫn (Confusion Matrix).

| Seed | Split | Accuracy | F1 Weighted | Tổng số câu sai | Tỉ lệ lỗi |
| --- | --- | --- | --- | --- | --- |
| **Seed 42** | Test | **0.981308** | **0.981304** | 40 / 2140 | 0.018692 |
| **Seed 123** | Test | 0.979439 | 0.979436 | 44 / 2140 | 0.020561 |
| **Seed 2024** | Test | 0.979439 | 0.979437 | 44 / 2140 | 0.020561 |
| **Mean ± Std** | **Test** | **0.980062 ± 0.001079** | **0.980059 ± 0.001078** | **42.7 / 2140** | **0.019938** |

*(Ghi chú phần cứng: Các seed được huấn luyện trên Google Colab với GPU T4 - 16GB VRAM. Thiết lập `batch_size=16`, `max_length=128`, huấn luyện 3 epoch).*

---

## 2. Nhật ký Phân tích lỗi & Kiểm thử câu ngoài (Error Analysis)

### 2.1. Phân tích lỗi trên tập Validation (Checkpoint Seed 2024)

Sử dụng checkpoint **Seed 2024 (Tốt nhất)** để dự đoán lại tập Validation, mô hình ghi nhận **42 câu dự đoán sai** (Tỷ lệ lỗi 1.96%). Từ việc trích xuất các mẫu lỗi, rút ra 2 nguyên nhân cốt lõi khiến mô hình nhầm lẫn:

- **Tin giả mang văn phong báo chí (False Positives – Nhãn thật `Fake=0`, đoán nhầm `Real=1`):** Mô hình dễ bị lừa bởi các tin giả "đội lốt" đính chính sự thật. Ví dụ: *"There is no evidence that children have died because of a COVID-19 vaccine..."*. Tin giả này sử dụng cấu trúc văn phong học thuật, trịnh trọng và chèn kèm URL. Do BERT phân loại dựa trên nội dung văn bản chứ không kết nối cơ sở dữ liệu y khoa để kiểm chứng sự thật (fact-checking), nó đánh giá cấu trúc này là của tin chính thống và gán nhãn `Real`.
    
- **Tin thật thiếu dấu hiệu nhận diện (False Negatives – Nhãn thật `Real=1`, đoán nhầm `Fake=0`):** Các đoạn thông báo thực tế nhưng viết bằng ngôn ngữ quá đời thường, hoặc là lời nhắc nhở ngắn gọn (Ví dụ: *"A reminder about maintaining good hygiene: wash your hands often..."*). Việc thiếu các đặc trưng quen thuộc của một bài báo cáo (như đường dẫn URL `t.co`, số liệu thống kê) khiến BERT nghi ngờ và đẩy về nhãn `Fake`.

### 2.2. Thử nghiệm 7 câu ngoài có nguồn (External Wild Samples)

Sử dụng checkpoint **Seed 2024** để kiểm thử trên 7 câu tiếng Anh thực tế ngoài tập dữ liệu. Kết quả mô hình đoán **đúng 3/7 câu**:

| Mã | Nguồn | Câu thực tế ngoài tập dữ liệu | Nhãn thật | Dự đoán | Kết quả |
| --- | --- | --- | --- | --- | --- |
| **E1** | CDC Official | *"According to the CDC, COVID-19 vaccines are safe and effective. Millions of people have received COVID-19 vaccines..."* | `Real (1)` | `Fake (0)` | **SAI (FN)** |
| **E2** | WHO Mythbusters | *"The WHO states that antibiotics do not work against viruses, they only work on bacterial infections..."* | `Real (1)` | `Fake (0)` | **SAI (FN)** |
| **E3** | Internet Myth | *"Holding your breath for 10 seconds or more without coughing or feeling discomfort means you are free from the coronavirus..."* | `Fake (0)` | `Fake (0)` | **ĐÚNG** |
| **E4** | Conspiracy Theory | *"COVID-19 vaccines contain microchips to track people's movements and control populations via 5G towers."* | `Fake (0)` | `Fake (0)` | **ĐÚNG** |
| **E5** | Internet Rumor | *"Drinking alcohol or spraying chlorine all over your body will kill the new coronavirus."* | `Fake (0)` | `Fake (0)` | **ĐÚNG** |
| **E6** | WHO Viet Nam | *"People of all ages can get infected by the new coronavirus."* | `Real (1)` | `Fake (0)` | **SAI (FN)** |
| **E7** | WHO Viet Nam | *"No, drinking alcohol does not protect you from COVID-19 infection."* | `Real (1)` | `Fake (0)` | **SAI (FN)** |

**Giải thích lỗi:** Mô hình BERT-base bắt tin đồn/tin giả (Fake) rất xuất sắc (đoán trúng 100% các câu Thuyết âm mưu và Lời đồn). Tuy nhiên, với các tin `Real` lấy từ Internet, mô hình đều đoán nhầm thành Fake. Nguyên nhân cốt lõi là do sự chênh lệch phân bố từ vựng: Các câu thử nghiệm không chứa các dấu hiệu quen thuộc trong tập Train (metadata tài khoản, hashtag). Việc sử dụng cấu trúc phủ định cục bộ (*"No"*, *"does not"*) trong câu E2 và E7 cũng đánh lừa mô hình, tương đồng với lỗi mà mô hình DistilBERT của nhóm gặp phải.

---

## 3. Phần Demo (Nhập câu và trả nhãn trực tiếp trong Notebook)

Để đáp ứng yêu cầu "Chạy tay" trực tiếp và đảm bảo không xảy ra hiện tượng đảo nhãn khi deploy, hàm suy luận (Inference) đã được cập nhật.

- **Đảm bảo đồng nhất nhãn (Label Consistency):** Hàm sử dụng trực tiếp cấu hình `best_model.config.id2label` đã được lưu cứng trong file `config.json` để ánh xạ dự đoán (0 -> Fake, 1 -> Real
