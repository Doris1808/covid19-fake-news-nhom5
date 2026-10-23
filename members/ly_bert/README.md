# Đặng Thị Trúc Ly

Phần phụ trách: BERT-base, huấn luyện 3 seed, đánh giá Test, phân tích lỗi, thử nghiệm câu ngoài và chuẩn bị demo (Chạy tay).

Thư mục này chứa mã nguồn (`.ipynb`), các file cấu hình (`config.json`, `training_args.bin`) và nhật ký kết quả của mô hình BERT. Tuân thủ quy định dùng chung split của nhóm, nhãn `fake=0`/`real=1`, tiêu chí chọn checkpoint tốt nhất là Validation F1 weighted. Các metric được ghi nhận trên thang 0–1. Tuyệt đối không commit file trọng số nặng (model.safetensors).

---

## 1. Kết quả thực nghiệm (Đánh giá trên tập Validation và Test)

Quá trình huấn luyện sử dụng mô hình `bert-base-uncased`, nhãn phân loại được map theo chuẩn chung (`fake=0`, `real=1`). Dữ liệu sử dụng chung split chuẩn của nhóm (6.420 mẫu Train / 2.140 mẫu Validation / 2.140 mẫu Test).

### 1.1. Kết quả trên tập Validation (2.140 mẫu: 1.020 Fake, 1.120 Real)

- **Seed 42:** Validation F1 Weighted = `0.979432` (Accuracy = `0.979439`)
- **Seed 123:** Validation F1 Weighted = `0.978503` (Accuracy = `0.978505`)
- **Seed 2024:** Validation F1 Weighted = **`0.980368`** (Accuracy = **`0.980374`**) — *(Tốt nhất)*
- **Trung bình (Mean ± Std trên tập Validation):** F1 Weighted = `0.979434 ± 0.000933`

**Lựa chọn Checkpoint:** Checkpoint tại **Epoch 3 của Seed 2024 (checkpoint-1206)** được chọn làm kết quả chính thức để đánh giá Test, tích hợp và chạy Demo vì đạt Validation F1 weighted cao nhất.

### 1.2. Kết quả đánh giá độc lập trên tập Test cả 3 Seed (2.140 mẫu)

Kết quả chạy thực tế trên tập Test giữ kín của cả 3 mô hình đã huấn luyện:

| Seed | Split | Accuracy | F1 Weighted | Tổng số câu sai | Tỉ lệ lỗi |
| --- | --- | --- | --- | --- | --- |
| **Seed 42** | Test | 0.981308 | 0.981304 | 40 / 2140 | 0.018692 |
| **Seed 123** | Test | 0.979439 | 0.979436 | 44 / 2140 | 0.020561 |
| **Seed 2024** | Test | 0.979439 | 0.979437 | 44 / 2140 | 0.020561 |
| **Mean ± Std** | **Test** | **0.980062 ± 0.001079** | **0.980059 ± 0.001078** | **42.7 / 2140** | **0.019938** |

*(Ghi chú phần cứng: Các seed được huấn luyện trên Google Colab với GPU T4 - 16GB VRAM. Thiết lập `batch_size=16`, max_length=128, huấn luyện 3 epoch).*

---

## 2. Nhật ký Phân tích lỗi & Kiểm thử câu ngoài (Error Analysis)

### 2.1. Phân tích lỗi trên tập Validation (Checkpoint Seed 2024)

Sử dụng checkpoint **Seed 2024** để dự đoán lại tập Validation, hệ thống ghi nhận có **42 câu dự đoán sai**. Từ việc trích xuất và phân tích 10 mẫu lỗi đầu tiên, nhận thấy 2 nguyên nhân cốt lõi khiến mô hình nhầm lẫn:

- **Lỗi False Positives (Nhãn thật `Fake=0`, đoán nhầm `Real=1`):** Mô hình dễ bị lừa bởi các tin giả "đội lốt" bài đính chính. Ví dụ: *"There is no evidence that children have died..."*. Tin giả này sử dụng văn phong học thuật, trịnh trọng và chèn kèm URL. Do BERT không thể tra cứu sự thật ngoài đời, nó đánh giá cấu trúc văn phong này là của tin chính thống và gán nhãn `Real`.
- **Lỗi False Negatives (Nhãn thật `Real=1`, đoán nhầm `Fake=0`):** Các đoạn thông báo y tế thực tế nhưng viết bằng ngôn ngữ quá đời thường, thân mật hoặc quá ngắn gọn (Ví dụ: *"A reminder about maintaining good hygiene: wash your hands often..."*) thường bị đẩy về nhãn `Fake` do thiếu các đặc trưng quen thuộc như đường dẫn URL hay metadata hashtag.

### 2.2. Thử nghiệm 7 câu ngoài có nguồn (External Wild Samples)

Sử dụng checkpoint tốt nhất **Seed 2024** để kiểm thử Live trên 7 câu tiếng Anh thực tế (có chứa 2 câu của tổ chức WHO mà báo cáo nhóm đã nhắc tới). Kết quả mô hình đoán **đúng 3/7 câu**:

| Mã | Nguồn | Câu thực tế ngoài tập dữ liệu | Nhãn thật | Dự đoán | Kết quả |
| --- | --- | --- | --- | --- | --- |
| **E1** | CDC Official | *"According to the CDC, COVID-19 vaccines are safe and effective. Millions of people have received COVID-19 vaccines..."* | `Real (1)` | `Fake (0)` | **SAI (FN)** |
| **E2** | WHO Mythbusters | *"The WHO states that antibiotics do not work against viruses, they only work on bacterial infections..."* | `Real (1)` | `Fake (0)` | **SAI (FN)** |
| **E3** | Internet Myth | *"Holding your breath for 10 seconds or more without coughing or feeling discomfort means you are free from the coronavirus..."* | `Fake (0)` | `Fake (0)` | **ĐÚNG** |
| **E4** | Conspiracy Theory | *"COVID-19 vaccines contain microchips to track people's movements and control populations via 5G towers."* | `Fake (0)` | `Fake (0)` | **ĐÚNG** |
| **E5** | Internet Rumor | *"Drinking alcohol or spraying chlorine all over your body will kill the new coronavirus."* | `Fake (0)` | `Fake (0)` | **ĐÚNG** |
| **E6** | WHO Viet Nam | *"People of all ages can get infected by the new coronavirus."* | `Real (1)` | `Fake (0)` | **SAI (FN)** |
| **E7** | WHO Viet Nam | *"No, drinking alcohol does not protect you from COVID-19 infection."* | `Real (1)` | `Fake (0)` | **SAI (FN)** |

**Kết luận phân tích câu ngoài:** Mô hình BERT-base bắt tin giả (Fake) rất xuất sắc (đoán trúng 100% các câu Conspiracy và Rumor). Tuy nhiên, với các tin `Real` lấy từ Internet, mô hình đều đoán nhầm thành Fake. Nguyên nhân do các câu này không chứa các dấu hiệu quen thuộc trong tập Train (như link URL `t.co`, hashtag). Việc thiếu metadata hoặc sử dụng cấu trúc phủ định cục bộ (*"No"*, *"does not"*) khiến mô hình bị lệch hướng, tương tự như lỗi của DistilBERT mà nhóm đã phân tích.

---

## 3. Phần Demo (Chạy tay trực tiếp trong Notebook)

Để đáp ứng yêu cầu "Chạy tay" trực tiếp của giảng viên, một hàm suy luận (Inference) đã được tích hợp ở cuối notebook `Ly_Bert_2.ipynb`.

- **Cách hoạt động:** Hàm `predict_fake_news(text)` nhận câu do giảng viên cung cấp, tự động Tokenize với `max_length=128`, nạp vào ma trận trọng số của GPU và trả về nhãn cuối cùng (`Fake` hoặc `Real`).
- **Ví dụ output thực tế trong Notebook:**
  - Input: `"States reported 1121 deaths a small rise from last Tuesday. Southern states reported 640 of those deaths. https://t.co/YASGRTT4ux,"`
  - Dự đoán tự động trả về: **`=> Kết quả dự đoán: Real (Tin thật)`**

---

## 4. Hướng dẫn chạy code và Minh chứng

- **Mã nguồn:** Toàn bộ quá trình từ nạp dữ liệu, xử lý Tokenizer, Train 3 Seed, Đánh giá Test, Phân tích lỗi Validation, Test 7 câu mạng và Live Demo đều nằm trọn vẹn trong file notebook `Ly_Bert_2.ipynb`.
- **Cài đặt thư viện:** `!pip install transformers datasets scikit-learn`
- **Tái lập kết quả:** Dùng Google Colab (GPU T4). Bấm *Runtime > Run all* để chạy từ trên xuống dưới. Mô hình sẽ tự động trỏ vào `checkpoint-1206` của thư mục `bert_base_seed_2024` để báo cáo điểm cuối cùng.
- **Trọng số:** Theo quy định của nhóm, các tệp quá lớn (`*.safetensors`) đã bị loại bỏ qua lệnh zip. File `.zip` được đính kèm ở thư mục này chứa đầy đủ `config.json` và log làm minh chứng thực nghiệm.
```eof

