# Đặng Thị Trúc Ly

Phần phụ trách: BERT-base, huấn luyện 3 seed, đánh giá Test (Precision/Recall, Confusion Matrix), phân tích lỗi, thử nghiệm 10 câu ngoài dữ liệu (`external_10_sentences.csv`) và chuẩn bị demo (tích hợp `id2label`).

Thư mục này chứa mã nguồn (`Ly_Bert.ipynb`), dữ liệu thử nghiệm ngoài (`external_10_sentences.csv`), các bảng nhật ký phân tích lỗi (`val_errors.csv`, `external_10_results.csv`) và tệp cấu hình minh chứng (`config_seed_123.zip`). Tuân thủ nghiêm ngặt quy định dùng chung split của nhóm, nhãn `fake=0`/`real=1`. Tiêu chí chọn checkpoint tốt nhất là **Validation F1 weighted**. Các metric được ghi nhận trên thang 0–1. Không ghi đè thư mục kết quả của thành viên khác và tuyệt đối không commit file trọng số nặng (`model.safetensors`).

---

## 1. Kết quả thực nghiệm (Đánh giá trên tập Validation và Test)

Quá trình huấn luyện sử dụng mô hình `bert-base-uncased`, nhãn phân loại được cấu hình cứng vào metadata của mô hình thông qua `id2label` (`0: Fake`, `1: Real`) để đảm bảo tính đồng nhất. Mô hình được tinh chỉnh với siêu tham số `learning_rate=2e-5`, `weight_decay=0.01`, `batch_size=16`, `max_length=128`, huấn luyện 3 epoch trên Google Colab GPU T4. Dữ liệu sử dụng chung split chuẩn của nhóm: **6.420 mẫu Train (3.060 Fake, 3.360 Real) / 2.140 mẫu Validation (1.020 Fake, 1.120 Real) / 2.140 mẫu Test (1.020 Fake, 1.120 Real)**.

### 1.1. Kết quả trên tập Validation (2.140 mẫu: 1.020 Fake, 1.120 Real)

| Seed | Val Loss | Accuracy | F1 Weighted | Tổng số câu sai | Tỉ lệ lỗi |
| --- | --- | --- | --- | --- | --- |
| **Seed 42** | 0.137352 | 0.971963 | 0.971946 | 60 / 2140 | 0.028037 |
| **Seed 123** *(Tốt nhất)* | 0.133601 | **0.974299** | **0.974280** | **55 / 2140** | **0.025701** |
| **Seed 2024** | 0.130571 | 0.971963 | 0.971954 | 60 / 2140 | 0.028037 |
| **Mean ± Std** | — | **0.972741 ± 0.001349** | **0.972727 ± 0.001345** | **58.33 / 2140** | **0.027259** |

**Lựa chọn Checkpoint:** Checkpoint tại **Epoch 3 của Seed 123 (`checkpoint-1206` / `best_model`)** được chọn làm mô hình chính thức để phân tích lỗi, kiểm thử 10 câu ngoài và chạy Demo vì đạt Validation F1 Weighted cao nhất (`0.974280`).

### 1.2. Kết quả trên tập Test cả 3 Seed (2.140 mẫu: 1.020 Fake, 1.120 Real)

Đánh giá độc lập cả 3 seed trên tập Test (thang 0–1), trích xuất đầy đủ Báo cáo phân loại (Precision, Recall, F1 từng lớp) và Ma trận nhầm lẫn (Confusion Matrix).

| Seed | Split | Accuracy | F1 Weighted | Tổng số câu sai | Tỉ lệ lỗi | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| --- | --- | --- | --- | --- | --- | --- |
| **Seed 42** | Test | 0.973832 | 0.973824 | 56 / 2140 | 0.026168 | `[[986, 34], [22, 1098]]` |
| **Seed 123** | Test | 0.973364 | 0.973350 | 57 / 2140 | 0.026636 | `[[981, 39], [18, 1102]]` |
| **Seed 2024** | Test | **0.976168** | **0.976158** | **51 / 2140** | **0.023832** | `[[986, 34], [17, 1103]]` |
| **Mean ± Std** | **Test** | **0.974455 ± 0.001502** | **0.974444 ± 0.001503** | **54.67 / 2140** | **0.025545** | — |

**Chi tiết Classification Report (Seed 123 – Checkpoint chọn theo Validation):**
- **Fake (0):** Precision = `0.9820` | Recall = `0.9618` | F1-score = `0.9718` (Support: 1.020)
- **Real (1):** Precision = `0.9658` | Recall = `0.9839` | F1-score = `0.9748` (Support: 1.120)

---

## 2. Nhật ký Phân tích lỗi & Kiểm thử 10 câu ngoài (Error Analysis)

### 2.1. Phân tích lỗi trên tập Validation (Checkpoint Seed 123)

Sử dụng mô hình tốt nhất (**Seed 123**) để phân tích lỗi trên tập Validation, mô hình ghi nhận **55 / 2.140 câu dự đoán sai** (tỉ lệ lỗi `2.57%`), gồm **14** lỗi False Negative (Real đoán nhầm Fake – `25.45%`) và **41** lỗi False Positive (Fake đoán nhầm Real – `74.55%`). Toàn bộ danh sách lỗi kèm độ tin cậy Softmax được xuất ra file `val_errors.csv`.

Hai nhóm nguyên nhân chính gây nhầm lẫn:
- **Tin giả mang văn phong báo chí / chính thống (False Positives – `Fake=0` đoán nhầm `Real=1`):** Các câu tin giả sử dụng cấu trúc tường thuật học thuật, trịnh trọng hoặc chèn con số, `#hashtag`, link `t.co` khiến mô hình phân loại dựa trên bề mặt ngôn ngữ nhầm là tin chính thống với độ tin cậy cao (ví dụ Index 61 – `99.92%`, Index 145 – `99.55%`, Index 305 – `99.83%`).
- **Tin thật thiếu đặc trưng định dạng Twitter (False Negatives – `Real=1` đoán nhầm `Fake=0`):** Các thông báo hoặc khuyến cáo y tế viết bằng ngôn ngữ phổ thông, dạng hỏi đáp/phủ định hoặc thiếu đường dẫn `t.co` thường bị đẩy về nhãn `Fake` (ví dụ Index 13 – `99.85%`, Index 407 – `99.67%`, Index 613 – `96.03%`). Ngoài ra, `Validation Loss` tăng từ Epoch 1 (`0.0979`) lên Epoch 3 (`0.1336`) cho thấy mô hình có xu hướng tự tin quá mức (overconfident) ở các trường hợp khó (độ tin cậy ở các mẫu sai dao động `91.74%`–`99.92%`).

### 2.2. Thử nghiệm 10 câu ngoài dữ liệu (`external_10_sentences.csv`)

Sử dụng checkpoint **Seed 123** để kiểm thử trên bộ 10 câu tiếng Anh ngoài tập dữ liệu (5 câu `Real`, 5 câu `Fake`). Kết quả tổng hợp: **Accuracy = 0.5000** (`5/10` câu đúng) | **F1 Weighted = 0.3333** | **Confusion Matrix:** `[[5, 0], [5, 0]]`. Bảng kết quả chi tiết được lưu tại `external_10_results.csv`.

| ID | Câu thực tế ngoài tập dữ liệu | Nhãn thật | Dự đoán | Độ tin cậy | Kết quả |
| --- | --- | --- | --- | --- | --- |
| **01** | WHO recommends washing your hands frequently with soap and water to reduce the risk of infection. | `Real (1)` | `Fake (0)` | `99.63%` | **SAI (FN)** |
| **02** | Wearing a mask in crowded indoor places can help reduce the spread of COVID-19. | `Real (1)` | `Fake (0)` | `99.97%` | **SAI (FN)** |
| **03** | The health ministry reported 1,250 new confirmed COVID-19 cases and 14 deaths in the last 24 hours. | `Real (1)` | `Fake (0)` | `99.87%` | **SAI (FN)** |
| **04** | Vaccines are approved only after clinical trials show they are safe and effective. | `Real (1)` | `Fake (0)` | `99.89%` | **SAI (FN)** |
| **05** | People with symptoms such as fever, cough or loss of taste or smell should get tested and isolate. | `Real (1)` | `Fake (0)` | `98.74%` | **SAI (FN)** |
| **06** | Drinking bleach or disinfectant kills the coronavirus inside your body. | `Fake (0)` | `Fake (0)` | `99.98%` | **ĐÚNG (TN)** |
| **07** | 5G mobile towers spread the coronavirus, so they must be shut down immediately. | `Fake (0)` | `Fake (0)` | `99.98%` | **ĐÚNG (TN)** |
| **08** | Eating raw garlic every morning completely prevents COVID-19 infection. | `Fake (0)` | `Fake (0)` | `99.98%` | **ĐÚNG (TN)** |
| **09** | Breaking: the government is hiding a secret cure for coronavirus that works in 24 hours. | `Fake (0)` | `Fake (0)` | `99.98%` | **ĐÚNG (TN)** |
| **10** | COVID-19 vaccines contain microchips that let the government track everyone. | `Fake (0)` | `Fake (0)` | `99.98%` | **ĐÚNG (TN)** |

**Nhận xét kiểm thử ngoài miền dữ liệu:** Mô hình nhận diện đúng `5/5` câu `Fake` (`100%` Recall lớp Fake) nhưng đoán nhầm toàn bộ `5/5` câu `Real` thành `Fake` (`98.74%`–`99.97%`). Nguyên nhân cốt lõi là hiện tượng lệch miền dữ liệu (**Domain Shift / Shortcut Learning**): trong tập huấn luyện Twitter (`Constraint@AAAI2021`), các tin `Real` thường gắn kèm đặc trưng định dạng như đường dẫn `t.co`, `#hashtag` hoặc `@mention`. Khi gặp câu văn tiếng Anh chuẩn hoàn toàn sạch metadata, mô hình mất đi tín hiệu hình thức quen thuộc — thể hiện rõ nhất khi so sánh **Câu 03** (có số liệu ca nhiễm/tử vong nhưng không có URL → đoán nhầm `Fake` `99.87%`) với **câu Demo mục 3** (có số liệu tử vong kèm link `https://t.co/...` → đoán đúng `Real` `99.94%`).

---

## 3. Phần Demo (Nhập câu và trả nhãn trực tiếp trong Notebook)

Hàm suy luận `predict_fake_news(text)` nạp trực tiếp mô hình tốt nhất (`Seed 123`) đã lưu tại Google Drive (`MyDrive/BERT_FakeNews/best_model`), tự động đọc ánh xạ nhãn từ `best_model.config.id2label` (`0 -> Fake`, `1 -> Real`) và trả về kèm xác suất độ tin cậy (Softmax).

- **Đảm bảo đồng nhất nhãn (Label Consistency):** Giải quyết hoàn toàn rủi ro đảo nhãn khi tích hợp lên ứng dụng web Streamlit.
- **Ví dụ output thực tế:**
  - **Input:** `"States reported 1121 deaths a small rise from last Tuesday. Southern states reported 640 of those deaths. https://t.co/YASGRTT4ux"`
  - **Dự đoán:** `=> Kết quả dự đoán: Real (Tin thật) | Độ tin cậy: 99.94%`

---

## 4. Hướng dẫn chạy code và Minh chứng

- **Mã nguồn:** Toàn bộ quy trình (đọc dữ liệu từ Google Drive, Tokenizer, Train 3 Seed, Đánh giá Test, Phân tích lỗi Validation, Kiểm thử 10 câu CSV và Live Demo) được tích hợp hoàn chỉnh trong `Ly_Bert.ipynb`.
- **Cấu trúc thư mục trên Google Drive:**

    MyDrive/BERT_FakeNews/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── external_10_sentences.csv
    ├── val_errors.csv
    ├── external_10_results.csv
    ├── config_seed_123.zip
    └── best_model/        # Lưu trọng số + tokenizer của Seed 123 (không đẩy lên GitHub)

- **Cài đặt thư viện:** `pip install transformers datasets scikit-learn pandas numpy torch`
- **Chạy lại toàn bộ:** Mở `Ly_Bert.ipynb` trên Google Colab (GPU T4), đặt 4 file CSV vào `MyDrive/BERT_FakeNews/` và chạy lần lượt từ Cell 1 đến Cell 10.
- **Chạy Demo nhanh (không cần huấn luyện lại):** Chỉ cần chạy **Cell 1 → Cell 2 → Cell 8** (kiểm thử 10 câu CSV) hoặc **Cell 9** (nhập câu bất kỳ).
- **Trọng số mô hình (Checkpoint):** Tuân thủ quy tắc nhóm (không commit file nặng `model.safetensors`, `optimizer.pt`, `scheduler.pt`), mô hình đầy đủ được lưu trên Google Drive và file `config_seed_123.zip` (`config.json`, `training_args.bin`, `trainer_state.json`, `tokenizer.json`, `tokenizer_config.json`) được đính kèm làm minh chứng quá trình huấn luyện.
