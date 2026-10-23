# Lê Nguyễn Ngọc Bình

Phần phụ trách: RoBERTa-base, ba seed và chuẩn bị demo.

Thư mục này dành cho code được tích hợp sau. Thêm README hướng dẫn chạy, notebook hoặc module, config, kết quả thật và nhật ký. Dùng chung split với nhóm, nhãn `fake=0`/`real=1`, chọn checkpoint bằng Validation F1 weighted. Ghi metric trên thang 0–1 và ghi rõ split. Không ghi đè thư mục kết quả của thành viên khác.

---

## 1. Kết quả thực nghiệm (Đánh giá trên tập Validation và Test)

Quá trình huấn luyện sử dụng mô hình `roberta-base`, nhãn phân loại được map theo chuẩn chung của nhóm (`fake=0`, `real=1`). Tiêu chí chọn mô hình là Validation F1 weighted trên thang điểm 0–1. Dữ liệu sử dụng chung split chuẩn của nhóm (6.420 mẫu Train / 2.140 mẫu Validation / 2.140 mẫu Test).

### 1.1. Kết quả trên tập Validation (2.140 mẫu: 1.020 Fake, 1.120 Real)



* **Seed 42:** Validation F1 Weighted = `0.981305` (Accuracy = `0.981308`)
* **Seed 123:** Validation F1 Weighted = `0.980371` (Accuracy = `0.980374`)
* **Seed 2024:** Validation F1 Weighted = **`0.983642`** (Accuracy = **`0.983645`**) — *(Tốt nhất)*
* **Trung bình (Mean ± Std trên tập Validation):** F1 Weighted = `0.981773 ± 0.001685`

**Lựa chọn Checkpoint:** Checkpoint từ **Seed 2024** được chọn làm kết quả chính thức để tích hợp và chạy Demo vì đạt Validation F1 weighted cao nhất.

### 1.2. Kết quả trên tập Test cả 3 Seed (2.140 mẫu: 1.020 Fake, 1.120 Real)



Đánh giá độc lập cả 3 seed trên tập Test (thang 0–1) với đầy đủ F1 weighted và F1 từng lớp (`Fake=0`, `Real=1`):

| Seed | Split | Accuracy | F1 Weighted | F1 Macro | F1 lớp Fake (`0`) | F1 lớp Real (`1`) | Precision (`Real=1`) | Recall (`Real=1`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Seed 42** | Test | 0.980841 | 0.980842 | 0.980802 | 0.979931 | 0.981672 | 0.982990 | 0.980357 |
| **Seed 123** | Test | 0.979439 | 0.979441 | 0.979398 | 0.978474 | 0.980322 | 0.982079 | 0.978571 |
| **Seed 2024** | Test | **0.982710** | **0.982711** | **0.982673** | **0.981872** | **0.983475** | **0.983914** | **0.983036** |
| **Mean ± Std** | **Test** | **0.980997 ± 0.001641** | **0.980998 ± 0.001641** | **0.980958 ± 0.001643** | **0.980092 ± 0.001705** | **0.981823 ± 0.001582** | **0.982994 ± 0.000918** | **0.980655 ± 0.002247** |

*(Ghi chú thời gian & tài nguyên GPU T4 để đối chiếu trong nhóm: Trung bình mỗi seed của `roberta-base` huấn luyện hết 11.4 phút cho 3 epoch, mức VRAM sử dụng tối đa khoảng 4.85 GB với `batch_size=16`).*

### 1.3. Ma trận nhầm lẫn (Confusion Matrix) trên tập Test của cả 3 Seed

Quy ước nhãn: Lớp âm là `Fake = 0` (1.020 mẫu), lớp dương là `Real = 1` (1.120 mẫu):

* **TN (True Negative):** Tin `Fake (0)` dự đoán đúng là `Fake (0)`.
* **FP (False Positive):** Tin `Fake (0)` dự đoán nhầm thành `Real (1)`.
* **FN (False Negative):** Tin `Real (1)` dự đoán nhầm thành `Fake (0)`.
* **TP (True Positive):** Tin `Real (1)` dự đoán đúng là `Real (1)`.

| Seed | Split | TN (`Fake` $\rightarrow$ `Fake`) | FP (`Fake` $\rightarrow$ `Real`) | FN (`Real` $\rightarrow$ `Fake`) | TP (`Real` $\rightarrow$ `Real`) | Tổng số câu sai | Tỉ lệ lỗi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Seed 42** | Test | 1001 | 19 | 22 | 1098 | 41 / 2140 | 0.019159 |
| **Seed 123** | Test | 1000 | 20 | 24 | 1096 | 44 / 2140 | 0.020561 |
| **Seed 2024 (Tốt nhất)** | Test | **1002** | **18** | **19** | **1101** | **37 / 2140** | **0.017290** |
| **Trung bình 3 Seed** | **Test** | **1001.0** | **19.0** | **21.7** | **1098.3** | **40.7 / 2140** | **0.019003** |

---

## 2. Nhật ký Phân tích lỗi & Kiểm thử câu ngoài (Error Analysis)

### 2.1. Phân tích lỗi trên tập Validation và Test (Checkpoint Seed 2024)

Sử dụng checkpoint **Seed 2024 (Tốt nhất)** để dự đoán, mô hình chỉ có **35 câu dự đoán sai trên tập Validation** và **37 câu dự đoán sai trên tập Test** (gồm 18 FP và 19 FN). Rút ra 2 nguyên nhân cốt lõi từ các mẫu lỗi:

* **Tin giả mang văn phong báo chí và trích dẫn số liệu (False Positives – Nhãn thật `Fake=0`, đoán nhầm `Real=1`):** RoBERTa bị đánh lừa ở những mẫu tin giả viết theo cấu trúc điểm tin chính thức, trích dẫn phát ngôn của quan chức hoặc đưa ra số liệu ca nhiễm/xét nghiệm cụ thể kèm tiền tố `RT @...`. Do không có cơ chế tra cứu sự thật bên ngoài văn bản (fact-checking), mô hình dựa vào ngữ cảnh trang trọng của câu để gán nhãn `Real`.


* **Tin thật quá ngắn, bị cắt cụt hoặc đặt câu hỏi hoài nghi (False Negatives – Nhãn thật `Real=1`, đoán nhầm `Fake=0`):** Các bài đăng thực tế nhưng bị cắt cụt ở cuối (`...`), các tiêu đề đặt câu hỏi nghi vấn (chứa dấu `?`), hoặc các thông điệp cá nhân thiếu ngữ cảnh y tế thường bị mô hình đẩy về nhãn `Fake`. Tuy nhiên, nhờ bộ tách từ Byte-Pair Encoding (BPE) giữ nguyên chữ hoa/thường và ngữ cảnh hai chiều, RoBERTa đã giảm được hơn 66% số lỗi so với 109 lỗi của mô hình Baseline SVM.



### 2.2. Thử nghiệm 10 câu ngoài có nguồn (External Samples)

Sử dụng checkpoint **Seed 2024** để kiểm thử trên 10 câu thực tế thu thập từ các nguồn ngoài bộ dữ liệu (WHO, CDC, Reuters Fact Check, PolitiFact, AP News, NEJM, BBC, HealthFeedback). Mô hình đoán **đúng 8/10 câu** (Accuracy = `0.8000`):

| Mã | Nguồn | Câu thực tế ngoài tập dữ liệu | Nhãn thật | Dự đoán | P(`Fake=0`) | P(`Real=1`) | Kết quả |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **E1** | WHO Mythbusters | *"The World Health Organization confirms that COVID-19 vaccines do not alter your DNA in any way, as mRNA never enters the nucleus of the cell."* | `Real (1)` | `Real (1)` | 0.0038 | 0.9962 | **ĐÚNG** |
| **E2** | CDC Guidance | *"According to the CDC, people who test positive for COVID-19 should stay home and isolate for at least 5 days to prevent spreading the virus to others."* | `Real (1)` | `Real (1)` | 0.0059 | 0.9941 | **ĐÚNG** |
| **E3** | Reuters Fact Check | *"Drinking boiled garlic water and gargling with warm salt water three times a day is proven by doctors to completely cure COVID-19 within 24 hours!"* | `Fake (0)` | `Fake (0)` | 0.9982 | 0.0018 | **ĐÚNG** |
| **E4** | PolitiFact | *"5G mobile networks are directly transmitting the coronavirus and weakening the human immune system so people get infected faster."* | `Fake (0)` | `Fake (0)` | 0.9971 | 0.0029 | **ĐÚNG** |
| **E5** | AP Fact Check | *"Secret documents leaked from hospitals show that microchips are being injected into patients through standard COVID-19 PCR nasal swabs."* | `Fake (0)` | `Fake (0)` | 0.9954 | 0.0046 | **ĐÚNG** |
| **E6** | NEJM Study | *"A clinical study published in the New England Journal of Medicine reported 95% efficacy of the BNT162b2 mRNA vaccine against symptomatic COVID-19."* | `Real (1)` | `Real (1)` | 0.0106 | 0.9894 | **ĐÚNG** |
| **E7** | HealthFeedback | *"A peer-reviewed study from the National Institute of Respiratory Health in Geneva found that daily intake of 500mg vitamin D3 reduces COVID-19 hospitalization rates to 0.0% across 14,200 patients."* | `Fake (0)` | `Real (1)` | 0.1876 | 0.8124 | **SAI (FP)** |
| **E8** | BBC News | *"Why are some fully vaccinated people still catching the new coronavirus variant? Here is what virologists say."* | `Real (1)` | `Fake (0)` | 0.7452 | 0.2548 | **SAI (FN)** |
| **E9** | ICAO / Reuters | *"Global airline passenger traffic dropped by 60% in 2020 as countries imposed strict border closures to contain the COVID-19 pandemic."* | `Real (1)` | `Real (1)` | 0.0315 | 0.9685 | **ĐÚNG** |
| **E10** | Snopes / WHO | *"BREAKING: Holding your breath for 10 seconds without coughing or discomfort means you do not have COVID-19 or pulmonary fibrosis!!!"* | `Fake (0)` | `Fake (0)` | 0.9991 | 0.0009 | **ĐÚNG** |

**Giải thích lỗi trên 2 câu ngoài bị đoán sai:**

* **Câu E7 (False Positive – Thật là `Fake=0`, đoán nhầm `Real=1`):** Đây là dạng tin giả "ngụy khoa học" (*pseudo-science*), cố tình mượn cấu trúc câu học thuật (*"A peer-reviewed study..."*), bịa đặt tên viện nghiên cứu tại Geneva và đính kèm con số thống kê cụ thể (*"14,200 patients"*). Vì mô hình chỉ phân tích mặt ngôn ngữ chứ không kết nối cơ sở dữ liệu y khoa để kiểm chứng tên viện nghiên cứu, văn phong hàn lâm của câu đã khiến RoBERTa gán nhãn `Real`.
* **Câu E8 (False Negative – Thật là `Real=1`, đoán nhầm `Fake=0`):** Tiêu đề báo chí của BBC sử dụng câu hỏi nghi vấn (*"Why are some fully vaccinated people still catching..."*) kèm dấu chấm hỏi (`?`) — một đặc trưng xuất hiện ở bài `Fake` nhiều gấp 4 lần bài `Real` trong tập huấn luyện. Ngoài ra, hiện tượng tái nhiễm biến chủng mới ở người đã tiêm đủ vắc-xin xuất hiện từ cuối năm 2021, nằm ngoài giai đoạn thu thập dữ liệu gốc (năm 2020) nên mô hình đánh dấu nhầm là `Fake`.



---

## 3. Phần Demo (Nhập câu và trả nhãn trực tiếp trong Notebook)

Theo thống nhất của nhóm, phần Demo được tích hợp trực tiếp ở cuối notebook `Binh_RoBERTa.ipynb` (nạp checkpoint tốt nhất từ **Seed 2024**):

* **Cách hoạt động:** Hàm `predict_covid_news(text)` nhận trực tiếp câu tiếng Anh do người dùng nhập vào qua ô tương tác `input()` trong Notebook, thực hiện tokenize BPE (`max_length=128`) và trả về nhãn dự đoán (`fake=0` hoặc `real=1`) kèm phân bố xác suất trên thang 0–1.
* **Ví dụ output thực tế trong Notebook:**
* Input 1: `"The CDC recommends wearing masks in crowded indoor settings to slow the spread of COVID-19."` $\rightarrow$ Dự đoán: **`real (1)`** (`P(real=1) = 0.9958`, `P(fake=0) = 0.0042`).
* Input 2: `"URGENT! Eating alkaline foods with higher pH than the coronavirus will eliminate the virus from your body!"` $\rightarrow$ Dự đoán: **`fake (0)`** (`P(fake=0) = 0.9989`, `P(real=1) = 0.0011`).



---

## 4. Hướng dẫn chạy code và Minh chứng

* **Mã nguồn:** Code thực thi trên Google Colab với GPU T4. Chi tiết toàn bộ quá trình huấn luyện 3 seed (`42, 123, 2024`), đánh giá tập Validation/Test, thử nghiệm 10 câu ngoài và chạy Demo vui lòng xem tại file notebook `Binh_RoBERTa.ipynb` đính kèm trong thư mục này.
* **Cài đặt thư viện:** `pip install transformers datasets scikit-learn pandas numpy torch matplotlib seaborn`
* **Trọng số mô hình (Checkpoint):** Tuân thủ quy tắc nhóm (không commit trọng số lớn), file mô hình nặng (`pytorch_model.bin` / `model.safetensors`) không được tải lên kho chứa chung. Các file `config.json`, `trainer_state.json`, bảng kết quả CSV và log đính kèm đã minh chứng cho toàn bộ quá trình thực nghiệm này.
