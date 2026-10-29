# Lê Nguyễn Ngọc Bình

Phần phụ trách: RoBERTa-base, ba seed và chuẩn bị demo.

Thư mục này dành cho code và kết quả thực nghiệm của nhánh `RoBERTa-base`. Bao gồm README hướng dẫn chạy, notebook thực thi, config, kết quả thật (CSV/PNG) và nhật ký phân tích lỗi. Dùng chung split với nhóm (`6.420 Train / 2.140 Validation / 2.140 Test`), nhãn thống nhất `fake=0`/`real=1`, chọn checkpoint bằng `Validation F1 weighted`. Ghi metric trên thang `0–1` và ghi rõ split. Không ghi đè thư mục kết quả của thành viên khác.

---

## 1. Kết quả thực nghiệm (Đánh giá trên tập Validation và Test)
Quá trình huấn luyện sử dụng mô hình `roberta-base` (12-layer, 768-hidden, 12-heads, Byte-level BPE tokenizer, `max_length=128`). Nhãn phân loại được map xuyên suốt theo chuẩn chung của nhóm (`fake=0`, `real=1`) ở cả bước Train, Validation, Test và Demo. Cấu hình huấn luyện sử dụng `learning_rate=1.5e-5`, `lr_scheduler_type="cosine"`, `warmup_steps=72`, `weight_decay=0.02`, `batch_size=16` trong 3 epoch giúp Loss giảm đều và F1 tăng đều. Tiêu chí chọn mô hình là `Validation F1 weighted` trên thang điểm `0–1`.

### 1.1. Diễn biến huấn luyện và Kết quả trên tập Validation (`split = val`, 2.140 mẫu: 1.020 Fake, 1.120 Real)
*   **Seed 42:** Epoch 1 (`0.968210`) $\rightarrow$ Epoch 2 (`0.976631`) $\rightarrow$ Epoch 3: **Validation F1 Weighted = `0.981305`** (Accuracy = `0.981308`, Val Loss = `0.054100`)
*   **Seed 123:** Epoch 1 (`0.966342`) $\rightarrow$ Epoch 2 (`0.975228`) $\rightarrow$ Epoch 3: **Validation F1 Weighted = `0.980371`** (Accuracy = `0.980374`, Val Loss = `0.056800`)
*   **Seed 2024:** Epoch 1 (`0.971019`) $\rightarrow$ Epoch 2 (`0.978968`) $\rightarrow$ Epoch 3: **Validation F1 Weighted = `0.983642`** (Accuracy = `0.983645`, Val Loss = `0.048900`) — *(Tốt nhất)*
*   **Trung bình Validation 3 Seed (Mean ± Std):** F1 Weighted = **`0.981773 ± 0.001685`**

**Lựa chọn Checkpoint:** Checkpoint từ **Seed 2024** (`best_model_checkpoint` tại Epoch 3) được chọn làm kết quả chính thức để tích hợp, kiểm thử câu ngoài và chạy Demo vì đạt `Validation F1 weighted` cao nhất.

### 1.2. Kết quả đánh giá trên tập Test cả 3 Seed (`split = test`, 2.140 mẫu: 1.020 Fake, 1.120 Real)
Đánh giá độc lập `best_model_checkpoint` của cả 3 seed trên tập Test (thang `0–1`) với đầy đủ `F1 weighted`, `F1 macro` và F1 từng lớp (`fake=0`, `real=1`):

| Seed | Split | Accuracy | F1 Weighted | F1 Macro | F1 lớp Fake (`0`) | F1 lớp Real (`1`) | Precision (`Fake=0`) | Recall (`Fake=0`) | Precision (`Real=1`) | Recall (`Real=1`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | `test` | 0.980841 | 0.980842 | 0.980802 | 0.979931 | 0.981672 | 0.978495 | 0.981373 | 0.982990 | 0.980357 |
| **Seed 123** | `test` | 0.979439 | 0.979441 | 0.979398 | 0.978474 | 0.980322 | 0.976562 | 0.980392 | 0.982079 | 0.978571 |
| **Seed 2024** | `test` | **0.982710** | **0.982711** | **0.982673** | **0.981872** | **0.983475** | **0.981391** | **0.982353** | **0.983914** | **0.983036** |
| **Mean ± Std** | **`test`** | **0.980997 ± 0.001641** | **0.980998 ± 0.001641** | **0.980958 ± 0.001643** | **0.980092 ± 0.001705** | **0.981823 ± 0.001582** | **0.978816 ± 0.002431** | **0.981373 ± 0.000980** | **0.982994 ± 0.000918** | **0.980655 ± 0.002247** |

*(Thống kê tài nguyên bổ sung trên GPU Colab T4: Thời gian train trung bình `~684 giây` (~11.4 phút) / seed cho 3 epoch; mức VRAM cấp phát tối đa đạt `~4.85 GB` với `batch_size=16`, `fp16=True`).*

### 1.3. Ma trận nhầm lẫn (Confusion Matrix) trên tập Test cả 3 Seed (`labels = [0, 1]`)
Theo quy ước `fake = 0` (1.020 mẫu) và `real = 1` (1.120 mẫu):
*   **TN (`Fake` $\rightarrow$ `Fake`):** Tin giả (`0`) được đoán đúng là giả (`0`).
*   **FP (`Fake` $\rightarrow$ `Real`):** Tin giả (`0`) bị đoán nhầm thành thật (`1`) — *Bỏ sót tin giả*.
*   **FN (`Real` $\rightarrow$ `Fake`):** Tin thật (`1`) bị đoán nhầm thành giả (`0`) — *Báo oan tin thật*.
*   **TP (`Real` $\rightarrow$ `Real`):** Tin thật (`1`) được đoán đúng là thật (`1`).

| Seed | Split | TN (`0` $\rightarrow$ `0`) | FP (`0` $\rightarrow$ `1`) | FN (`1` $\rightarrow$ `0`) | TP (`1` $\rightarrow$ `1`) | Tổng số lỗi | Tỉ lệ lỗi |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | `test` | 1001 | 19 | 22 | 1098 | 41 / 2140 | 0.019159 |
| **Seed 123** | `test` | 1000 | 20 | 24 | 1096 | 44 / 2140 | 0.020561 |
| **Seed 2024 (Tốt nhất)** | `test` | **1002** | **18** | **19** | **1101** | **37 / 2140** | **0.017290** |
| **Trung bình 3 Seed** | **`test`** | **1001.0** | **19.0** | **21.7** | **1098.3** | **40.7 / 2140** | **0.019003** |

---

## 2. Nhật ký Phân tích lỗi & Kiểm thử câu ngoài (Error Analysis)

### 2.1. Phân tích lỗi nội bộ trên tập Validation và Test (Checkpoint Seed 2024)
Sử dụng checkpoint **Seed 2024 (Tốt nhất)** để dự đoán, mô hình chỉ có **35 câu sai trên tập Validation** và **37 câu sai trên tập Test** (18 FP và 19 FN, giảm 66.1% lỗi so với 109 lỗi của mô hình Baseline SVM). Rút ra 2 nguyên nhân cốt lõi từ file `roberta_test_errors_seed2024.csv`:
*   **Tin giả "đội lốt" điểm tin báo chí và số liệu thống kê (`Fake=0` đoán nhầm thành `Real=1` – FP):** RoBERTa bị đánh lừa bởi các bài tin giả được viết với văn phong trung lập, trích dẫn phát ngôn của quan chức hoặc đưa con số thống kê xét nghiệm/ca nhiễm kèm tiền tố `RT @...` và đường dẫn rút gọn. Do mô hình chỉ học biểu diễn ngôn ngữ chứ không tra cứu cơ sở dữ liệu sự thật bên ngoài (external fact-checking), cấu trúc trang trọng của câu khiến mô hình gán nhãn `real (1)`.
*   **Tin thật mang văn phong đời thường, câu hỏi nghi vấn hoặc bị cắt cụt (`Real=1` đoán nhầm thành `Fake=0` – FN):** Các bài đăng thực tế nhưng rất ngắn, bị cắt cụt ở cuối (`...`), dùng nhiều cảm thán/emoji, hoặc đặt câu hỏi hoài nghi (chứa dấu `?`) thường bị đẩy về nhãn `fake (0)` do mang hình thức bề mặt giống với tin đồn mạng xã hội trong tập huấn luyện.

### 2.2. Kiểm thử 10 câu ngoài có nguồn (External Samples)
Sử dụng checkpoint **Seed 2024** để dự đoán trên 10 câu thực tế bên ngoài bộ dữ liệu, có nguồn kiểm chứng rõ ràng (WHO, CDC, Reuters Fact Check, PolitiFact, AP News, NEJM, BBC, HealthFeedback). Mô hình dự đoán **đúng 8/10 câu** (`Accuracy = 0.8000`):

| Mã | Nguồn tham chiếu | Câu tiếng Anh thực tế ngoài bộ dữ liệu | Nhãn chuẩn | Dự đoán | `P(fake=0)` | `P(real=1)` | Đánh giá |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **E1** | WHO Mythbusters | *"The World Health Organization confirms that COVID-19 vaccines do not alter your DNA in any way, as mRNA never enters the nucleus of the cell."* | `real (1)` | `real (1)` | 0.0038 | 0.9962 | **ĐÚNG** |
| **E2** | CDC Guidance | *"According to the CDC, people who test positive for COVID-19 should stay home and isolate for at least 5 days to prevent spreading the virus to others."* | `real (1)` | `real (1)` | 0.0059 | 0.9941 | **ĐÚNG** |
| **E3** | Reuters Fact Check | *"Drinking boiled garlic water and gargling with warm salt water three times a day is proven by doctors to completely cure COVID-19 within 24 hours!"* | `fake (0)` | `fake (0)` | 0.9982 | 0.0018 | **ĐÚNG** |
| **E4** | PolitiFact | *"5G mobile networks are directly transmitting the coronavirus and weakening the human immune system so people get infected faster."* | `fake (0)` | `fake (0)` | 0.9971 | 0.0029 | **ĐÚNG** |
| **E5** | AP Fact Check | *"Secret documents leaked from hospitals show that microchips are being injected into patients through standard COVID-19 PCR nasal swabs."* | `fake (0)` | `fake (0)` | 0.9954 | 0.0046 | **ĐÚNG** |
| **E6** | NEJM Study | *"A clinical study published in the New England Journal of Medicine reported 95% efficacy of the BNT162b2 mRNA vaccine against symptomatic COVID-19."* | `real (1)` | `real (1)` | 0.0106 | 0.9894 | **ĐÚNG** |
| **E7** | HealthFeedback | *"A peer-reviewed study from the National Institute of Respiratory Health in Geneva found that daily intake of 500mg vitamin D3 reduces COVID-19 hospitalization rates to 0.0% across 14,200 patients."* | `fake (0)` | `real (1)` | 0.1876 | 0.8124 | **SAI (FP)** |
| **E8** | BBC News | *"Why are some fully vaccinated people still catching the new coronavirus variant? Here is what virologists say."* | `real (1)` | `fake (0)` | 0.7452 | 0.2548 | **SAI (FN)** |
| **E9** | ICAO / Reuters | *"Global airline passenger traffic dropped by 60% in 2020 as countries imposed strict border closures to contain the COVID-19 pandemic."* | `real (1)` | `real (1)` | 0.0315 | 0.9685 | **ĐÚNG** |
| **E10** | Snopes / WHO | *"BREAKING: Holding your breath for 10 seconds without coughing or discomfort means you do not have COVID-19 or pulmonary fibrosis!!!"* | `fake (0)` | `fake (0)` | 0.9991 | 0.0009 | **ĐÚNG** |

**Giải thích nguyên nhân 2 câu ngoài dự đoán sai:**
*   **Câu E7 (`fake=0` đoán nhầm thành `real=1`):** Đây là câu tin giả "ngụy khoa học" (*pseudo-science*), mượn cấu trúc bài báo học thuật (*"A peer-reviewed study..."*), gán tên viện nghiên cứu tại Geneva và đưa số liệu cụ thể (*"14,200 patients"*). Vì trong tập Train các câu báo cáo số liệu nghiên cứu gắn chặt với nhãn `real`, đồng thời mô hình không có khả năng tra cứu ngoài văn bản để biết viện nghiên cứu và con số `0.0%` là hư cấu, nên bị đánh lừa bởi văn phong học thuật.
*   **Câu E8 (`real=1` đoán nhầm thành `fake=0`):** Tiêu đề báo chí của BBC đặt câu hỏi nghi vấn về hiệu quả vắc-xin trước biến chủng mới (có dấu `?` – đặc trưng xuất hiện ở bài `fake` nhiều gấp 4 lần bài `real` trong dữ liệu gốc) và thiếu ngữ cảnh giải thích phía sau. Hơn nữa, khái niệm tái nhiễm biến chủng mới ở người đã tiêm đủ vắc-xin xuất hiện từ cuối năm 2021, lệch thời điểm so với dữ liệu huấn luyện năm 2020 (*temporal shift*).

---

## 3. Phần Demo theo phân công (Nhập câu và trả nhãn trong Notebook)
Theo thống nhất chung của nhóm, phần Demo được xây dựng trực tiếp trong notebook `Binh_RoBERTa.ipynb` thông qua hàm `predict_covid_news(text)`:
*   **Cơ chế:** Nạp trực tiếp `best_model_checkpoint` (**Seed 2024**) cùng cấu hình `id2label = {0: 'fake', 1: 'real'}`. Người dùng nhập một bài đăng tiếng Anh bất kỳ, mô hình tách token BPE (`max_length=128`), tính xác suất Softmax và in ra nhãn dự đoán kèm phân bố xác suất `P(fake=0)` và `P(real=1)`.
*   **Minh họa kết quả chạy thực tế trong Notebook:**
    *   *Câu thử 1:* `"The World Health Organization states that COVID-19 is caused by the SARS-CoV-2 virus and spreads mainly between people who are in close contact."`
        $\rightarrow$ Dự đoán: **`REAL (Tin thật - nhãn 1)`** (`P(fake=0) = 0.0031` | `P(real=1) = 0.9969`).
    *   *Câu thử 2:* `"BREAKING: Spraying chlorine or alcohol all over your body and drinking hot lemon water every hour will completely kill the coronavirus inside your lungs!"`
        $\rightarrow$ Dự đoán: **`FAKE (Tin giả - nhãn 0)`** (`P(fake=0) = 0.9992` | `P(real=1) = 0.0008`).

---

## 4. Hướng dẫn chạy code và Danh mục Minh chứng
*   **Mã nguồn:** Thực thi trên Google Colab với GPU Tesla T4. Toàn bộ quy trình huấn luyện 3 seed (`42, 123, 2024`), đánh giá Validation/Test, thử 10 câu ngoài và chạy Demo nằm trong file `Binh_RoBERTa.ipynb`.
*   **Cài đặt thư viện:**
    ```bash
    pip install transformers datasets evaluate scikit-learn accelerate pandas numpy torch matplotlib seaborn
    ```
*   **Các tệp minh chứng kết quả trong thư mục:**
    *   `Binh_RoBERTa.ipynb`: Notebook mã nguồn và output thực chạy.
    *   `roberta_val_3seeds_summary.csv` & `roberta_epoch_progression_3seeds.csv`: Bảng kết quả Validation và lịch sử hội tụ qua 3 Epoch của 3 Seed.
    *   `roberta_test_3seeds_summary.csv`: Bảng kết quả Test đầy đủ (Accuracy, F1 weighted, F1 macro, F1 từng lớp, TN, FP, FN, TP) của 3 Seed.
    *   `roberta_test_errors_seed2024.csv`: Danh sách toàn bộ các mẫu dự đoán sai trên tập Test của Seed 2024.
    *   `roberta_external_10_samples.csv`: Kết quả dự đoán và xác suất trên 10 câu thực tế có nguồn bên ngoài.
    *   `roberta_learning_curves_3seeds.png` & `roberta_confusion_matrices_3seeds.png`: Biểu đồ đường cong học và Ma trận nhầm lẫn của 3 Seed.
    *   `seed_42_logs/`, `seed_123_logs/`, `seed_2024_logs/`: Chứa `config.json`, `tokenizer_config.json` và `trainer_state.json` minh chứng quá trình huấn luyện (tuân thủ quy tắc nhóm không commit file trọng số lớn `model.safetensors` / `pytorch_model.bin`).
