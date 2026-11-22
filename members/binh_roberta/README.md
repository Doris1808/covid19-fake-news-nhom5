# Lê Nguyễn Ngọc Bình — Nhánh RoBERTa-base & Demo

**Phần phụ trách:** Fine-tune `roberta-base` (5 seed), kiểm tra rò rỉ dữ liệu, đánh giá đa tiêu chí (gồm chi phí & độ trễ), phân tích lỗi tự động (FP/FN), thiết kế thí nghiệm cải tiến dựa trên lỗi, kiểm thử 10 câu ngoài và chuẩn bị Demo.

**Quy ước đồng bộ toàn nhóm:**
* **Dữ liệu cố định (`CONSTRAINT@AAAI2021`):** `6.420 Train` (3.060 fake, 3.360 real) / `2.140 Validation` (1.020 fake, 1.120 real) / `2.140 Test` (1.020 fake, 1.120 real).
* **Ánh xạ nhãn thống nhất:** `fake = 0`, `real = 1` — lưu trực tiếp vào `config.json` (`id2label = {0: "fake", 1: "real"}`, `label2id = {"fake": 0, "real": 1}`) và đọc trực tiếp từ `model.config.id2label[pred_id]` ở bước Test và Demo.
* **Nguyên tắc chống rò rỉ:** Chọn `best_model_checkpoint`, chọn `BEST_SEED`, phân tích lỗi và chỉnh ngưỡng phân loại hoàn toàn trên tập `Validation`; tập `Test` chỉ dùng để chấm điểm cuối cùng.
* **Thang đo:** Toàn bộ chỉ số ghi trên thang `0–1`, ghi rõ `split` và đồng bộ 100% với các tệp CSV/JSON trong gói minh chứng.

---

## 1. Khảo sát dữ liệu đa góc độ & Kiểm tra rò rỉ dữ liệu
*(Nguồn đối chiếu: `data_leakage_report.json` và `eda_data_issues_train.csv` — Cell 2)*

* **Kiểm tra chất lượng & rò rỉ (`Train` vs `Val` / `Test`):**
  * Giá trị thiếu (missing): `<0>`
  * Số dòng trùng nguyên văn (`exact overlap`): Train–Val = `<...>`, Train–Test = `<...>`
  * Số dòng trùng sau chuẩn hóa (chữ thường, bỏ link rút gọn, bỏ dấu câu): Train–Val = `<...>`, Train–Test = `<...>`
  * Số mẫu tập Test sạch sau khi loại dòng trùng với Train (`test_clean`): `<...>` / 2.140 mẫu.
* **Các vấn đề hình thức phát hiện trong dữ liệu (`Train`):**
  * Hai lớp chênh lệch mạnh về lối tắt bề mặt mạng xã hội: tin `real` có tỉ lệ chứa URL, hashtag, `@mention`, tiền tố `RT` và chữ số cao hơn hẳn tin `fake`; ngược lại tin `fake` ngắn hơn, xuất hiện nhiều dấu `?`, `!` và xuống dòng bất thường.
  * Dữ liệu còn chứa thực thể HTML (`&amp;`, `&quot;`, ...) và ký tự đặc biệt non-ASCII/emoji — đây là cơ sở để phân tích lỗi và thiết kế bước làm sạch ở Mục 4–5.

---

## 2. Thiết lập thực nghiệm 5 Seed & Chỉ số cho bài toán Nhị phân (2 lớp)
*(Nguồn đối chiếu: `roberta_val_5seeds_summary.csv`, `roberta_epoch_progression_5seeds.csv`, `seed_*_logs/training_args.json` — Cell 1 & Cell 3)*

### 2.1. Cấu hình huấn luyện & Lý do chạy 5 Seed
* **Kiến trúc:** `roberta-base` (12 tầng Encoder, hidden = 768, 12 attention heads, Byte-level BPE giữ nguyên chữ hoa/thường, `max_length = 128`).
* **Siêu tham số cố định:** `epochs = 3`, `batch_size = 16`, `learning_rate = 1.5e-5`, `lr_scheduler_type = "cosine"`, `warmup_steps = 72`, `weight_decay = 0.02`, `fp16 = True`, `metric_for_best_model = "f1_weighted"`.
* **5 Seed độc lập (`42, 123, 2024, 2025, 2026`):** Mở rộng từ 3 seed lên 5 seed để ước lượng `Mean ± Sample Std` (`ddof=1`) đáng tin cậy hơn, kiểm chứng độ ổn định của quá trình hội tụ thay vì dựa vào ít lượt chạy.

### 2.2. Lựa chọn chỉ số cho bài toán Phân loại Nhị phân (2 lớp)
* `Macro` và `Micro` là cách gom nhóm thường dùng trong **phân loại đa lớp**. Với bài toán **2 lớp đơn nhãn**, `Micro-F1` luôn trùng với `Accuracy` nên không mang thêm thông tin.
* Do đó, báo cáo sử dụng bộ chỉ số chuẩn cho phân loại nhị phân: **Binary Precision, Recall, F1 cho lớp mục tiêu cần phát hiện (`fake = 0`)**, kết hợp **F1 lớp `real = 1`** và **ROC-AUC**.
* Đồng thời giữ thêm **`F1 weighted`** để đối chiếu trực tiếp với mốc bài báo gốc Patwa et al. (2021) (`Val = 0.9346`, `Test = 0.9332`) và các thành viên khác trong nhóm.

### 2.3. Kết quả Validation & Chọn Seed đại diện (`split = val`, 2.140 mẫu)

| Seed | Val Accuracy | Val F1 Weighted | Val F1 Fake (`0`) | Val ROC-AUC | Thời gian Train (s) | Peak VRAM (GB) | Best Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<checkpoint-...>` |
| **123** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<checkpoint-...>` |
| **2024** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<checkpoint-...>` |
| **2025** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<checkpoint-...>` |
| **2026** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<checkpoint-...>` |
| **Mean ± Std (`ddof=1`)** | `<... ± ...>` | `<... ± ...>` | `<... ± ...>` | `<... ± ...>` | `<...>` | `<...>` | **Best Seed: `<BEST_SEED>`** |

---

## 3. Kết quả Tập Test & Đánh giá thực tiễn 5 tiêu chí so với Baseline
*(Nguồn đối chiếu: `roberta_test_5seeds_summary.csv`, `roberta_confusion_matrices_5seeds.png`, `practical_5criteria_vs_baseline.csv` — Cell 4)*

### 3.1. Kết quả trên tập Test gốc (2.140 mẫu) và Test sạch rò rỉ (`split = test`)

| Seed | Acc Test | F1 Weighted Test | Binary F1 Fake (`0`) | Prec Fake (`0`) | Rec Fake (`0`) | ROC-AUC | F1w Test sạch rò rỉ | TN (`0→0`) | FP (`0→1`) | FN (`1→0`) | TP (`1→1`) | Tổng lỗi | Latency (ms/câu) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| **123** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| **2024** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| **2025** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| **2026** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| **Mean ± Std** | `<... ± ...>` | `<... ± ...>` | `<... ± ...>` | `<... ± ...>` | `<... ± ...>` | `<... ± ...>` | `<... ± ...>` | — | — | — | — | `<...>` | `<... ± ...>` |

*(Ký hiệu ma trận nhầm lẫn đồng bộ với slide nhóm: lấy `Real = 1` làm lớp dương; `FP` là tin giả bị bỏ lọt thành tin thật, `FN` là tin thật bị báo nhầm thành tin giả).*

### 3.2. So sánh 5 tiêu chí thực tiễn với Baseline SVM & Vai trò của Baseline

| Mô hình | 1. Chất lượng (`F1w Test`) | 2. Độ ổn định (`Std`) | 3. Chi phí huấn luyện | 4. Tài nguyên (`VRAM / Disk`) | 5. Độ trễ suy luận (`batch=1`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bài báo gốc (SVM)** | `0.9332` (`Val: 0.9346`) | — | CPU | Nhẹ | Rất thấp |
| **Baseline SVM + TF-IDF + stats** | `0.949069` (109 lỗi) | `0.0000` (tất định) | `~0.02 giây` (CPU) | CPU / `~5 MB` | `< 0.5 ms/câu` |
| **RoBERTa-base (5 seeds)** | `<... ± ...>` (`<...>` lỗi) | `<...>` | `<... giây/seed>` (`<GPU>`) | `<... GB>` / `<... MB>` | `<... ms/câu>` |

**Tại sao cần Baseline trước khi chạy Transformer?**
1. **Đo mức tăng ròng thực sự:** Nếu bỏ thẳng dữ liệu vào RoBERTa mà không có Baseline SVM (`0.949069`, chạy 0.02s trên CPU), ta không biết chi phí GPU và độ trễ cao hơn gấp nhiều lần có mang lại cải thiện đáng kể hay không.
2. **Phát hiện sớm lỗi dữ liệu & lối tắt:** Mô hình tuyến tính giúp soi trực tiếp trọng số từ khóa, phát hiện sớm bài toán phụ thuộc vào URL, `RT`, hashtag trước khi huấn luyện các mô hình hộp đen.
3. **Cân đối triển khai thực tế:** Hệ thống ưu tiên tốc độ trên CPU sẽ cân nhắc SVM; hệ thống chấp nhận GPU để cắt giảm mạnh số lỗi phân loại sẽ chọn RoBERTa.

---

## 4. Phân tích lỗi tự động từ FP & FN trên tập Validation
*(Nguồn đối chiếu: `val_errors_categorized_seed<BEST_SEED>.csv`, `error_analysis_groups_val.csv`, `error_groups_distribution.png` — Cell 5)*

Thay vì chỉ đọc rời rạc thủ công, Cell 5 trích xuất toàn bộ các mẫu **False Positive (`Fake=0 → Real=1`)** và **False Negative (`Real=1 → Fake=0`)** của `BEST_SEED` trên tập Validation, tự động phân vào 7 nhóm lỗi, đếm số lượng và áp dụng quy tắc quyết định (**> 3 câu mắc lỗi $\rightarrow$ ưu tiên thiết kế thí nghiệm cải tiến**):

* **Tổng số lỗi trên Validation (`BEST_SEED`):** `<...>` / 2.140 mẫu (`FP = <...>`, `FN = <...>`).

| Nhóm lỗi tự động | Số câu mắc phải | Tỉ lệ (%) | Nguyên nhân cốt lõi | Quyết định |
| :--- | :---: | :---: | :--- | :---: |
| `1_Ky_tu_dac_biet_Emoji_HTML` | `<...>` | `<...>%` | Nhiễu thực thể HTML (`&amp;`, `&quot;`), ký tự non-ASCII/emoji | `<...>` |
| `2_Cau_ngan_hoac_bi_cat_cut` | `<...>` | `<...>%` | Câu quá ngắn (≤ 12 từ) hoặc bị cắt cụt (`...`) thiếu ngữ cảnh | `<...>` |
| `3_Cau_chua_phu_dinh` | `<...>` | `<...>%` | Tin thật chứa từ phủ định bác bỏ tin đồn (mang từ khóa của lớp Fake) | `<...>` |
| `4_Cau_hoi_hoac_cam_than` | `<...>` | `<...>%` | Dấu `?`, `!` phân bố lệch về lớp Fake trong dữ liệu gốc | `<...>` |
| `5_Tin_gia_giong_bao_chi_co_URL_so_lieu` | `<...>` | `<...>%` | Tin giả giả dạng văn phong báo chí, kèm URL, `@mention`, `RT`, số liệu | `<...>` |
| `6_Tin_that_viet_tron_thieu_URL_hashtag` | `<...>` | `<...>%` | Tin thật viết trơn, không có URL/hashtag nên bị đẩy về Fake | `<...>` |
| `7_Loi_ngu_nghia_hoac_can_factcheck_ngoai` | `<...>` | `<...>%` | Cần kiểm chứng tri thức thực tế ngoài văn bản | `<...>` |

---

## 5. Thiết kế thí nghiệm cải tiến mô hình dựa trên lỗi (Trước vs Sau)
*(Nguồn đối chiếu: `improvement_ablation_comparison.csv` — Cell 6)*

Dựa trực tiếp vào các nhóm lỗi chi phối ở Mục 4 (nhiễu HTML/xuống dòng, lệch ngưỡng FP/FN và phụ thuộc lối tắt URL/`RT`/`@mention`/`#`), thực hiện 2 thí nghiệm cải tiến trên cùng `BEST_SEED`:
* **Cải tiến A — Threshold Tuning:** Quét ngưỡng $\tau \in [0.10, 0.90]$ cho $P(\text{real})$ trên tập `Validation` để tìm $\tau^*$ tối ưu (không cần train lại).
* **Cải tiến B — Clean HTML + Shortcut Dropout Augmentation:** Giải mã thực thể HTML, bỏ xuống dòng nhiễu, đồng thời bổ sung 40% mẫu tập Train đã loại bỏ lối tắt bề mặt (`RT`, URL, `@mention`, ký tự `#`) để buộc RoBERTa học ngữ nghĩa thay vì học vẹt quy tắc *"thiếu URL/hashtag = Fake"*.

| Phiên bản thực nghiệm (`Seed <BEST_SEED>`) | Test Accuracy | Test F1 Weighted | FP (`Fake → Real`) | FN (`Real → Fake`) | Tổng lỗi Test |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. RoBERTa gốc (`tau = 0.50`)** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| **2. Cải tiến A: Threshold Tuning (`tau = <best_tau>`)** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| **3. Cải tiến B: Clean HTML + Shortcut Dropout** | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |

---

## 6. Kiểm thử 10 câu ngoài (`external_10_sentences.csv`) & Phần Demo
*(Nguồn đối chiếu: `external_10_sentences_comparison.csv` — Cell 7)*

### 6.1. So sánh RoBERTa gốc vs RoBERTa cải tiến trên 10 câu ngoài viết trơn
* **RoBERTa gốc (`BEST_SEED`):** Đúng `<...>` / 10 câu.
* **RoBERTa cải tiến (Shortcut Dropout):** Đúng `<...>` / 10 câu.

| ID | Nhãn chuẩn | Câu kiểm thử (`external_10_sentences.csv`) | Gốc đoán (`P_real`) | Cải tiến đoán (`P_real`) |
| :-: | :-: | :--- | :-: | :-: |
| 1 | `real` | WHO recommends washing your hands frequently with soap and water to reduce the risk of infection. | `<...>` (`<...>`) | `<...>` (`<...>`) |
| 2 | `real` | Wearing a mask in crowded indoor places can help reduce the spread of COVID-19. | `<...>` (`<...>`) | `<...>` (`<...>`) |
| 3 | `real` | The health ministry reported 1,250 new confirmed COVID-19 cases and 14 deaths in the last 24 hours. | `<...>` (`<...>`) | `<...>` (`<...>`) |
| 4 | `real` | Vaccines are approved only after clinical trials show they are safe and effective. | `<...>` (`<...>`) | `<...>` (`<...>`) |
| 5 | `real` | People with symptoms such as fever, cough or loss of taste or smell should get tested and isolate. | `<...>` (`<...>`) | `<...>` (`<...>`) |
| 6 | `fake` | Drinking bleach or disinfectant kills the coronavirus inside your body. | `<...>` (`<...>`) | `<...>` (`<...>`) |
| 7 | `fake` | 5G mobile towers spread the coronavirus, so they must be shut down immediately. | `<...>` (`<...>`) | `<...>` (`<...>`) |
| 8 | `fake` | Eating raw garlic every morning completely prevents COVID-19 infection. | `<...>` (`<...>`) | `<...>` (`<...>`) |
| 9 | `fake` | Breaking: the government is hiding a secret cure for coronavirus that works in 24 hours. | `<...>` (`<...>`) | `<...>` (`<...>`) |
| 10 | `fake` | COVID-19 vaccines contain microchips that let the government track everyone. | `<...>` (`<...>`) | `<...>` (`<...>`) |

### 6.2. Demo nhập câu và trả nhãn trong Notebook
* Sử dụng hàm `predict_covid_news(text)` ở Cell 7 của `Binh_RoBERTa_Complete.ipynb`.
* Tự động chọn phiên bản tốt hơn trên bộ 10 câu ngoài làm mô hình Demo (`ACTIVE_DEMO_MODEL`), tiền xử lý nhẹ bằng `clean_special_noise()`, đọc nhãn trực tiếp qua `model.config.id2label[pred_id]` và trả về nhãn (`FAKE` / `REAL`) kèm xác suất `P(fake=0)` và `P(real=1)`.

---

## 7. Hướng dẫn chạy lại & Danh mục Minh chứng
* **Môi trường:** Google Colab GPU T4 (`transformers`, `datasets`, `evaluate`, `scikit-learn`, `accelerate`, `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`).
* **Cách chạy:** Mở `Binh_RoBERTa_Complete.ipynb` trên Google Colab (chọn GPU T4), chạy tuần tự từ **Cell 1 $\rightarrow$ Cell 8**. Cell 8 tự động nén và tải về tệp `binh_roberta_github_final.zip`.
* **Danh mục tệp minh chứng trong thư mục (không commit trọng số lớn `.safetensors`/`.bin`):**
  * `Binh_RoBERTa_Complete.ipynb` — Notebook hoàn chỉnh kèm output thực chạy.
  * `data_leakage_report.json` & `eda_data_issues_train.csv` — Báo cáo kiểm tra rò rỉ và khảo sát dữ liệu.
  * `roberta_val_5seeds_summary.csv`, `roberta_epoch_progression_5seeds.csv`, `roberta_learning_curves_5seeds.png` — Minh chứng huấn luyện 5 seed.
  * `roberta_test_5seeds_summary.csv`, `roberta_confusion_matrices_5seeds.png`, `practical_5criteria_vs_baseline.csv` — Minh chứng kết quả Test, 5 tiêu chí và độ trễ.
  * `error_analysis_groups_val.csv`, `val_errors_categorized_seed<BEST_SEED>.csv`, `error_groups_distribution.png` — Minh chứng phân tích lỗi tự động từ FP/FN.
  * `improvement_ablation_comparison.csv` — Bảng so sánh Trước vs Sau cải tiến.
  * `external_10_sentences.csv` & `external_10_sentences_comparison.csv` — Kết quả kiểm thử 10 câu ngoài.
  * `seed_42_logs/` … `seed_2026_logs/` & `improved_seed_<BEST_SEED>_logs/` — Chứa `config.json`, `training_args.json`, `tokenizer_config.json`, `special_tokens_map.json` và `trainer_state.json`.
