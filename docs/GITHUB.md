# Đưa dự án lên GitHub và làm việc nhóm

## Lần đầu — Quân

1. Giải nén, mở đúng thư mục `covid19-fake-news-nlp` trong VS Code.
2. Trên GitHub tạo repository trống, ví dụ `covid19-fake-news-nlp`. Không thêm README tự động vì gói đã có.
3. Mở Terminal của VS Code rồi chạy:

```bash
git init
git add .
git status
git commit -m "Khoi tao do an NLP va pipeline DistilBERT"
git branch -M main
git remote add origin https://github.com/TEN_TAI_KHOAN/covid19-fake-news-nlp.git
git push -u origin main
```

Thay `TEN_TAI_KHOAN` bằng tài khoản thật. Nếu Git yêu cầu danh tính, đặt `git config user.name` và `git config user.email` bằng thông tin của bạn. Đăng nhập bằng luồng GitHub/VS Code; không đưa token vào code. Trước commit, xem `git status` và xác nhận không có dữ liệu riêng, checkpoint lớn hoặc thông tin bí mật.

## Mời các thành viên

Chủ repository thêm các bạn trong Settings → Collaborators. Mỗi bạn clone repo về, tạo nhánh riêng rồi bổ sung phần được giao:

```bash
git clone https://github.com/TEN_TAI_KHOAN/covid19-fake-news-nlp.git
cd covid19-fake-news-nlp
git switch -c feature/bert-ly
# Sửa code, chạy kiểm tra, lưu kết quả của mình
git add .
git commit -m "Them pipeline BERT va huong dan chay"
git push -u origin feature/bert-ly
```

Mở Pull Request vào `main`, nhờ một thành viên xem rồi merge. Nhánh đề xuất: `feature/baseline-huy`, `feature/distilbert-quan`, `feature/bert-ly`, `feature/roberta-binh`.

## Kết quả và trọng số

Commit config, JSON metric, log gọn, bảng tổng hợp và notebook cần thiết. Trọng số trong `model/`, checkpoint và ZIP đã bị `.gitignore` loại trừ. Lưu chúng ở nơi tải phù hợp như output Kaggle, ghi link phiên bản cụ thể trong `docs/MODEL_DOWNLOAD.md` để người khác chạy demo. Kiểm tra quyền truy cập link bằng tài khoản khác trước lúc nộp.

GitHub chứa mã nguồn; để chạy giao diện hiện tại, người xem tải code và checkpoint rồi chạy Streamlit trên máy. Gói này chưa triển khai một đường link ứng dụng online.

Trước khi sửa tiếp:

```bash
git switch main
git pull origin main
git switch -c feature/ten-cong-viec-moi
```

Không dùng force push vào nhánh chung. Chỉ giải quyết conflict sau khi đọc thay đổi của cả hai bên.
