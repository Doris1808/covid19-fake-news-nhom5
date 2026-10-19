# Đưa dự án lên GitHub và làm việc nhóm

## Repository nhóm 5

Repository: https://github.com/Doris1808/covid19-fake-news-nhom5

Chủ repository: Doris1808. Phụ trách DistilBERT: Lương Hoàng Bảo Quân — 24DH111520.

Repo đã có mã nguồn trên nhánh `main`. Máy Mac của Quân đã kết nối Git và push thành công, không cần khởi tạo lại hoặc cấu hình lại email để tiếp tục.

Sau khi sửa code trong thư mục dự án trên Mac:

```bash
git status
git add .
git commit -m "Cap nhat phan DistilBERT nhom 5"
git push origin main
```

Xem danh sách file trong `git status` trước khi commit. Thư mục `.venv`, trọng số và checkpoint đã được loại khỏi Git bằng `.gitignore`.

Nếu tải code trên máy mới:

```bash
git clone https://github.com/Doris1808/covid19-fake-news-nhom5.git
cd covid19-fake-news-nhom5
```

## Mời các thành viên

Chủ repository thêm các bạn trong Settings → Collaborators. Mỗi bạn clone repo về, tạo nhánh riêng rồi bổ sung phần được giao:

```bash
git clone https://github.com/Doris1808/covid19-fake-news-nhom5.git
cd covid19-fake-news-nhom5
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
git switch -c feature/distilbert-quan
```

Không dùng force push vào nhánh chung. Chỉ giải quyết conflict sau khi đọc thay đổi của cả hai bên.
