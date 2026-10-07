# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Bùi Thị Thu Uyên |
| MSSV | 2A202602613 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/Ujandok/K4-L3L4-Track2-Day21-BuiThiThuUyen-2A202602613-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 4 | 200 | 0.05 | 3 | 0.7014 | 0.8740 |
| 5 | 300 | 0.1 | 4 | 0.7123 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ này có f1_score cao nhất (0.7149). Lần có accuracy cao nhất (lần 1) lại có F1 thấp hơn, cho thấy hai chỉ số có thể xếp hạng mô hình khác nhau và accuracy bị lớp đa số chi phối. Giảm learning_rate xuống 0.05 thì 200 cây (lần 4) vẫn kém 100 cây với 0.1 (lần 1), còn lần 2 bị underfit. Ba lần tốt nhất chỉ chênh khoảng 0.004, tức vài mẫu holdout.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu có thu nhập trên 50K, nên mô hình luôn trả lời "thu nhập thấp" vẫn đạt accuracy 0,752 dù không nhận ra người thu nhập cao nào. F1 của lớp dương kết hợp precision và recall của lớp thu nhập cao, nên đo đúng khả năng tìm ra lớp thiểu số; mô hình đoán bừa có F1 bằng 0. Lần chạy 2 có accuracy 0,846 nhưng F1 chỉ 0,605; khi push bộ tham số này, Quality Gate trên CI đã chặn đúng (`FAILED: f1_score 0.6051 < 0.65`, run 37638100244) và Release không chạy. Không dùng `average="weighted"` hay `"macro"` vì chúng cộng F1 rất cao của lớp đa số vào: với lần chạy 2, weighted F1 là 0,830 và macro F1 là 0,755, đều vượt ngưỡng 0,65.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Không cài được `requirements.txt`. | Python 3.14 không có bản build của các thư viện đã pin. | Tạo `.venv` bằng Python 3.11. |
| MLflow lỗi `ImportError` khi dùng `sqlite:///mlflow.db`. | pip cài SQLAlchemy 2.1, không tương thích mlflow 2.13. | Pin `sqlalchemy==2.0.30`. |
| `git push` không kích hoạt GitHub Actions. | Repo là fork nên GitHub tắt workflow kế thừa. | Bật workflow trong tab Actions; run Bước 2 được chạy bằng `workflow_dispatch`. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** F1 tăng 0,0205 nhưng chỉ tương ứng 3 người thu nhập cao được nhận đúng thêm trên 500 mẫu holdout. Hai batch cùng phân phối nên dữ liệu mới không mang thông tin mới, vì vậy đây là cải thiện nhỏ, không chứng minh thêm dữ liệu luôn tốt hơn. Điều quan trọng là pipeline tự chạy lại chỉ từ một commit dữ liệu.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: ngưỡng tốt nhất 0,30 cho F1 0,7368 so với 0,7149 ở 0,5; ghi vào `report.json` và MLflow.
- [x] Bonus 3 - Báo cáo precision / recall tự động: `outputs/detail.txt` (lớp thu nhập cao: precision 0,81, recall 0,64); khi tìm khách hàng thu nhập cao, bỏ sót tốn kém hơn gán nhầm.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: cảnh báo khi tỷ lệ lớp dương lệch quá 5 điểm so với 24,8%, ghi `positive_rate` vào `report.json`.
