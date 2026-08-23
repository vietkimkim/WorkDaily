# XSMN — tự chạy 6h sáng, gửi email, không cần mở máy

Chạy trên GitHub Actions. iPhone tắt màn hình, đóng app, cất túi — vẫn chạy.

---

## Cài đặt: 3 bước, khoảng 10 phút

### Bước 1 — Tạo mật khẩu ứng dụng Gmail

Nếu đã có rồi thì bỏ qua. Nếu chưa:

1. `myaccount.google.com` → **Bảo mật**
2. Bật **Xác minh 2 bước** (bắt buộc, chưa bật thì không thấy mục sau)
3. Tìm **Mật khẩu ứng dụng** → tạo mới, tên `xsmn`
4. Google hiện chuỗi **16 chữ cái thường** dạng `abcd efgh ijkl mnop` — copy ngay

> Đây KHÔNG phải mật khẩu đăng nhập Gmail. Mật khẩu thường sẽ bị Gmail từ chối.

### Bước 2 — Upload code lên repo

Tạo repo mới trên GitHub (Public hay Private đều được), upload toàn bộ thư mục này.

**Kiểm tra bắt buộc:** repo phải thấy đủ `.github/workflows/daily.yml`.
Thư mục bắt đầu bằng dấu chấm hay bị Windows ẩn khi kéo thả. Nếu thiếu, bấm
**Add file → Create new file**, gõ đường dẫn `.github/workflows/daily.yml`
rồi dán nội dung vào.

### Bước 3 — Nạp 3 Secrets

Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Tên | Giá trị |
|---|---|
| `GMAIL_APP_PASSWORD` | Chuỗi 16 chữ cái ở Bước 1 |
| `MAIL_USER` | Gmail dùng để gửi |
| `MAIL_TO` | Địa chỉ nhận báo cáo |

### Xong — chạy thử

Tab **Actions** → **XSMN 6h sáng** → **Run workflow** → **Run workflow**.

Lần đầu mất ~20 phút (quét kho dữ liệu 200 kỳ × 36 đài, tính null cache).
Xong thì email về. Từ hôm sau tự chạy 6h sáng, mỗi lần ~3 phút.

---

## Vì sao lần sau nhanh hơn nhiều

Workflow **commit lại thư mục `data/`** sau mỗi lần chạy:

| File | Vai trò | Tính lại khi nào |
|---|---|---|
| `data/master.json` | Kho 200 kỳ × 36 đài | Chỉ tải phần mới mỗi ngày |
| `data/null_cache.json` | Phân phối null | Chỉ khi đổi `SO_KY` hoặc số con |
| `data/lich_quay.json` | Lịch quay 36 đài | Hiếm khi |

Nhờ vậy lần chạy thứ hai trở đi chỉ mất ~3 phút thay vì 20.

Thư mục `log/` lưu báo cáo từng ngày — mở ra xem lại bất cứ lúc nào.

---

## Cấu hình hiện tại

Sửa trong `engine.py`, phần **BẢNG ĐIỀU KHIỂN** ở đầu file:

```python
SO_KY       = 150     # số kỳ huấn luyện
SO_CON_DB   = 64      # ① Đề đặc biệt
SO_CON_LO2  = 0       # ② Bao lô 2 số (đang TẮT)
SO_CON_3SO  = 50      # ③ Lô 3 số
DIEM_DB     = 5       # điểm cược mỗi con
DIEM_3SO    = 1
```

**Vốn hiện tại: 1.170 điểm/đài Miền Nam** (64×5 + 17×50×1).

> Đổi `SO_KY` hoặc số con thì phải **xoá `data/null_cache.json`** rồi chạy lại.
> Quên bước này thì code dùng cache sai và p-value trở nên vô nghĩa.

---

## Giờ chạy

Cron đặt `0 23 * * *` (23:00 UTC = **06:00 giờ Việt Nam**).

Actions có thể trễ 5–30 phút vào giờ cao điểm. Không sao — dữ liệu dùng để dự
báo là kỳ quay **chiều hôm trước**, đã có sẵn từ tối.

---

## Khi có sự cố

| Hiện tượng | Xử lý |
|---|---|
| Bước "Kiểm tra truy cập nguồn" đỏ | IP GitHub bị `xskt.com.vn` chặn. Phải quay lại chạy Colab thủ công hoặc thuê VPS. |
| Báo thiếu mật khẩu | Sai tên Secret. Phải chính xác `GMAIL_APP_PASSWORD`, phân biệt hoa thường. |
| Email vào Spam | Lần đầu Gmail hay lọc nhầm thư tự gửi cho mình. Đánh dấu "Không phải spam". |
| Tab Actions trống | Thiếu `.github/workflows/daily.yml`. Xem lại Bước 2. |
| Cảnh báo `⚠ LỆCH` trong email | Trang nguồn đổi cấu trúc HTML. **Đừng dùng số** cho tới khi sửa parser. |
| `p HỌ = 0.000` | Null cache tạo với quá ít lượt. Xoá `data/null_cache.json`, chạy lại. |

---

## Đọc kết quả

Đầu email là khối vàng: **tổng vốn và kỳ vọng cả ngày**. Đó là con số duy nhất
nên dùng để đánh giá.

Với nhiều bộ số mỗi ngày, gần như ngày nào cũng có bộ trúng — nhưng kỳ vọng
toán học không đổi:

```
Đề / bao lô 2 số :  95/100  − 1 = −5,0%
Lô 3 số          : 961/1000 − 1 = −3,9%
```

Không phụ thuộc số con, không phụ thuộc thuật toán.

Cột `p HỌ` trong bảng đã hiệu chỉnh so sánh bội bằng min-p. Từ 0,05 trở lên
nghĩa là bộ số **không phân biệt được với chọn ngẫu nhiên**.

Muốn con số sạch hơn (không rò rỉ chọn trọng số), chạy tay:

```python
import engine as E
E.chay_backtest(chon_dai=19, so_ky=150)
```
