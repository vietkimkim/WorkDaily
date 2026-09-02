# 4 Chạm — tự chạy 6h sáng, gửi email

M��i sáng 6h, workflow tự tìm các đài xổ số quay hôm đó, chạy phương pháp cầu chạm
cho **Đề Đặc Biệt · Đề Giải Nhất · Đề Đầu (G8)**, rồi gửi email.

Không cần mở máy, không cần trình duyệt.

---

## Danh sách file — đủ 6 file là chạy được

```
├── .github/workflows/cham_daily.yml   ← workflow, cron 6h sáng
├── engine.py                          ← hạ tầng: bóc dữ liệu, lịch quay, kho, email
├── cham_engine.py                     ← phương pháp 4 chạm
├── cham_daily.py                      ← điểm vào
├── requirements.txt
├── .gitignore
└── data/                              ← kho dữ liệu (code tự tạo)
```

`engine.py` chứa cả phần thống kê cũ không dùng đến — **không xoá**, vì
`cham_engine.py` cần 13 hàm từ nó: bóc dữ liệu, lịch quay 36 đài, kho dữ liệu,
cắt dữ liệu chống rò rỉ, và lấy mật khẩu email.

---

## Cài đặt: 3 bước

### 1. Upload code

Tạo repo mới, upload 5 file gốc.

**Workflow phải tạo tay** — Windows ẩn thư mục bắt đầu bằng dấu chấm nên kéo thả
sẽ mất. Làm thế này:
- **Add file** → **Create new file**
- Ô tên file gõ đủ: `.github/workflows/cham_daily.yml`
- Dán nội dung → **Commit changes**

### 2. Nạp 3 Secrets

**Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Tên (gõ chính xác) | Giá trị |
|---|---|
| `GMAIL_APP_PASSWORD` | Mật khẩu **ứng dụng** Gmail, 16 chữ cái thường |
| `MAIL_USER` | Gmail dùng để gửi |
| `MAIL_TO` | Địa chỉ nhận |

> Mật khẩu ứng dụng KHÔNG phải mật khẩu đăng nhập Gmail. Tạo tại:
> Tài khoản Google → Bảo mật → bật Xác minh 2 bước → Mật khẩu ứng dụng.

### 3. Chạy thử

**Actions** → **4 Chạm 6h sáng** → **Run workflow**

Lần đầu ~10 phút (quét kho 200 kỳ × 36 đài). Các lần sau ~4 phút.

---

## Bảng điều khiển — sửa trong `cham_engine.py`

```python
SO_KY_CHAM   = 150    # số kỳ lịch sử
SO_CHAM      = 4      # 3 chạm phủ 51 số · 4 phủ 64 · 5 phủ 75
CHAY_DB      = True   # ① Đề Đặc Biệt
CHAY_G1      = True   # ② Đề Giải Nhất
CHAY_G8      = True   # ③ Đề Đầu (G8) — chỉ MN/MT, Miền Bắc tự bỏ qua

CON_LAI_6TH  = 0.10   # kỳ cách 6 THÁNG còn 10% trọng số
NGUON_CAU    = "GON"  # "GON" = 13 vị trí (ĐB+G1+G8), ít overfit
                      # "DAY" = 82-107 vị trí, nhiều ứng viên hơn
TOP_CAU      = 5      # số cầu tốt nhất được bỏ phiếu
```

Không cần xoá cache khi đổi — phương pháp chạm tính null control trực tiếp.

---

## Phương pháp

**Cầu chạm:** một vị trí chữ số cố định (ví dụ "G1 chữ số 3"). Chữ số tại vị trí
đó ở kỳ t làm chạm dự báo cho kỳ t+1. Trúng nếu nó là 1 trong 2 chữ số mục tiêu.

**Năm tín hiệu**, trọng số do walk-forward chọn trong 31 tổ hợp:

| | Tín hiệu | Nội dung |
|---|---|---|
| C1 | TanSuatW | Tần suất trúng, trọng số giảm dần theo thời gian |
| C2 | ChuoiNay | Chuỗi đang chạy |
| C3 | ChuoiMaxW | Chuỗi dài nhất, chuỗi cũ bị hạ điểm |
| C4 | DeuDan | Độ đều đặn — khoảng cách giữa các lần trúng ít biến động |
| C5 | MoHinh | Mô hình xác suất phân phối chạm |

**Trọng số thời gian** đặt theo mốc 6 tháng, tự tính theo nhịp quay của đài:

| Đài | Nửa đời | 6 tháng còn | 1 năm còn |
|---|---|---|---|
| Quay tuần | 7,8 kỳ | 10% | 1,0% |
| Hàng ngày | 54,8 kỳ | 10% | 1,0% |

---

## Đọc kết quả — theo đúng thứ tự này

**① `p cấu trúc cầu`** — con số quan trọng nhất.

Đây là **một** phép kiểm định cho mỗi giải, không phải 107 → không bị vấn đề so
sánh bội. Nó trả lời: giữa chữ số kỳ trước và chạm kỳ sau có liên hệ nào không?

| p | Ý nghĩa |
|---|---|
| ≥ 0,05 | Mọi vị trí hành xử như nhau. **Bộ chạm ngang bốc bừa** |
| < 0,05 | Có bằng chứng cấu trúc cầu — ghi lại, theo dõi xem có lặp lại |

**② `Hit rate ngoài mẫu`** — walk-forward thật, mỗi kỳ chọn chạm chỉ từ dữ liệu
trước đó. Mốc ngẫu nhiên 64%, hoà vốn 67,37% ở tỷ lệ trả 95.

**③ `Độ ổn định`** — bootstrap: bỏ 20% số kỳ, chạy lại, xem chạm còn được chọn không.

> Độ ổn định đo **độ vững của việc chọn**, KHÔNG phải xác suất trúng.
> 4 chạm phủ đúng 64/100 số → P(trúng) = 64% trừ khi máy quay lệch thật.
> Ổn định 90% mà p ≥ 0,05 nghĩa là thuật toán đang **nhất quán chọn nhiễu**.

---

## Khi có sự cố

| Hiện tượng | Xử lý |
|---|---|
| Actions trống | File .yml chưa vào đúng `.github/workflows/` |
| Báo thiếu mật khẩu | Sai tên Secret, phải chính xác `GMAIL_APP_PASSWORD` |
| Bước "Kiểm tra truy cập nguồn" đỏ | IP GitHub bị trang nguồn chặn |
| Email vào Spam | Đánh dấu "Không phải spam" lần đầu |
| `⚠ CHỈ TÌM ĐƯỢC N ĐÀI` | Lịch quay thiếu — xoá `data/lich_quay.json`, chạy lại |
| Đài lỗi lẻ tẻ | Xem lý do trong email, thường do trang nguồn đổi cấu trúc |
