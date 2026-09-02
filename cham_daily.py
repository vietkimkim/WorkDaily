"""
Điểm vào workflow — chạy 06:00 giờ Việt Nam mỗi ngày.

Luồng:
  1. Kiểm tra mật khẩu ứng dụng (dừng sớm nếu thiếu, khỏi chạy 20 phút rồi mới hỏng)
  2. Cập nhật kho dữ liệu (chỉ tải kỳ mới, ~40 giây)
  3. Tìm đài quay hôm nay, chạy 4 chạm cho ĐB · G1 · G8
  4. Gửi email

Chạy tay:  python cham_daily.py            → hôm nay
           python cham_daily.py 05.09.2026 → ngày khác
"""
import sys, traceback
from datetime import datetime, timedelta, timezone
import engine as E
import cham_engine as CH

VN = timezone(timedelta(hours=7))


def main():
    ngay = sys.argv[1] if len(sys.argv) > 1 else None
    hom_nay = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    print(f"=== {CH.SO_CHAM} CHẠM — {hom_nay:%d.%m.%Y} (giờ VN) ===\n")

    # --- Kiểm tra secret TRƯỚC khi làm gì tốn thời gian ---
    try:
        E._lay_mat_khau()
        print("  Mật khẩu ứng dụng: OK\n")
    except RuntimeError as e:
        print(f"  ✗ {e}")
        sys.exit(1)

    # --- Kho dữ liệu ---
    kho = E.doc_master()
    if kho is None:
        print("  Chưa có kho dữ liệu — quét lần đầu (~4 phút, chỉ 1 lần)...")
        E.tao_master(so_ky=200)
    else:
        print(f"  Kho dữ liệu: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
        E.cap_nhat_master(so_ky_moi=20)
    print()

    # --- Chạy 4 chạm ---
    CH.main(ngay=f"{hom_nay:%d.%m.%Y}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
