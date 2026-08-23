"""
Điểm vào cho GitHub Actions. Chạy 06:00 giờ Việt Nam mỗi ngày.

Luồng:
  1. Cập nhật kho dữ liệu (chỉ tải kỳ mới, ~40 giây)
  2. Chạy dự báo cho tất cả đài quay hôm nay
  3. Gửi email
  4. Workflow commit lại data/ để lần sau chạy nhanh

Chạy tay:  python daily.py            → hôm nay
           python daily.py 25.08.2026 → ngày khác
"""
import os, sys, traceback
from datetime import datetime, timedelta, timezone

import engine as E

VN = timezone(timedelta(hours=7))


def main():
    ngay = sys.argv[1] if len(sys.argv) > 1 else None
    hom_nay = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    print(f"=== XSMN GitHub Actions — {hom_nay:%d.%m.%Y} (giờ VN) ===\n")

    # --- Kiểm tra secret trước khi làm gì tốn thời gian ---
    try:
        E._lay_mat_khau()
        print("  Mật khẩu ứng dụng: OK\n")
    except RuntimeError as e:
        print(f"  ✗ {e}")
        sys.exit(1)

    # --- Bước 1: kho dữ liệu ---
    kho = E.doc_master()
    if kho is None:
        print("  Chưa có kho dữ liệu — quét lần đầu (~4 phút, chỉ 1 lần)...")
        E.tao_master(so_ky=200)
    else:
        print(f"  Kho dữ liệu: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
        print("  Đang tải các kỳ mới...")
        E.cap_nhat_master(so_ky_moi=30)

    # --- Bước 2 + 3: dự báo và gửi email ---
    print()
    E.main(ngay=f"{hom_nay:%d.%m.%Y}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
