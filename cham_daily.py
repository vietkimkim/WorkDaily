"""
Job 4 chạm — chạy 06:20 giờ Việt Nam, ĐỘC LẬP với daily.py và cau_daily.py.

Chạy tay:  python cham_daily.py            → hôm nay
           python cham_daily.py 26.08.2026 → ngày khác
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
    try:
        E._lay_mat_khau()
        print("  Mật khẩu ứng dụng: OK\n")
    except RuntimeError as e:
        print(f"  ✗ {e}"); sys.exit(1)

    kho = E.doc_master()
    if kho:
        print(f"  Kho dữ liệu: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
        E.cap_nhat_master(so_ky_moi=20)
    else:
        print("  Chưa có kho dữ liệu — tải trực tiếp từ web.")
    print()
    CH.main(ngay=f"{hom_nay:%d.%m.%Y}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
