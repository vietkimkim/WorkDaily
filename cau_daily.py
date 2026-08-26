"""
Job soi cầu — chạy 06:00 giờ Việt Nam mỗi ngày, ĐỘC LẬP với daily.py.

Chạy tay:  python cau_daily.py            → hôm nay
           python cau_daily.py 25.08.2026 → ngày khác
"""
import sys, traceback
from datetime import datetime, timedelta, timezone
import engine as E
import cau_engine as C

VN = timezone(timedelta(hours=7))


def main():
    ngay = sys.argv[1] if len(sys.argv) > 1 else None
    hom_nay = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    print(f"=== SOI CẦU — {hom_nay:%d.%m.%Y} (giờ VN) ===\n")

    try:
        E._lay_mat_khau()
        print("  Mật khẩu ứng dụng: OK\n")
    except RuntimeError as e:
        print(f"  ✗ {e}")
        sys.exit(1)

    kho = E.doc_master()
    if kho:
        print(f"  Kho dữ liệu: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
        E.cap_nhat_master(so_ky_moi=20)
    else:
        print("  Chưa có kho dữ liệu — tải trực tiếp từ web (chậm hơn).")
    print()
    C.main(ngay=f"{hom_nay:%d.%m.%Y}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
