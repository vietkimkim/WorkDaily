"""
Job 20 con bao lô — chạy 06:05 giờ VN, ĐỘC LẬP với cham_daily.py.

Chạy tay:  python lo_daily.py            → hôm nay
           python lo_daily.py 15.09.2026 → ngày khác
"""
import sys, traceback
from datetime import datetime, timedelta, timezone
import engine as E
import lo_engine12 as L

VN = timezone(timedelta(hours=7))


def main():
    ngay = sys.argv[1] if len(sys.argv) > 1 else None
    hom_nay = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    print(f"=== {L.SO_CON} CON BAO LÔ — {hom_nay:%d.%m.%Y} (giờ VN) ===\n")
    try:
        E._lay_mat_khau()
        print("  Mật khẩu ứng dụng: OK\n")
    except RuntimeError as e:
        print(f"  ✗ {e}"); sys.exit(1)

    kho = E.doc_master()
    if kho is None:
        print("  Chưa có kho dữ liệu — quét lần đầu (~4 phút)...")
        E.tao_master(so_ky=200)
    else:
        print(f"  Kho dữ liệu: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
        E.cap_nhat_master(so_ky_moi=20)
    print()
    L.main(ngay=f"{hom_nay:%d.%m.%Y}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
