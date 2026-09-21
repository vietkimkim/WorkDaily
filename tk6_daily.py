"""
Job 6 engine thống kê — 64 con cho ĐB · G1 · G8.
  python tk6_daily.py              -> hôm nay
  python tk6_daily.py 28.09.2026   -> ngày khác
"""
import sys, traceback
from datetime import datetime, timedelta, timezone
import engine as E
import thongke6 as TK

VN = timezone(timedelta(hours=7))

if __name__ == "__main__":
    ngay = next((a for a in sys.argv[1:] if "." in a), None)
    hn = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    print(f"=== 6 ENGINE THỐNG KÊ — {hn:%d.%m.%Y} (giờ VN) ===\n")
    try:
        E._lay_mat_khau(); print("  Mật khẩu ứng dụng: OK\n")
        kho = E.doc_master()
        if kho is None:
            print("  Chưa có kho — quét lần đầu..."); E.tao_master(so_ky=200)
        else:
            print(f"  Kho: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
            E.cap_nhat_master(so_ky_moi=20)
        print()
        TK.main(ngay=f"{hn:%d.%m.%Y}")
    except Exception:
        traceback.print_exc(); sys.exit(1)
