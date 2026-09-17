"""
Job 12 engine thống kê.
  python de_daily.py         -> 64 con ĐỀ (ĐB · G1 · G8)
  python de_daily.py LO      -> 20 con BAO LÔ (MN/MT)
  python de_daily.py DE 20.09.2026
"""
import sys, traceback
from datetime import datetime, timedelta, timezone
import engine as E
import de_engine12 as D

VN = timezone(timedelta(hours=7))

if __name__ == "__main__":
    args = sys.argv[1:]
    che_do = args[0].upper() if args and args[0].upper() in ("DE", "LO") else "DE"
    ngay = next((a for a in args if "." in a), None)
    hn = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    print(f"=== 12 ENGINE THỐNG KÊ — {che_do} — {hn:%d.%m.%Y} (giờ VN) ===\n")
    try:
        E._lay_mat_khau(); print("  Mật khẩu ứng dụng: OK\n")
        kho = E.doc_master()
        if kho is None:
            print("  Chưa có kho — quét lần đầu..."); E.tao_master(so_ky=200)
        else:
            print(f"  Kho: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
            E.cap_nhat_master(so_ky_moi=20)
        print()
        D.main(ngay=f"{hn:%d.%m.%Y}", che_do=che_do)
    except Exception:
        traceback.print_exc(); sys.exit(1)
