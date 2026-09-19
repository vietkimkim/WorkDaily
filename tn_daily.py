"""
Job engine THÍCH NGHI — chạy 6h sáng.
Mỗi lần chạy, engine tự đúc kết kinh nghiệm và lưu vào data/trang_thai_hedge.json.

  python tn_daily.py                 -> hôm nay, mọi giải
  python tn_daily.py 25.09.2026      -> ngày khác
"""
import sys, traceback
from datetime import datetime, timedelta, timezone
import engine as E
import engine_thich_nghi as TN

VN = timezone(timedelta(hours=7))

if __name__ == "__main__":
    ngay = next((a for a in sys.argv[1:] if "." in a), None)
    hn = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    print(f"=== ENGINE THÍCH NGHI — {hn:%d.%m.%Y} (giờ VN) ===\n")
    try:
        E._lay_mat_khau(); print("  Mật khẩu ứng dụng: OK\n")
        kho = E.doc_master()
        if kho is None:
            print("  Chưa có kho — quét lần đầu..."); E.tao_master(so_ky=200)
        else:
            print(f"  Kho: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
            E.cap_nhat_master(so_ky_moi=20)
        print()
        TN.main(ngay=f"{hn:%d.%m.%Y}")
    except Exception:
        traceback.print_exc(); sys.exit(1)
