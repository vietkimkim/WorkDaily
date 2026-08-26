"""
Chấm điểm dự báo — chạy 22h tối mỗi ngày (giờ VN).

Dò kết quả thật của các đài đã quay hôm nay, điền vào các bản ghi trong
theo_doi/YYYY-MM.jsonl. Chỉ ghi nhận, KHÔNG đưa ra khuyến nghị nào.
"""
import sys, traceback
from datetime import datetime, timedelta, timezone
import theo_doi

VN = timezone(timedelta(hours=7))

if __name__ == "__main__":
    ngay = None
    if len(sys.argv) > 1:
        import engine as E
        ngay = E.doc_ngay(sys.argv[1])
    hn = ngay or datetime.now(VN).date()
    print(f"=== CHẤM ĐIỂM {hn:%d.%m.%Y} (giờ VN) ===\n")
    try:
        theo_doi.cham_diem(hn)
        theo_doi.thong_ke_nhanh()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
