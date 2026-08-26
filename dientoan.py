"""
XỔ SỐ ĐIỆN TOÁN — nạp dữ liệu từ ảnh, kiểm tra, phân tích.

Cơ cấu 27 giải giống hệt Miền Bắc nên dùng lại toàn bộ engine.py.

QUY TRÌNH VỚI CLAUDE CODE
  1. Bỏ ảnh vào thư mục anh/
  2. Bảo Claude Code: "đọc ảnh trong anh/, ghi vào data/dientoan.json theo
     đúng schema trong dientoan.py, rồi chạy python dientoan.py"
  3. Script tự kiểm tra từng kỳ, loại kỳ sai, rồi phân tích

SCHEMA data/dientoan.json
{
  "ten": "Điện toán 3 phút",
  "ky": [
    {
      "ma": "20260824-0372",
      "giai": ["83639","45666","62139","86361","07120","57870","55534","25065",
               "38485","35857","2736","5466","3823","8405","8241","1800","0862",
               "1896","4096","4864","641","005","286","75","87","11","62"],
      "bang_loto": {"0":[0,5], "1":[1], "2":[0,3], "3":[4,6,9], "4":[1],
                    "5":[7], "6":[1,2,4,5,6], "7":[0,5], "8":[5,6,7], "9":[6]}
    }
  ]
}

  giai      : 27 số, THỨ TỰ ĐB -> G1 -> G2 -> G3 -> G4 -> G5 -> G6 -> G7
  bang_loto : bảng Chục/Đơn vị ở cuối ảnh. KHÔNG BẮT BUỘC nhưng RẤT NÊN CÓ —
              nó là checksum độc lập, bắt được mọi lỗi đọc số.
"""
import json, os, sys
import numpy as np
from scipy import stats
import engine as E

FILE_DL = "data/dientoan.json"
CO_CAU  = E.CO_CAU_GIAI["MB"]          # 27 số, ĐB 5 chữ số — trùng điện toán


# ==============================================================================
#  KIỂM TRA TỪNG KỲ
# ==============================================================================

def kiem_tra_ky(giai, bang_loto=None):
    """Trả list lỗi. Rỗng = kỳ hợp lệ."""
    loi = []
    tong = sum(q for q, _ in CO_CAU.values())
    if len(giai) != tong:
        loi.append(f"có {len(giai)} số, cần {tong}")
        return loi
    if any(not str(x).isdigit() for x in giai):
        loi.append("có phần tử không phải chữ số")
        return loi

    # --- Chữ ký độ dài ---
    chuan = {}
    for q, d in CO_CAU.values():
        chuan[d] = chuan.get(d, 0) + q
    thuc = {}
    for x in giai:
        thuc[len(x)] = thuc.get(len(x), 0) + 1
    if thuc != chuan:
        loi.append(f"phân bố độ dài {dict(sorted(thuc.items()))} "
                   f"khác chuẩn {dict(sorted(chuan.items()))}")

    # --- Checksum bảng Chục/Đơn vị ---
    if bang_loto:
        tinh = {}
        for x in giai:
            tinh.setdefault(int(x[-2]), set()).add(int(x[-1]))
        for d in range(10):
            a = sorted(tinh.get(d, set()))
            b = sorted(int(v) for v in bang_loto.get(str(d), []))
            if a != b:
                loi.append(f"bảng loto hàng {d}: tính ra {a}, ảnh ghi {b}")
    return loi


def nap_dientoan(path=FILE_DL, im_lang=False):
    """Đọc file, kiểm tra mọi kỳ, trả (danh sách kỳ hợp lệ, danh sách lỗi)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Chưa có {path}. Bảo Claude Code đọc ảnh trong anh/ và tạo file này "
            f"theo schema ghi ở đầu dientoan.py.")
    d = json.load(open(path))
    ok, hong = [], []
    for i, k in enumerate(d["ky"]):
        loi = kiem_tra_ky(k["giai"], k.get("bang_loto"))
        if loi:
            hong.append((i, k.get("ma", f"#{i}"), loi))
        else:
            ok.append(k["giai"])
    if not im_lang:
        co_cs = sum(1 for k in d["ky"] if k.get("bang_loto"))
        print(f"  Nạp {path}: {len(d['ky'])} kỳ")
        print(f"    Hợp lệ      : {len(ok)}")
        print(f"    Có checksum : {co_cs}/{len(d['ky'])}"
              + ("" if co_cs == len(d['ky']) else "   ⚠ kỳ thiếu checksum có thể sai mà không biết"))
        if hong:
            print(f"    ✗ LỖI {len(hong)} kỳ — ĐỌC LẠI ẢNH:")
            for i, ma, loi in hong[:8]:
                print(f"        [{ma}] " + " | ".join(loi))
            if len(hong) > 8:
                print(f"        ... và {len(hong)-8} kỳ nữa")
    return ok, hong


# ==============================================================================
#  PHÂN TÍCH
# ==============================================================================

def chay_dientoan(K=20, path=FILE_DL, min_train=None, n_null=40):
    """Chạy module BAO LÔ 2 SỐ trên dữ liệu điện toán."""
    tg, hong = nap_dientoan(path)
    n = len(tg)
    if n < 30:
        raise RuntimeError(f"chỉ có {n} kỳ hợp lệ, cần tối thiểu 30")
    n_lo = len(tg[0])
    pool = [[int(s[-2:]) for s in ky] for ky in tg]
    min_train = min_train or max(15, n // 3)
    cap = E.tinh_cap(K)

    # ---------- [A] CHẨN ĐOÁN: chi-square trên 100 ô ----------
    dem = np.zeros(100)
    for ky in pool:
        for v in ky:
            dem[v] += 1
    N = int(dem.sum())
    print(f"\n[A] CHẨN ĐOÁN — {n} kỳ × {n_lo} lô = {N:,} quan sát")
    print("-" * 74)
    r = stats.chisquare(dem, f_exp=np.full(100, N / 100))
    hop_le = N / 100 >= 5
    print(f"  Chi-square 100 ô : chi2={r.statistic:7.2f}  p={r.pvalue:.4f}")
    print(f"  Tần số kỳ vọng   : {N/100:.1f}/ô  → kiểm định "
          f"{'HỢP LỆ ✓' if hop_le else 'VÔ HIỆU (cần >=5)'}")
    if hop_le:
        print(f"  → {'LỆCH khỏi phân phối đều — ĐÁNG CHÚ Ý' if r.pvalue < .05 else 'Không bác bỏ được giả thuyết RNG công bằng'}")
    ctx = E.dung_ctx(pool)
    print(f"  λ Jelinek-Mercer : {ctx['lam']}   (ước lượng từ held-out)")
    print(f"  Hệ số co ngót B  : {ctx['B_js']:.4f}  → "
          + ("chênh lệch giữa các ô CHỈ LÀ NHIỄU ĐẾM"
             if ctx["B_js"] < 0.05 else "có cấu trúc vượt mức nhiễu"))

    # ---------- [B] BACKTEST ----------
    wins = E.cua_so(tg, pool, min_train, None)
    w, hr, h, per = E.toi_uu(wins, K, cap)
    nt = len(wins); tong = nt * n_lo; p0 = K / 100.
    lo_, hi_ = E.wilson_ci(h, tong)
    print(f"\n[B] BACKTEST — {nt} kỳ test × {n_lo} lô = {tong:,} quan sát")
    print("-" * 74)
    print("  Trọng số bật: " + (", ".join(nm for nm, x in zip(E.SIG_NAMES, w) if x > 0) or "(không)"))
    print(f"  Hit rate {h}/{tong:,} = {hr:.3%}  |  ngẫu nhiên {p0:.3%}  |  chênh {hr-p0:+.3%}")
    print(f"  KTC95 [{lo_:.3%}, {hi_:.3%}]")

    # ---------- [C] NULL CONTROL ----------
    print(f"\n[C] NULL CONTROL — {n_null} lượt trên RNG công bằng mô phỏng")
    print("-" * 74)
    null = E.null_ho_v8(n_null, n, CO_CAU, [("LO2", "2so", None, K, cap)],
                        min_train, None)
    p_tho, p_bon, p_ho = E.p_min_ho(null, [hr])
    print(f"  Nhiễu đạt: TB {null[:,0].mean():.3%} | trung vị {np.median(null[:,0]):.3%} | "
          f"P90 {np.percentile(null[:,0],90):.3%} | max {null[:,0].max():.3%}")
    print(f"  → p-value = {p_ho[0]:.4f}  → " +
          ("VƯỢT ngưỡng nhiễu — đáng theo dõi thêm"
           if p_ho[0] < .05 else "NẰM TRONG biên độ nhiễu, không hơn chọn ngẫu nhiên"))

    # ---------- [D] BỘ SỐ ----------
    Zf, _ = E.ma_tran_tin_hieu(tg, pool)
    sel, cu = E.select_top(w @ Zf, K, cap)
    so = sorted(f"{i:02d}" for i in sel)
    print("\n\n" + "=" * 74)
    print(f"  BỘ {K} CON LÔ 2 SỐ — XỔ SỐ ĐIỆN TOÁN")
    print(f"  {n} kỳ huấn luyện | {n_lo} lô/kỳ | p-value {p_ho[0]:.3f}")
    print("=" * 74 + "\n")
    for i in range(0, len(so), 10):
        print("   " + "  ".join(so[i:i+10]))
    print("\n  " + ",".join(so))
    print("=" * 74)
    return so


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    chay_dientoan(K)
