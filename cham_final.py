# ==============================================================================
#  4 CHẠM — BẢN HỢP NHẤT CUỐI  |  ĐB · Giải Nhất · Giải 8
#
#  BỐN THIẾU SÓT CỦA CÁC BẢN TRƯỚC, ĐÃ SỬA:
#
#  1. TRỌNG SỐ THỜI GIAN CHỈ ÁP CHO SỐ LẦN TRÚNG
#     Bản cũ: chuoi_max và so_lan tính thô -> cầu trúng 5 kỳ liên tiếp CÁCH ĐÂY
#     2 NĂM ngang với cầu vừa trúng 5 kỳ tháng trước.
#     Sửa: MỌI chỉ số đều có trọng số thời gian. Nửa đời đặt theo mốc 6 tháng.
#
#  2. KHÔNG ĐO ĐỘ ĐỀU ĐẶN
#     Bạn nói "chuỗi cầu chạy ĐỀU". Bản cũ chỉ đo chuỗi DÀI NHẤT.
#     Hai cầu cùng trúng 6/20 nhưng khác hẳn:
#        A trúng kỳ 1,2,3,4,5,6      -> chuoi_max=6 rồi TẮT HẲN 14 kỳ
#        B trúng kỳ 3,6,9,12,15,18   -> chuoi_max=1 nhưng ĐỀU ĐẶN
#     Bản cũ cho A điểm cao hơn. Sửa: thêm tín hiệu ĐỘ ĐỀU ĐẶN (C5).
#
#  3. CHƯA BAO GIỜ KIỂM ĐỊNH "CẦU CÓ TỒN TẠI KHÔNG"
#     Bản cũ luôn đi TÌM cầu tốt nhất, chưa hỏi câu gốc: giữa chữ số kỳ trước
#     và chạm kỳ sau có BẤT KỲ liên hệ nào không?
#     Sửa: KIỂM ĐỊNH PHƯƠNG SAI — dưới độc lập, hit rate mọi vị trí quanh 19%.
#     Có cấu trúc cầu -> phương sai giữa các vị trí vượt mức nhiễu nhị thức.
#     Đây là MỘT phép kiểm định, không phải 107 -> KHÔNG bị so sánh bội.
#
#  4. HAI HƯỚNG TIẾP CẬN KHÔNG BAO GIỜ ĐƯỢC GỘP
#     Mô hình xác suất (phân phối chữ số mục tiêu) và cầu chạm (vị trí -> chạm)
#     bổ sung cho nhau nhưng chạy tách rời. Sửa: gộp, trọng số do backtest chọn.
# ==============================================================================

# ┌────────────────────────────────────────────────────────────────────────┐
# │  BẢNG ĐIỀU KHIỂN                                                       │
# │  Mỗi dòng:  <số kỳ> <ĐB> <G1> [G8]                                     │
# │  G8 để trống nếu là Miền Bắc / điện toán (không có giải 8)             │
# └────────────────────────────────────────────────────────────────────────┘

DU_LIEU = """
01
02
03
04
05
06
07
08
09
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
"""

MUC_TIEU     = "DB"   # "DB" = Đặc Biệt | "G1" = Giải Nhất | "G8" = Giải 8
SO_CHAM      = 4      # 3 chạm phủ 51 số · 4 phủ 64 · 5 phủ 75

# --- Trọng số thời gian: mốc "6 tháng trước còn bao nhiêu %" ---
KY_MOI_TUAN  = 1      # số kỳ mỗi tuần. Đài tuần = 1, hàng ngày = 7
CON_LAI_6TH  = 0.10   # kỳ cách 6 tháng còn 10% trọng số. Nhỏ = ưu tiên gần hơn

TOP_CAU      = 5      # số cầu tốt nhất được bỏ phiếu
MIN_KY       = 12
N_NULL       = 3000
N_BOOTSTRAP  = 300
BO_BOT       = 0.20

# ==============================================================================

import numpy as np, time, re, itertools


def _nua_doi():
    """Nửa đời (kỳ) sao cho kỳ cách 6 THÁNG còn đúng CON_LAI_6TH trọng số."""
    ky_6thang = max(4, int(26 * KY_MOI_TUAN))
    return ky_6thang / max(np.log2(1 / CON_LAI_6TH), 0.1)


def doc_du_lieu(text=None, muc_tieu=None):
    mt = (muc_tieu or MUC_TIEU).upper()
    cot = {"DB": 1, "G1": 2, "G8": 3}[mt]
    ky, cb = [], []
    for dong in (text or DU_LIEU).strip().splitlines():
        p = dong.split("#")[0].split()
        if len(p) < 3:
            continue
        stt = p[0]
        if len(p) <= cot:
            cb.append(f"kỳ {stt}: thiếu cột cho {mt}"); continue
        so = [x for x in p[1:]]
        if any(x.upper() == "LOI" for x in so):
            cb.append(f"kỳ {stt}: đánh dấu LOI"); continue
        if not all(x.isdigit() for x in so):
            cb.append(f"kỳ {stt}: có phần tử không phải số"); continue
        # ĐB/G1 phải 5 chữ số; G8 phải 2
        if len(p[1]) != 5 or len(p[2]) != 5:
            cb.append(f"kỳ {stt}: ĐB={p[1]}({len(p[1])}) G1={p[2]}({len(p[2])}) "
                      f"— cần 5 chữ số. Kiểm tra số 0 đứng đầu!"); continue
        if len(p) > 3 and len(p[3]) != 2:
            cb.append(f"kỳ {stt}: G8={p[3]} cần đúng 2 chữ số"); continue
        ky.append({"stt": stt, "so": so, "mt": p[cot]})
    ky.sort(key=lambda k: int(re.sub(r"\D", "", k["stt"]) or 0))
    return ky, cb


def bang_chu_so(ky):
    """Mọi chữ số nguồn của mỗi kỳ + tên vị trí. Nguồn = TOÀN BỘ số đã nhập."""
    n_so = len(ky[0]["so"])
    ten = []
    nhan = ["ĐB", "G1", "G8"]
    for i in range(n_so):
        for j in range(len(ky[0]["so"][i])):
            ten.append(f"{nhan[i] if i < 3 else f'S{i}'}[{j+1}]")
    M = np.array([[int(c) for s in k["so"] for c in s] for k in ky], dtype=np.int8)
    return M, ten


# ==============================================================================
#  QUÉT CẦU — MỌI CHỈ SỐ ĐỀU CÓ TRỌNG SỐ THỜI GIAN  (sửa thiếu sót 1)
# ==============================================================================

def quet(ky):
    M, ten = bang_chu_so(ky)
    n, V = M.shape
    h = _nua_doi()
    cham = [{int(k["mt"][-2]), int(k["mt"][-1])} for k in ky]

    hit = np.zeros((V, n - 1), dtype=bool)          # ma trận trúng đầy đủ
    w = np.zeros(n - 1)
    for t in range(n - 1):
        hit[:, t] = np.isin(M[t], list(cham[t + 1]))
        w[t] = 0.5 ** ((n - 2 - t) / h)             # kỳ mới nhất -> w = 1

    trung_w = (hit * w).sum(axis=1) / w.sum()       # tần suất CÓ trọng số
    so_lan = hit.sum(axis=1)

    # --- chuỗi: max CÓ TRỌNG SỐ + đang chạy ---
    chuoi_max_w = np.zeros(V); chuoi_nay = np.zeros(V, dtype=int)
    for v in range(V):
        cur = 0
        for t in range(n - 1):
            cur = cur + 1 if hit[v, t] else 0
            chuoi_max_w[v] = max(chuoi_max_w[v], cur * w[t])   # chuỗi cũ bị hạ điểm
        chuoi_nay[v] = cur

    # --- ĐỘ ĐỀU ĐẶN (sửa thiếu sót 2) ---
    # Cầu trúng rải đều -> khoảng cách giữa các lần trúng ít biến động.
    # Đo bằng 1 / (1 + hệ số biến thiên của khoảng cách).
    deu = np.zeros(V)
    for v in range(V):
        idx = np.where(hit[v])[0]
        if len(idx) < 3:
            continue
        gap = np.diff(idx)
        cv = gap.std() / max(gap.mean(), 1e-9)
        deu[v] = 1.0 / (1.0 + cv)
    # chỉ thưởng cho cầu còn hoạt động gần đây
    gan_day = (hit[:, -max(3, int(h)):]).any(axis=1)
    deu = deu * gan_day

    return {"M": M, "ten": ten, "hit": hit, "w": w, "trung_w": trung_w,
            "so_lan": so_lan, "chuoi_max_w": chuoi_max_w, "chuoi_nay": chuoi_nay,
            "deu": deu, "n_test": n - 1, "du_bao": M[n - 1], "nua_doi": h}


# ==============================================================================
#  KIỂM ĐỊNH GỐC: CẦU CÓ TỒN TẠI KHÔNG?  (sửa thiếu sót 3)
# ==============================================================================

def kiem_dinh_cau_ton_tai(q, n_lap=N_NULL, seed=2026):
    """MỘT phép kiểm định, không phải 107 -> KHÔNG bị vấn đề so sánh bội.

    Dưới giả thuyết độc lập, hit rate của MỌI vị trí đều quanh 19%, và phương
    sai giữa các vị trí đúng bằng mức nhiễu nhị thức p(1-p)/n.
    Nếu CÓ cấu trúc cầu -> phương sai quan sát VƯỢT mức đó.
    """
    V, ne = q["hit"].shape
    hr = q["hit"].mean(axis=1)
    p = 1 - 0.81
    ty_so = hr.var() / (p * (1 - p) / ne)

    rng = np.random.default_rng(seed)
    null = (rng.random((n_lap, V, ne)) < p).mean(axis=2).var(axis=1) / (p * (1 - p) / ne)
    return {"ty_so": float(ty_so), "p_value": float((null >= ty_so).mean()),
            "null_tb": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "hr_moi_vi_tri": hr}


# ==============================================================================
#  GỘP MÔ HÌNH XÁC SUẤT + CẦU CHẠM  (sửa thiếu sót 4)
# ==============================================================================

def _z(v):
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-12 else (v - v.mean()) / sd


def phan_phoi_cham(ky, h):
    """Mô hình xác suất: tần suất mỗi chữ số làm chạm, CÓ trọng số thời gian."""
    n = len(ky)
    d = np.full(10, 1.0)
    for t, k in enumerate(ky):
        w = 0.5 ** ((n - 1 - t) / h)
        for c in {int(k["mt"][-2]), int(k["mt"][-1])}:
            d[c] += w
    return d / d.sum()


# 5 tín hiệu, trọng số nhị phân -> chỉ 31 tổ hợp (diện tích dò rất nhỏ)
TEN_TH = ["C1_TanSuatW", "C2_ChuoiNay", "C3_ChuoiMaxW", "C4_DeuDan", "C5_MoHinh"]


def diem_cham(q, ky, w_th, top=TOP_CAU):
    """Điểm cho từng chữ số 0-9 từ 5 tín hiệu."""
    diem_cau = (w_th[0] * _z(q["trung_w"]) + w_th[1] * _z(q["chuoi_nay"])
                + w_th[2] * _z(q["chuoi_max_w"]) + w_th[3] * _z(q["deu"]))
    phieu = np.zeros(10)
    if np.any(w_th[:4]):
        tot = np.argsort(-diem_cau)[:min(top, len(diem_cau))]
        nen = diem_cau[tot].min()
        for j in tot:
            phieu[q["du_bao"][j]] += max(diem_cau[j] - nen + 0.1, 0.1)
        phieu = _z(phieu)
    mh = _z(np.log(phan_phoi_cham(ky, q["nua_doi"]))) if w_th[4] else np.zeros(10)
    return phieu + w_th[4] * mh, diem_cau


def chon_cham(q, ky, w_th, k=SO_CHAM, top=TOP_CAU):
    d, diem_cau = diem_cham(q, ky, w_th, top)
    return tuple(int(x) for x in sorted(np.argsort(-d)[:k])), d, diem_cau


def chon_trong_so(ky, k=SO_CHAM):
    """Trọng số do WALK-FORWARD chọn. 31 tổ hợp nhị phân."""
    n = len(ky); mt = max(MIN_KY, n // 2)
    best = (None, -1.0, 0)
    for w_th in itertools.product([0.0, 1.0], repeat=5):
        if sum(w_th) == 0:
            continue
        trung = tong = 0
        for t in range(mt, n):
            bo, _, _ = chon_cham(quet(ky[:t]), ky[:t], w_th, k)
            s = set(bo); m = ky[t]["mt"]
            trung += int(m[-2]) in s or int(m[-1]) in s
            tong += 1
        hr = trung / max(tong, 1)
        if hr > best[1]:
            best = (w_th, hr, tong)
    return best


# ==============================================================================
#  ĐỘ ỔN ĐỊNH & BÁO CÁO
# ==============================================================================

def do_on_dinh(ky, w_th, k=SO_CHAM, n_lap=N_BOOTSTRAP, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(ky); giu = max(MIN_KY, int(n * (1 - BO_BOT)))
    dem = np.zeros(10)
    for _ in range(n_lap):
        idx = sorted(rng.choice(n, min(giu, n), replace=False))
        con = [ky[i] for i in idx]
        try:
            bo, _, _ = chon_cham(quet(con), con, w_th, k)
        except Exception:
            continue
        for c in bo:
            dem[c] += 1
    return dem / n_lap


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p, den = k / n, 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / den
    return (max(0., ctr - hw), min(1., ctr + hw))


def so_cua_cham(bo):
    s = set(bo)
    return [f"{v:02d}" for v in range(100) if (v // 10 in s) or (v % 10 in s)]


def chay(text=None, muc_tieu=None, k=None):
    t0 = time.time()
    mt_ten = (muc_tieu or MUC_TIEU).upper()
    k = k or SO_CHAM
    ky, cb = doc_du_lieu(text, mt_ten)
    n = len(ky)
    nhan = {"DB": "ĐỀ ĐẶC BIỆT", "G1": "ĐỀ GIẢI NHẤT", "G8": "ĐỀ ĐẦU (GIẢI 8)"}[mt_ten]

    print("=" * 74)
    print(f"  {k} CHẠM — {nhan}   |   {n} kỳ")
    print("=" * 74)
    if cb:
        print(f"\n  ⚠ {len(cb)} dòng có vấn đề:")
        for c in cb[:6]: print(f"     {c}")
    if n < MIN_KY:
        print(f"\n  ✗ Chỉ {n} kỳ hợp lệ, cần tối thiểu {MIN_KY}."); return None

    q = quet(ky)
    h = q["nua_doi"]
    print(f"\n  Trọng số thời gian: nửa đời {h:.1f} kỳ "
          f"→ kỳ cách 6 tháng còn {CON_LAI_6TH:.0%}, cách 1 năm còn "
          f"{CON_LAI_6TH**2:.1%}")
    print(f"  Quét {len(q['ten'])} vị trí cầu × {q['n_test']} lần kiểm chứng")

    # ---------- [A] CÂU HỎI GỐC ----------
    kd = kiem_dinh_cau_ton_tai(q)
    print(f"\n[A] CẦU CÓ TỒN TẠI KHÔNG?  (1 phép kiểm định, không bị so sánh bội)")
    print("-" * 74)
    print(f"  Tỷ số phương sai hit rate giữa các vị trí: {kd['ty_so']:.2f}")
    print(f"  Dưới giả thuyết ĐỘC LẬP: TB {kd['null_tb']:.2f}, ngưỡng P95 {kd['null_p95']:.2f}")
    print(f"  p-value = {kd['p_value']:.3f}")
    if kd["p_value"] < 0.05:
        print(f"  → CÓ bằng chứng cấu trúc cầu. Các vị trí KHÔNG đồng nhất.")
        print(f"     Đây là phát hiện đáng giá — bộ chạm dưới đây có cơ sở.")
    else:
        print(f"  → KHÔNG có bằng chứng cấu trúc cầu. Mọi vị trí hành xử như nhau,")
        print(f"     đều quanh mốc 19%. Cầu 'mạnh' tìm được chỉ là dao động nhị thức.")

    # ---------- [B] TRỌNG SỐ ----------
    w_th, hr_wf, n_wf = chon_trong_so(ky, k)
    bat = [t for t, x in zip(TEN_TH, w_th) if x]
    print(f"\n[B] TRỌNG SỐ DO WALK-FORWARD CHỌN (31 tổ hợp)")
    print("-" * 74)
    print(f"  Bật: {', '.join(bat)}")
    print(f"  Hit rate NGOÀI MẪU: {hr_wf:.1%} trên {n_wf} kỳ  (mốc {1-((10-k)/10)**2:.0%})")

    # ---------- [C] BỘ CHẠM ----------
    bo, diem, diem_cau = chon_cham(q, ky, w_th, k)
    on_dinh = do_on_dinh(ky, w_th, k)
    print("\n" + "=" * 74)
    print(f"     {k} CHẠM CHO KỲ TIẾP THEO:   {'   '.join(str(c) for c in bo)}")
    print("=" * 74)

    print(f"\n  ĐỘ ỔN ĐỊNH ({N_BOOTSTRAP} lượt, bỏ ngẫu nhiên {BO_BOT:.0%} số kỳ)")
    print("  " + "-" * 70)
    for c in bo:
        v = on_dinh[c]
        nh = ("RẤT ỔN ĐỊNH" if v >= .80 else "ổn định" if v >= .60
              else "trung bình" if v >= .45 else "YẾU — dễ đổi")
        print(f"     chạm {c}   {v:>5.0%}  {'█'*int(v*28):<28} {nh}")
    sat = sorted([(d, on_dinh[d]) for d in range(10)
                  if d not in bo and on_dinh[d] >= .35], key=lambda x: -x[1])
    if sat:
        print(f"     Sát nút: " + ", ".join(f"chạm {d} ({v:.0%})" for d, v in sat))
    tb = float(np.mean([on_dinh[c] for c in bo]))
    print(f"\n     ĐỘ TỰ TIN TỔNG THỂ: {tb:.0%}")

    so = so_cua_cham(bo)
    print(f"\n  {len(so)} SỐ QUY RA:")
    print("  " + ",".join(so))

    # ---------- [D] CẦU ----------
    tot = np.argsort(-diem_cau)[:TOP_CAU]
    print(f"\n  CẦU HÀNG ĐẦU  (đã tính trọng số thời gian)")
    print("  " + "-" * 70)
    print(f"  {'Vị trí':<10}{'→chạm':>7}{'Trúng':>9}{'Tần suất W':>12}"
          f"{'Đều đặn':>10}{'Chuỗi':>14}")
    for j in tot:
        song = f"ĐANG CHẠY {q['chuoi_nay'][j]}" if q["chuoi_nay"][j] else "đã gãy"
        print(f"  {q['ten'][j]:<10}{q['du_bao'][j]:>7}"
              f"{q['so_lan'][j]:>6}/{q['n_test']:<3}{q['trung_w'][j]:>11.0%}"
              f"{q['deu'][j]:>10.2f}{song:>14}")

    lo, hi = wilson(int(hr_wf * n_wf), n_wf)
    print(f"\n  KTC 95% hit rate ngoài mẫu: [{lo:.0%}, {hi:.0%}]")
    print(f"\n  ⚠ Nhìn theo thứ tự: [A] p-value cấu trúc cầu TRƯỚC. Nếu ≥ 0,05 thì")
    print(f"    bộ chạm này ngang bốc bừa, độ ổn định cao đến mấy cũng không đổi.")
    print(f"    {k} chạm phủ {len(so)}/100 số → P(trúng) = {len(so)}% trừ khi máy quay lệch thật.")
    print(f"\n  ({time.time()-t0:.1f}s)  Kỳ tiếp: điền dòng {int(re.sub(r'\D','',ky[-1]['stt']) or 0)+1:02d} rồi Run lại.")
    return {"cham": bo, "so": so, "tu_tin": tb, "p_cau": kd["p_value"],
            "hr_wf": hr_wf, "trong_so": bat}


if __name__ == "__main__":
    chay()
