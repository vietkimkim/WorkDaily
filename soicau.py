"""
DÒ CẦU CÓ NULL CONTROL — kiểm định phương pháp soi cầu vị trí.

CẦU VỊ TRÍ là gì:
    Chọn 2 vị trí chữ số CỐ ĐỊNH trong bảng kết quả kỳ trước, ghép thành số
    2 chữ số, đánh số đó ở kỳ sau. Ví dụ (ĐB chữ số 3) × (G.nhất chữ số 2).

VẤN ĐỀ:
    Miền Bắc/điện toán có 107 vị trí chữ số  -> 11.342 cầu khả dĩ
    Miền Nam/Trung     có  82 vị trí chữ số  ->  6.642 cầu khả dĩ

    Với dữ liệu NGẪU NHIÊN hoàn toàn, số cầu "chạy" 5 kỳ liên tiếp:
        Lô 2 số (27 lô, P=23,77%) : ~817 cầu
        Đề đặc biệt (1 lô, P=1%)  : ~0,0 cầu

    Nên: cầu chạy dài cho LÔ gần như chắc chắn là nhiễu.
         Cầu chạy dài cho ĐỀ mới thật sự đáng chú ý.

CÁCH LÀM ĐÚNG:
    Quét toàn bộ cầu, backtest walk-forward từng cái, lấy cầu tốt nhất.
    Rồi chạy Y HỆT quy trình đó trên dữ liệu ngẫu nhiên để biết "cầu tốt nhất
    do may mắn" đạt bao nhiêu. So hai con số mới ra kết luận.

DÙNG:
    import soicau
    soicau.do_cau(du_lieu, muc_tieu="LO2")   # hoặc "DB"
"""
import numpy as np
import engine as E


# ==============================================================================
#  BẢN ĐỒ VỊ TRÍ CHỮ SỐ
# ==============================================================================

def ban_do_vi_tri(giai_mau):
    """Trả list (chi_so_giai, chi_so_chu_so) cho MỌI vị trí chữ số trong 1 kỳ."""
    return [(i, j) for i, s in enumerate(giai_mau) for j in range(len(s))]


def _ma_tran_chu_so(toan_giai, vt):
    """Ma trận (n_ky, n_vi_tri) chứa chữ số tại từng vị trí. Vector hoá để nhanh."""
    n = len(toan_giai)
    M = np.zeros((n, len(vt)), dtype=np.int8)
    for t, ky in enumerate(toan_giai):
        for k, (i, j) in enumerate(vt):
            M[t, k] = int(ky[i][j])
    return M


# ==============================================================================
#  QUÉT CẦU
# ==============================================================================

def quet_cau(toan_giai, muc_tieu="LO2", min_train=20):
    """Chấm điểm MỌI cầu. Trả (hit_rate[], chuoi_dai_nhat[], danh sách cặp vị trí).

    Cầu (a,b): lấy chữ số vị trí a và b của kỳ t -> số 2 chữ số -> dự báo kỳ t+1.
    """
    vt = ban_do_vi_tri(toan_giai[0])
    M = _ma_tran_chu_so(toan_giai, vt)
    n, V = M.shape

    # Mục tiêu của từng kỳ, dạng tập hợp số 0-99
    if muc_tieu == "DB":
        mt = [{int(ky[0][-2:])} for ky in toan_giai]
    else:
        mt = [{int(x[-2:]) for x in ky} for ky in toan_giai]

    # Ma trận trúng: T[t, v] = cầu v dự báo từ kỳ t có trúng ở kỳ t+1 không
    # Duyệt theo cặp (a,b) sẽ là V^2 ~ 11.442 cột -> vector hoá theo t
    cap = [(a, b) for a in range(V) for b in range(V) if a != b]
    n_cau = len(cap)
    A = np.array([a for a, _ in cap]); B = np.array([b for _, b in cap])

    hit = np.zeros((n - 1 - min_train, n_cau), dtype=bool)
    for idx, t in enumerate(range(min_train, n - 1)):
        so = M[t, A] * 10 + M[t, B]          # số cầu dự báo, vector 11.342
        thuoc = np.zeros(100, dtype=bool)
        for v in mt[t + 1]:
            thuoc[v] = True
        hit[idx] = thuoc[so]

    hr = hit.mean(axis=0)

    # Chuỗi trúng liên tiếp dài nhất của từng cầu
    dai = np.zeros(n_cau, dtype=np.int16)
    cur = np.zeros(n_cau, dtype=np.int16)
    for i in range(hit.shape[0]):
        cur = np.where(hit[i], cur + 1, 0)
        dai = np.maximum(dai, cur)
    return hr, dai, cap, vt, hit


def _mo_ta_cau(cap_ab, vt, giai_mau):
    """Diễn giải cầu ra chữ, ví dụ 'giải#0 (5 chữ số) vị trí 3 × giải#1 vị trí 2'."""
    a, b = cap_ab
    ga, ja = vt[a]; gb, jb = vt[b]
    return (f"giải#{ga}[chữ số {ja+1}] × giải#{gb}[chữ số {jb+1}]"
            f"   (mẫu: {giai_mau[ga]}[{ja}]={giai_mau[ga][ja]}, "
            f"{giai_mau[gb]}[{jb}]={giai_mau[gb][jb]})")


# ==============================================================================
#  NULL CONTROL — chạy Y HỆT trên dữ liệu ngẫu nhiên
# ==============================================================================

def do_cau(toan_giai, muc_tieu="LO2", min_train=20, n_null=20, top=10, seed=2026):
    n = len(toan_giai)
    khu = "MB" if len(toan_giai[0]) == 27 else "MN"
    p0 = 1 - (1 - 0.01) ** len(toan_giai[0]) if muc_tieu == "LO2" else 0.01

    print("=" * 78)
    print(f"  DÒ CẦU VỊ TRÍ — mục tiêu {'ĐỀ ĐẶC BIỆT' if muc_tieu=='DB' else 'LÔ 2 SỐ'}")
    print(f"  {n} kỳ | {khu} | mốc ngẫu nhiên mỗi cầu = {p0:.2%}")
    print("=" * 78)

    hr, dai, cap, vt, hit = quet_cau(toan_giai, muc_tieu, min_train)
    n_test = hit.shape[0]
    print(f"  {len(vt)} vị trí chữ số → {len(cap):,} cầu khả dĩ, {n_test} kỳ kiểm tra")

    # ---------- Null: cầu TỐT NHẤT do may mắn đạt bao nhiêu ----------
    print(f"\n  Đang dựng phân phối null ({n_null} lượt trên RNG công bằng)...")
    rng = np.random.default_rng(seed)
    null_hr, null_dai = [], []
    for i in range(n_null):
        gia = [E.sinh_ky_gia(E.CO_CAU_GIAI[khu], rng) for _ in range(n)]
        h2, d2, _, _, _ = quet_cau(gia, muc_tieu, min_train)
        null_hr.append(h2.max()); null_dai.append(d2.max())
        if (i + 1) % max(1, n_null // 5) == 0:
            print(f"     ... {i+1}/{n_null}", end="\r")
    print(" " * 40, end="\r")
    null_hr = np.array(null_hr); null_dai = np.array(null_dai)

    # ---------- Kết quả ----------
    print(f"\n[A] CẦU TỐT NHẤT THEO HIT RATE")
    print("-" * 78)
    thu_tu = np.argsort(-hr)[:top]
    print(f"  {'Hạng':>5}{'Hit rate':>11}{'Chuỗi dài':>11}   Cầu")
    print("  " + "-" * 74)
    for r, j in enumerate(thu_tu, 1):
        print(f"  {r:>5}{hr[j]:>10.1%}{dai[j]:>11}   {_mo_ta_cau(cap[j], vt, toan_giai[-1])}")

    print(f"\n[B] SO VỚI NHIỄU — đây mới là phép so đúng")
    print("-" * 78)
    p_hr = (null_hr >= hr.max()).mean()
    p_dai = (null_dai >= dai.max()).mean()
    print(f"  {'':<26}{'Dữ liệu THẬT':>15}{'Nhiễu (TB)':>14}{'Nhiễu (max)':>14}")
    print("  " + "-" * 74)
    print(f"  {'Hit rate cầu tốt nhất':<26}{hr.max():>14.1%}"
          f"{null_hr.mean():>14.1%}{null_hr.max():>14.1%}")
    print(f"  {'Chuỗi trúng dài nhất':<26}{dai.max():>14}"
          f"{null_dai.mean():>14.1f}{null_dai.max():>14}")
    print("  " + "-" * 74)
    print(f"  p-value (hit rate)   = {p_hr:.4f}")
    print(f"  p-value (chuỗi dài)  = {p_dai:.4f}")
    print()
    if min(p_hr, p_dai) < 0.05:
        print("  → VƯỢT ngưỡng nhiễu. Đáng kiểm chứng lại trên đài/kỳ khác.")
    else:
        print("  → NẰM TRONG biên độ nhiễu.")
        print("    Cầu tốt nhất trên dữ liệu thật KHÔNG hơn cầu tốt nhất tìm được")
        print("    trên dữ liệu ngẫu nhiên. Đây là hệ quả của việc dò "
              f"{len(cap):,} cầu")
        print("    trên chỉ {} kỳ — tìm thấy 'quy luật' là điều chắc chắn xảy ra.".format(n_test))

    # ---------- Cầu đang chạy ----------
    print(f"\n[C] CẦU ĐANG CHẠY (trúng liên tiếp tính đến kỳ mới nhất)")
    print("-" * 78)
    cur = np.zeros(len(cap), dtype=np.int16)
    for i in range(hit.shape[0]):
        cur = np.where(hit[i], cur + 1, 0)
    dang = np.argsort(-cur)[:5]
    for j in dang:
        if cur[j] == 0: break
        print(f"  Chạy {cur[j]:>2} kỳ | hit {hr[j]:>5.1%} | {_mo_ta_cau(cap[j], vt, toan_giai[-1])}")
    ky_vong = len(cap) * n_test * (p0 ** int(cur[dang[0]])) if cur[dang[0]] > 0 else 0
    if cur[dang[0]] > 0:
        print(f"\n  Số cầu chạy >= {cur[dang[0]]} kỳ KỲ VỌNG do ngẫu nhiên: {ky_vong:,.1f}")
        print("  → " + ("Hiếm — đáng chú ý." if ky_vong < 1
                        else f"Bình thường. Có ~{ky_vong:,.0f} cầu như vậy trong dữ liệu ngẫu nhiên."))
    print("=" * 78)
    return {"hr": hr, "dai": dai, "cap": cap, "vt": vt,
            "p_hr": p_hr, "p_dai": p_dai, "dang_chay": cur}
