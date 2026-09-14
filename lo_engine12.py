"""
LO_ENGINE12 — 12 engine thi đấu, chọn 5 engine để sinh 20 con 2 số cho BAO LÔ.

CHỈ CHẠY MIỀN NAM + MIỀN TRUNG (18 giải -> 18 lô). Miền Bắc 27 lô, cơ cấu khác.

KIẾN TRÚC
    100 kỳ mỗi đài
      ├── 90 kỳ xa nhất  : 12 engine học
      └── 10 kỳ gần nhất : chấm điểm, chọn 5 engine chính xác nhất
    Mỗi engine ra 4 con -> 5 engine x 4 = 20 con.

BA CON SỐ NỀN (đo bằng mô phỏng 20.000 lượt):

  ① P(1 số cụ thể xuất hiện trong 18 lô) = 1 - 0,99^18 = 16,55%
     20 số -> trúng TB 3,60 lần/kỳ. Vốn 20x18 = 360. Hoà vốn cần 3,79 lần.
     -> cần hơn ngẫu nhiên 5,26% tương đối. EV = -5,0%.

  ② KHÔNG dùng "có trúng hay không" làm thước đo
     Với 20 số, P(trúng >=1) = 98,2% — gần như LUÔN trúng, kể cả bốc bừa.
     -> Thước đo phải là SỐ LẦN TRÚNG.

  ③ CHỌN 5/12 TRÊN 10 KỲ VẪN CÓ ẢO GIÁC, nhưng NHỎ HƠN bài 4 chạm
     Mỗi engine ra 4 số x 10 kỳ = 40 quan sát (bài chạm chỉ có 10).
     Nếu CẢ 12 engine vô dụng: 5 engine top vẫn đạt 21,7% vs mốc 16,5%
     -> ảo giác +5,2 điểm (bài chạm là +13,0 điểm).
     -> Sửa: chấm walk-forward trên 90 kỳ, 10 kỳ gần chỉ CỘNG THƯỞNG.

  ④ 20 SỐ TỪ 5 ENGINE HAY TRÙNG
     Trung bình chỉ ra 18,5 số khác nhau. P(đủ 20) = 17,8%.
     -> Sửa: bù từ engine hạng 6 trở đi cho đủ SO_CON.
"""
import os, smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import numpy as np

import engine as E

VN = timezone(timedelta(hours=7))

# ==============================================================================
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  BẢNG ĐIỀU KHIỂN                                                     │
#  └──────────────────────────────────────────────────────────────────────┘
# ==============================================================================

SO_KY        = 100    # 90 học + 10 đo phong độ
CUA_SO_GAN   = 10     # số kỳ gần nhất đo phong độ
SO_CON       = 36     # tổng số con cần ra
CON_MOI_ENG  = 6      # mỗi engine ra bao nhiêu con
SO_ENGINE    = 6      # chọn bao nhiêu engine  (6 x 6 = 36)
                      #
                      # VÌ SAO 36: mục tiêu "6-8 con trúng mỗi kỳ".
                      #   20 con -> trúng TB 3,31 · P(>=6 con) chỉ 10%
                      #   36 con -> trúng TB 5,96 · P(>=6 con) = 56%
                      #   50 con -> trúng TB 8,27 · P(>=6 con) = 86%
                      # NHƯNG EV KHÔNG ĐỔI (-5,0%) ở mọi mức: số con trúng tăng,
                      # tiền vào cuộc tăng ĐÚNG BẰNG. Hoà vốn cần 6,82 thay vì 3,79.
                      # "Trúng nhiều con" mua được bằng tiền, không phải thuật toán.

W_GAN        = 0.6    # trọng số phong độ 10 kỳ (0 = bỏ qua, 1 = ngang 90 kỳ)
HALF_LIFE    = 25     # nửa đời trọng số thời gian
MIN_TRAIN    = 25     # số kỳ tối thiểu trước khi engine bắt đầu dự báo
TY_LE_TRA    = 95.0   # thưởng mỗi lần trúng
DIEM_MOI_CON = 1      # điểm cược mỗi con (vốn = SO_CON x 18 x DIEM_MOI_CON)

EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")

PAIRS = np.array([(i // 10, i % 10) for i in range(100)])
P_D1, P_D2 = PAIRS[:, 0], PAIRS[:, 1]
MIRROR = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}


# ==============================================================================
#  12 ENGINE — mỗi engine nhận (lo, M) -> mảng điểm 100 phần tử
#     lo[t] = list số 2 chữ số của 18 lô kỳ t
#     M[t]  = mọi chữ số nguồn của kỳ t (toàn bảng 18 giải)
# ==============================================================================

def _z(v):
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-12 else (v - v.mean()) / sd


def _dem_lo(lo, w=None):
    d = np.zeros(100); n = len(lo)
    for t, ky in enumerate(lo):
        wt = 1.0 if w is None else 0.5 ** ((n - 1 - t) / w)
        for v in ky:
            d[v] += wt
    return d


def L01_TanSuat(lo, M):
    """Số ra nhiều nhất trong 18 lô, toàn bộ lịch sử."""
    return _z(_dem_lo(lo))


def L02_TanSuatW(lo, M):
    """Như L01 nhưng trọng số giảm dần theo thời gian."""
    return _z(_dem_lo(lo, HALF_LIFE))


def L03_Bigram(lo, M):
    """P(hàng đơn vị | hàng chục) — mô hình hoá quan hệ 2 chữ số."""
    c1 = np.ones(10); B = np.ones((10, 10))
    for ky in lo:
        for v in ky:
            c1[v // 10] += 1; B[v // 10, v % 10] += 1
    p1 = c1 / c1.sum(); Pb = B / B.sum(1, keepdims=True)
    return _z(np.log(p1[P_D1]) + np.log(Pb[P_D1, P_D2]))


def L04_MarkovKy(lo, M):
    """Số hay xuất hiện NGAY SAU khi ĐB kỳ trước là x (2 chữ số cuối)."""
    T1 = np.ones((10, 10)); T2 = np.ones((10, 10))
    for t in range(len(lo) - 1):
        a, b = lo[t][0] // 10, lo[t][0] % 10      # lô[0] = 2 số cuối ĐB
        for v in lo[t + 1]:
            T1[a, v // 10] += 1; T2[b, v % 10] += 1
    T1 /= T1.sum(1, keepdims=True); T2 /= T2.sum(1, keepdims=True)
    a, b = lo[-1][0] // 10, lo[-1][0] % 10
    return _z(np.log(T1[a][P_D1]) + np.log(T2[b][P_D2]))


def L05_CauViTri(lo, M):
    """Ghép 2 vị trí chữ số của kỳ trước thành số 2 chữ số. Chọn cầu tốt nhất."""
    n, V = M.shape
    if n < 5:
        return np.zeros(100)
    A = np.repeat(np.arange(V), V); B = np.tile(np.arange(V), V)
    giu = A != B; A, B = A[giu], B[giu]
    sc = np.zeros(len(A), dtype=np.int32)
    for t in range(n - 1):
        so = M[t, A] * 10 + M[t, B]
        thuoc = np.zeros(100, dtype=bool)
        for v in lo[t + 1]:
            thuoc[v] = True
        sc += thuoc[so]
    top = np.argsort(-sc)[:40]
    d = np.zeros(100)
    du = M[n - 1, A[top]] * 10 + M[n - 1, B[top]]
    for j, v in enumerate(du):
        d[v] += sc[top[j]]
    return _z(d)


def L06_ChamKep(lo, M):
    """Số có CẢ HAI chữ số đều là chạm nóng gần đây."""
    c = np.zeros(10); n = len(lo)
    for t, ky in enumerate(lo):
        w = 0.5 ** ((n - 1 - t) / HALF_LIFE)
        for v in ky:
            c[v // 10] += w; c[v % 10] += w
    p = c / c.sum()
    return _z(np.log(p[P_D1]) + np.log(p[P_D2]))


def L07_GanLo(lo, M):
    """Số lâu nhất chưa về. Hình học KHÔNG NHỚ -> lý thuyết = 0."""
    g = np.full(100, float(len(lo)))
    for t in range(len(lo) - 1, -1, -1):
        for v in lo[t]:
            g[v] = min(g[v], len(lo) - 1 - t)
    return _z(g)


def L08_Bet(lo, M):
    """Số vừa ra kỳ trước lặp lại. Không có cơ chế."""
    d = np.zeros(100)
    for v in lo[-1]:
        d[v] += 1
    return _z(d)


def L09_LonSo(lo, M):
    """Đảo ngược số vừa ra: 12 -> 21. Quy tắc dân gian."""
    d = np.zeros(100)
    for v in lo[-1]:
        d[(v % 10) * 10 + v // 10] += 1
    return _z(d)


def L10_BongSo(lo, M):
    """Bóng số: 0<->5, 1<->6... áp cho cả 2 chữ số. Quy tắc dân gian."""
    d = np.zeros(100)
    for v in lo[-1]:
        d[MIRROR[v // 10] * 10 + MIRROR[v % 10]] += 1
    return _z(d)


def L11_KepSatKep(lo, M):
    """Ưu tiên kép (11,22...) và sát kép. Không có cơ chế."""
    kep = (P_D1 == P_D2).astype(float)
    sat = ((np.abs(P_D1 - P_D2) == 1) | (np.abs(P_D1 - P_D2) == 9)).astype(float)
    gan = _dem_lo(lo, HALF_LIFE)
    return _z(2.0 * kep + 1.0 * sat + 0.01 * _z(gan))


def L12_SoLanh(lo, M):
    """Số ra ÍT nhất — giả định bù trừ, sai với các kỳ độc lập."""
    return _z(-_dem_lo(lo, HALF_LIFE))


ENGINES = [("L01_TanSuat", L01_TanSuat), ("L02_TanSuatW", L02_TanSuatW),
           ("L03_Bigram", L03_Bigram), ("L04_MarkovKy", L04_MarkovKy),
           ("L05_CauViTri", L05_CauViTri), ("L06_ChamKep", L06_ChamKep),
           ("L07_GanLo", L07_GanLo), ("L08_Bet", L08_Bet),
           ("L09_LonSo", L09_LonSo), ("L10_BongSo", L10_BongSo),
           ("L11_KepSatKep", L11_KepSatKep), ("L12_SoLanh", L12_SoLanh)]
TEN_ENGINE = [n for n, _ in ENGINES]
CO_CHE = {"L01_TanSuat": "có", "L02_TanSuatW": "có", "L03_Bigram": "có",
          "L04_MarkovKy": "k.định", "L05_CauViTri": "k.định", "L06_ChamKep": "k.định",
          "L07_GanLo": "KHÔNG", "L08_Bet": "KHÔNG", "L09_LonSo": "KHÔNG",
          "L10_BongSo": "KHÔNG", "L11_KepSatKep": "KHÔNG", "L12_SoLanh": "KHÔNG"}


# ==============================================================================
#  CHẤM ĐIỂM 12 ENGINE
#
#  Thước đo = SỐ LẦN TRÚNG (không phải "có trúng hay không" — với 20 số thì
#  P(trúng >=1) = 98,2%, gần như luôn trúng kể cả bốc bừa).
#  Walk-forward: tại mỗi kỳ t, engine chỉ nhìn dữ liệu < t.
#  Điểm = hit rate 90 kỳ + W_GAN x hit rate 10 kỳ gần.
# ==============================================================================

def _top_k(diem, k):
    return np.argsort(-diem, kind="stable")[:k]


def cham_diem_engines(lo, M, k=CON_MOI_ENG, cua_so_gan=CUA_SO_GAN,
                      min_train=MIN_TRAIN):
    n = len(lo)
    mt = min(min_train, max(8, n // 3))
    n_eng = len(ENGINES)
    hit = np.zeros((n_eng, n - mt), dtype=np.int16)   # số con trúng mỗi kỳ

    for i, t in enumerate(range(mt, n)):
        thuc = set(lo[t])
        for j, (_, f) in enumerate(ENGINES):
            try:
                sel = _top_k(f(lo[:t], M[:t]), k)
            except Exception:
                sel = np.arange(k)
            hit[j, i] = sum(1 for v in sel if int(v) in thuc)

    n_test = hit.shape[1]
    g = min(cua_so_gan, n_test)
    ra = []
    for j, (ten, f) in enumerate(ENGINES):
        hr_all = float(hit[j].mean() / k)            # tỷ lệ trúng mỗi con
        hr_gan = float(hit[j, -g:].mean() / k) if g else hr_all
        try:
            sel = _top_k(f(lo, M), k)
        except Exception:
            sel = np.arange(k)
        ra.append({"ten": ten, "hr_all": hr_all, "hr_gan": hr_gan,
                   "trung_all": int(hit[j].sum()), "trung_gan": int(hit[j, -g:].sum()),
                   "tb_gan": float(hit[j, -g:].mean()),
                   "diem": hr_all + W_GAN * hr_gan,
                   "so": sorted(int(v) for v in sel), "co_che": CO_CHE[ten]})
    ra.sort(key=lambda x: -x["diem"])
    return ra, n_test, g, hit


def chon_bo_so(bang, tong=SO_CON, k=CON_MOI_ENG):
    """Lấy số từ engine hạng cao. Trùng thì BÙ từ engine kế tiếp cho đủ."""
    da, nguon = [], []
    for e in bang:
        them = [v for v in e["so"] if v not in da]
        if not them:
            continue
        nhan = them[:max(0, tong - len(da))]
        if not nhan:
            break
        da.extend(nhan)
        nguon.append({**e, "dong_gop": nhan})
        if len(da) >= tong:
            break
    # cực hiếm: 12 engine không đủ số khác nhau -> bù theo tần suất
    i = 0
    while len(da) < tong and i < 100:
        if i not in da:
            da.append(i)
        i += 1
    return sorted(da[:tong]), nguon


def null_chon(n_test, g, k=CON_MOI_ENG, n_eng=5, n_lap=1500, seed=2026):
    """Nếu CẢ 12 engine đều vô dụng, 5 engine top đạt bao nhiêu?"""
    rng = np.random.default_rng(seed)
    p = 1 - (1 - 0.01) ** 18
    h = rng.random((n_lap, len(ENGINES), n_test, k)) < p
    hr_all = h.mean(axis=(2, 3)); hr_gan = h[:, :, -g:, :].mean(axis=(2, 3))
    diem = hr_all + W_GAN * hr_gan
    top = np.argsort(-diem, axis=1)[:, :n_eng]
    return np.array([hr_gan[i, top[i]].mean() for i in range(n_lap)])


# ==============================================================================
#  CHẠY 1 ĐÀI  (chỉ MN/MT — 18 giải)
# ==============================================================================

def p_dat_muc_tieu(k, m, n_lo=18):
    """P(ít nhất m trong k con xuất hiện trong n_lo lô), giả định ngẫu nhiên."""
    from scipy import stats as _st
    p1 = 1 - (1 - 0.01) ** n_lo
    return float(1 - _st.binom.cdf(m - 1, k, p1))


def chay_dai(stt, ngay_moc, so_ky=None, tong=None):
    so_ky = so_ky or SO_KY
    tong = tong or SO_CON
    ten, ma, mien, nd = E.lay_dai(stt)
    if ma == "xsmb":
        raise RuntimeError("bỏ qua Miền Bắc (27 lô, cơ cấu khác MN/MT)")

    m = E.lay_tu_master(stt, so_ky + 20)
    if m:
        toan_giai, ngay_full, info = m; nguon = "KHO"
    else:
        _, ngay, ngay_full, toan_giai, info = E.lay_du_lieu(stt, min(200, so_ky + 30))
        nguon = "WEB"
    if not toan_giai:
        raise RuntimeError("không bóc được toàn bảng giải")

    tg, ng, that = E.cat_truoc_ngay(toan_giai, ngay_full, ngay_moc, so_ky)
    if len(tg) < 35:
        raise RuntimeError(f"chỉ còn {len(tg)} kỳ trước {ngay_moc:%d.%m.%Y}, cần >=35")
    if len(tg[0]) != 18:
        raise RuntimeError(f"đài này có {len(tg[0])} giải, không phải 18 (MN/MT)")

    lo = [[int(s[-2:]) for s in ky] for ky in tg]
    M = np.array([[int(c) for s in ky for c in s] for ky in tg], dtype=np.int8)
    n_lo = len(lo[0])

    bang, n_test, g, hit = cham_diem_engines(lo, M)
    bo, nguon_e = chon_bo_so(bang, tong)
    nb = null_chon(n_test, g, n_eng=len(nguon_e))
    hr5 = float(np.mean([e["hr_gan"] for e in nguon_e]))
    p_val = float((nb >= hr5).mean())

    von = tong * n_lo * DIEM_MOI_CON
    p1 = 1 - (1 - 0.01) ** n_lo
    tb_con = tong * p1                      # số CON khác nhau trúng
    tb = n_lo * tong / 100.0                # số LƯỢT trúng (tính cả trùng)
    hoa_von = von / (TY_LE_TRA * DIEM_MOI_CON)
    muc_tieu = {m: p_dat_muc_tieu(tong, m, n_lo) for m in (4, 6, 8, 10)}

    return {"stt": stt, "dai": ten, "mien": mien, "nguon": nguon, "n_ky": len(tg),
            "n_lo": n_lo, "ngay_ky_truoc": ng[-1].strftime("%d/%m/%Y"),
            "db_ky_truoc": tg[-1][0], "ket_qua_that": that,
            "so": [f"{v:02d}" for v in bo], "bang": bang, "nguon_e": nguon_e,
            "n_test": n_test, "g": g, "hr5_gan": hr5, "null_tb": float(nb.mean()),
            "p_value": p_val, "von": von, "trung_tb": tb, "trung_con": tb_con,
            "hoa_von": hoa_von, "muc_tieu": muc_tieu,
            "ev": tb * TY_LE_TRA * DIEM_MOI_CON - von}


# ==============================================================================
#  EMAIL
# ==============================================================================

def _html(ket, ngay_moc, loi):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    thu = E.THU_VN[ngay_moc.weekday()]
    r0 = ket[0]
    h = [f'<div style="{css}max-width:760px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">{SO_CON} con bao lô — {thu} {ngay_moc:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài Miền Nam + Miền Trung · 12 engine thi đấu · '
             f'{SO_KY} kỳ ({SO_KY-CUA_SO_GAN} học + {CUA_SO_GAN} đo phong độ)</p>')
    h.append(f'<div style="background:#fff8e1;border-left:4px solid #f9a825;'
             f'padding:10px 13px;margin:0 0 22px;font-size:13px;line-height:1.6">'
             f'<b>Ba mốc để đọc:</b><br>'
             f'· Vốn {r0["von"]:,}/đài · trúng TB <b>{r0["trung_tb"]:.2f}</b> lần · '
             f'hoà vốn cần <b>{r0["hoa_von"]:.2f}</b> lần · EV {r0["ev"]/r0["von"]:+.1%}<br>'
             f'· <b>KHÔNG</b> dùng "có trúng hay không" làm thước đo — với {SO_CON} số, '
             f'P(trúng ít nhất 1) = 98,2%, gần như luôn trúng kể cả bốc bừa. '
             f'Thước đo là SỐ LẦN trúng.<br>'
             f'· <b>p-value</b> so 5 engine của bạn với mức ảo giác chọn lọc. Nếu cả 12 '
             f'engine đều vô dụng, 5 engine top vẫn đạt ~21,7% vs mốc thật 16,5%.</div>')

    for r in ket:
        mau = "#2e7d32" if r["p_value"] < .05 else "#c62828"
        h.append(f'<div style="margin:26px 0 0;padding:9px 12px;background:#263238;'
                 f'color:#fff;border-radius:5px 5px 0 0">'
                 f'<span style="font-size:17px;font-weight:700">{r["dai"].upper()}</span>'
                 f'<span style="font-size:13px;opacity:.85"> &nbsp;|&nbsp; {r["mien"]}'
                 f' &nbsp;|&nbsp; {thu} {ngay_moc:%d.%m.%Y}</span><br>'
                 f'<span style="font-size:12px;opacity:.7">Kỳ gần nhất {r["ngay_ky_truoc"]}'
                 f' · ĐB {r["db_ky_truoc"]} · {r["n_ky"]} kỳ · {r["n_lo"]} lô</span></div>')
        h.append('<div style="border:1px solid #cfd8dc;border-top:0;'
                 'border-radius:0 0 5px 5px;padding:10px 12px 14px">')
        h.append(f'<div style="font-size:12px;color:#455a64;margin:0 0 4px;'
                 f'font-weight:600">{SO_CON} CON BAO LÔ — bôi đen để copy</div>'
                 f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                 f'font-size:16px;background:#e8f5e9;border-left:4px solid #2e7d32;'
                 f'padding:12px 14px;word-break:break-all;line-height:1.9;'
                 f'letter-spacing:.5px">{",".join(r["so"])}</div>')
        h.append('<div style="font-size:12px;color:#546e7a;margin:8px 0 0">'
                 '<b>Nguồn từng nhóm số:</b><ul style="margin:3px 0;padding-left:18px">')
        for e in r["nguon_e"]:
            h.append(f'<li>{e["ten"]} ({e["co_che"]}, {r["g"]} kỳ gần '
                     f'{e["hr_gan"]:.0%}) → {", ".join(f"{v:02d}" for v in e["dong_gop"])}</li>')
        h.append('</ul></div>')
        h.append(f'<div style="font-size:12px;margin:4px 0 0;color:{mau}">'
                 f'{len(r["nguon_e"])} engine đạt <b>{r["hr5_gan"]:.1%}</b> mỗi con trong '
                 f'{r["g"]} kỳ gần · ảo giác chọn lọc {r["null_tb"]:.1%} · '
                 f'<b>p = {r["p_value"]:.3f}</b> — '
                 f'{"VƯỢT mức ảo giác" if r["p_value"] < .05 else "chưa hơn ảo giác"}</div>')
        h.append('<details style="margin:8px 0 0"><summary style="font-size:12px;'
                 'color:#546e7a;cursor:pointer">Bảng 12 engine</summary>'
                 '<table style="font-size:11px;border-collapse:collapse;margin-top:5px">'
                 '<tr style="background:#eceff1"><th style="padding:3px 7px">#</th>'
                 '<th style="padding:3px 7px;text-align:left">Engine</th>'
                 '<th style="padding:3px 7px">Cơ chế</th>'
                 f'<th style="padding:3px 7px">{r["n_test"]} kỳ</th>'
                 f'<th style="padding:3px 7px">{r["g"]} kỳ gần</th>'
                 '<th style="padding:3px 7px;text-align:left">4 con</th></tr>')
        dung = {e["ten"] for e in r["nguon_e"]}
        for i, e in enumerate(r["bang"], 1):
            bg = "#e8f5e9" if e["ten"] in dung else "#fff"
            h.append(f'<tr style="background:{bg}"><td style="padding:3px 7px">{i}</td>'
                     f'<td style="padding:3px 7px">{e["ten"]}</td>'
                     f'<td style="padding:3px 7px;text-align:center;color:#90a4ae">{e["co_che"]}</td>'
                     f'<td style="padding:3px 7px;text-align:center">{e["hr_all"]:.0%}</td>'
                     f'<td style="padding:3px 7px;text-align:center"><b>{e["hr_gan"]:.0%}</b></td>'
                     f'<td style="padding:3px 7px">{", ".join(f"{v:02d}" for v in e["so"])}</td></tr>')
        h.append('</table></details></div>')

    if loi:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;'
                 'padding:10px 13px;margin:18px 0;font-size:13px"><b>Đài lỗi:</b><ul>')
        for s_, t_, e_ in loi:
            h.append(f'<li>[{s_}] {t_}: <span style="font-family:monospace;font-size:11px">{e_}</span></li>')
        h.append('</ul></div>')

    h.append(f'<hr style="margin:26px 0 10px;border:0;border-top:1px solid #ddd">'
             f'<p style="font-size:12px;color:#888;line-height:1.6">'
             f'Sáu engine cuối bảng (Gan, Bệt, Lộn, Bóng, Kép, Số lạnh) KHÔNG có cơ chế '
             f'vật lý — giữ lại để cuộc thi tự phán xử bằng dữ liệu.<br>'
             f'EV = {TY_LE_TRA:.0f}/100 − 1 = {TY_LE_TRA/100-1:+.1%}, không đổi theo số con '
             f'hay thuật toán. Engine dẫn đầu hôm nay chưa chắc dẫn đầu ngày mai — '
             f'theo dõi cột "{r0["g"]} kỳ gần" qua nhiều ngày.</p></div>')
    return "".join(h)


def gui_email(ket, ngay_moc, loi):
    mk = E._lay_mat_khau()
    thu = E.THU_VN[ngay_moc.weekday()]
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[{SO_CON} CON BAO LÔ] {thu} {ngay_moc:%d.%m.%Y} — {len(ket)} đài MN/MT"
    msg["From"] = formataddr((f"XSMN {SO_CON} Con Lô", EMAIL_GUI)); msg["To"] = EMAIL_NHAN
    t = [f"{SO_CON} CON BAO LÔ — {thu} {ngay_moc:%d.%m.%Y}", ""]
    for r in ket:
        t.append("=" * 60)
        t.append(f"{r['dai'].upper()} | {r['mien']} | kỳ trước {r['ngay_ky_truoc']}")
        t.append(f"(vốn {r['von']:,} · trúng TB {r['trung_tb']:.2f} · "
                 f"hoà vốn {r['hoa_von']:.2f} · p {r['p_value']:.3f})")
        t.append(",".join(r["so"]))
        t.append("")
    msg.attach(MIMEText("\n".join(t), "plain", "utf-8"))
    msg.attach(MIMEText(_html(ket, ngay_moc, loi), "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk); sv.send_message(msg)
    print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")


# ==============================================================================
#  PIPELINE
# ==============================================================================

def main(ngay=None, so_ky=None, tong=None, gui_mail=True):
    ngay_moc = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    so_ky = so_ky or SO_KY
    tong = tong or SO_CON
    thu = E.THU_VN[ngay_moc.weekday()]

    print("=" * 80)
    print(f"  {tong} CON BAO LÔ — 12 ENGINE THI ĐẤU  |  {thu} {ngay_moc:%d.%m.%Y}")
    print(f"  {so_ky} kỳ ({so_ky-CUA_SO_GAN} học + {CUA_SO_GAN} đo phong độ) | "
          f"mỗi engine {CON_MOI_ENG} con | chọn {SO_ENGINE} engine")
    print(f"  CHỈ Miền Nam + Miền Trung (18 lô). Miền Bắc bỏ qua.")
    print("=" * 80)

    dsach = [s for s in E.dai_theo_ngay(ngay_moc) if E.lay_dai(s)[1] != "xsmb"]
    lich = E.xay_lich()
    print(f"\n  Đài MN/MT quay hôm nay: {len(dsach)}")

    ket, loi = [], []
    for s in dsach:
        ten = lich[str(s)]["ten"]
        try:
            r = chay_dai(s, ngay_moc, so_ky, tong)
            ket.append(r)
            print(f"     ✓ [{s:>2}] {ten:<20} {r['n_ky']} kỳ ({r['nguon']}) | "
                  f"p={r['p_value']:.2f} | {len(r['nguon_e'])} engine")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<20} LỖI: {e}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")

    r0 = ket[0]
    print(f"\n[A] KINH TẾ MỖI ĐÀI")
    print("-" * 80)
    print(f"  {tong} con × {r0['n_lo']} lô × {DIEM_MOI_CON} điểm = vốn {r0['von']:,}")
    print(f"  Trúng trung bình {r0['trung_tb']:.2f} lần/kỳ · "
          f"hoà vốn cần {r0['hoa_von']:.2f} lần · thiếu {r0['hoa_von']-r0['trung_tb']:.2f}")
    print(f"  EV = {r0['ev']:+,.1f}/đài = {r0['ev']/r0['von']:+.1%}   "
          f"(tổng {len(ket)} đài: {r0['ev']*len(ket):+,.0f})")
    print(f"\n  XÁC SUẤT ĐẠT MỤC TIÊU (giả định ngẫu nhiên, {tong} con / {r0['n_lo']} lô)")
    print("  " + "-" * 50)
    for m, pr in r0["muc_tieu"].items():
        thanh = "█" * int(pr * 30)
        print(f"     >= {m:>2} con trúng   {pr:>5.0%}  {thanh}")
    print(f"\n  Số CON khác nhau trúng TB: {r0['trung_con']:.2f}  |  "
          f"số LƯỢT trúng TB: {r0['trung_tb']:.2f}")
    print(f"\n  ⚠ Tăng số con làm TRÚNG NHIỀU HƠN nhưng KHÔNG đổi EV.")
    print(f"    20 con: trúng TB 3,31 · hoà vốn 3,79 · EV -5,0%")
    print(f"    {tong} con: trúng TB {r0['trung_con']:.2f} · hoà vốn {r0['hoa_von']:.2f} · EV -5,0%")
    print(f"    Vốn tăng ĐÚNG BẰNG mức trúng tăng. 'Trúng nhiều con' mua bằng tiền,")
    print(f"    không phải bằng thuật toán.")

    print(f"\n[B] ENGINE NÀO MẠNH — gộp mọi đài hôm nay")
    print("-" * 80)
    gop = {t: {"all": [], "gan": [], "dung": 0} for t in TEN_ENGINE}
    for r in ket:
        d = {e["ten"] for e in r["nguon_e"]}
        for e in r["bang"]:
            gop[e["ten"]]["all"].append(e["hr_all"])
            gop[e["ten"]]["gan"].append(e["hr_gan"])
            if e["ten"] in d:
                gop[e["ten"]]["dung"] += 1
    print(f"  {'Engine':<16}{'Cơ chế':>9}{'Hit toàn kỳ':>13}{'Hit 10 kỳ gần':>15}"
          f"{'Được dùng':>12}")
    print("  " + "-" * 76)
    for t in sorted(TEN_ENGINE, key=lambda x: -np.mean(gop[x]["all"])):
        g = gop[t]
        print(f"  {t:<16}{CO_CHE[t]:>9}{np.mean(g['all']):>12.1%}"
              f"{np.mean(g['gan']):>15.1%}{g['dung']:>8}/{len(ket):<3}")
    print(f"\n  Mốc ngẫu nhiên mỗi con: 16,5%   (= 1 − 0,99^18)")

    for r in ket:
        print("\n" + "=" * 80)
        print(f"  {r['dai'].upper()}  —  {tong} CON BAO LÔ")
        print(f"  {thu} {ngay_moc:%d.%m.%Y}  |  {r['mien']}  |  "
              f"{r['n_lo']} lô  |  kỳ trước {r['ngay_ky_truoc']} (ĐB {r['db_ky_truoc']})")
        print("=" * 80)
        print(f"\n  {','.join(r['so'])}\n")
        print(f"  Nguồn từng nhóm số:")
        for e in r["nguon_e"]:
            print(f"     {e['ten']:<16}({e['co_che']:>6}, {r['g']} kỳ gần {e['hr_gan']:>5.0%}) "
                  f"→ {', '.join(f'{v:02d}' for v in e['dong_gop'])}")
        print(f"\n  {len(r['nguon_e'])} engine đạt {r['hr5_gan']:.1%} mỗi con trong {r['g']} kỳ gần"
              f"  |  ảo giác {r['null_tb']:.1%}  |  p = {r['p_value']:.3f}")
        print("  → " + ("VƯỢT mức ảo giác chọn lọc — ghi lại theo dõi."
                        if r["p_value"] < .05 else
                        "CHƯA hơn ảo giác. Chọn 5/12 engine luôn cho con số đẹp"
                        " kể cả khi 12 engine đều vô dụng."))
        print(f"\n  BẢNG 12 ENGINE")
        print(f"  {'#':>3} {'Engine':<16}{'Cơ chế':>8}{r['n_test']:>7} kỳ{r['g']:>8} kỳ gần"
              f"   4 con")
        print("  " + "-" * 74)
        dung = {e["ten"] for e in r["nguon_e"]}
        for i, e in enumerate(r["bang"], 1):
            dau = " ←" if e["ten"] in dung else ""
            print(f"  {i:>3} {e['ten']:<16}{e['co_che']:>8}{e['hr_all']:>9.0%}"
                  f"{e['hr_gan']:>13.0%}   {', '.join(f'{v:02d}' for v in e['so'])}{dau}")
    print("\n" + "=" * 80)

    if gui_mail:
        print("\n  Đang gửi email...")
        try:
            gui_email(ket, ngay_moc, loi)
        except Exception as e:
            print(f"  ✗ Không gửi được email: {e}")
    return ket
