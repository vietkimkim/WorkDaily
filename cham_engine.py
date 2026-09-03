"""
CHAM ENGINE v3 — 4 CHẠM cho Đề Đặc Biệt · Đề Giải Nhất · Đề Đầu (G8)

BỐN SỬA LỖI so với v2 (mang từ cham_final.py sang):

  1. TRỌNG SỐ THỜI GIAN CHO MỌI CHỈ SỐ
     v2 chỉ tính trọng số cho tần suất; chuỗi dài nhất và số lần trúng tính thô.
     -> cầu trúng 5 kỳ liên tiếp CÁCH ĐÂY 2 NĂM ngang cầu vừa trúng tháng trước.
     v3: mọi chỉ số đều giảm dần. Nửa đời đặt theo mốc 6 THÁNG, không phải số kỳ.

  2. TÍN HIỆU ĐỘ ĐỀU ĐẶN
     v2 chỉ đo chuỗi DÀI NHẤT. Hai cầu cùng trúng 6/20 nhưng khác hẳn:
        A: trúng kỳ 1-6 rồi TẮT HẲN     -> chuỗi max 6
        B: trúng kỳ 3,6,9,12,15,18      -> chuỗi max 1, nhưng ĐỀU
     v2 cho A điểm cao hơn. v3 thêm C4_DeuDan (hệ số biến thiên khoảng cách).

  3. KIỂM ĐỊNH GỐC: CẦU CÓ TỒN TẠI KHÔNG
     v2 luôn đi TÌM cầu tốt nhất, chưa hỏi câu gốc.
     v3: kiểm định PHƯƠNG SAI hit rate giữa các vị trí. Dưới độc lập mọi vị trí
     quanh 19%; có cấu trúc cầu thì phương sai vượt mức nhiễu nhị thức.
     Đây là MỘT phép kiểm định -> KHÔNG bị vấn đề so sánh bội.

  4. GỘP MÔ HÌNH XÁC SUẤT + CẦU CHẠM
     v2 chạy tách rời. v3 gộp thành 5 tín hiệu, trọng số do walk-forward chọn
     trong 31 tổ hợp nhị phân.
"""
import os, smtplib, itertools, re
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

SO_KY_CHAM   = 10     # số kỳ lịch sử. 10 = chỉ nhìn cầu ĐANG chạy gần đây.
                      # ⚠ 10 kỳ -> 9 lần kiểm chứng/cầu. Cầu "mạnh nhất" trong
                      #   13 vị trí đạt 42% CHỈ DO MAY (mốc thật 19%).
                      #   Đo trên nhiễu: cầu chạm 52,1% vs bốc bừa 51,8% -> chênh
                      #   0,3 điểm, nằm trong sai số ±1,1%. KHÔNG phân biệt được.
SO_CHAM      = 3      # 3 chạm phủ 51 số (mốc 51,00%, hoà vốn 51/95 = 53,68%)
                      # Thắng +44 / thua -51 -> tỷ lệ 0,86, tốt hơn hẳn 4 chạm (0,48)
CHAY_DB      = True   # ① Đề Đặc Biệt
CHAY_G1      = True   # ② Đề Giải Nhất
CHAY_G8      = True   # ③ Đề Đầu (giải 8) — chỉ MN/MT, Miền Bắc tự bỏ qua

# --- Trọng số thời gian: khai theo MỐC 6 THÁNG, không phải số kỳ ---
CON_LAI_6TH  = 0.10   # với 10 kỳ, trọng số thời gian gần như không tác dụng —
                      # 10 kỳ đài tuần chỉ là 2,3 tháng, mọi kỳ đều "gần"

# --- Nguồn vị trí cầu ---
NGUON_CAU    = "GON"  # "GON" = chỉ ĐB+G1(+G8): 10-12 vị trí, ÍT OVERFIT
                      # "DAY" = toàn bảng giải: 82-107 vị trí, nhiều ứng viên hơn
                      #         nhưng ngưỡng nhiễu cao hơn hẳn

TOP_CAU      = 5      # số cầu tốt nhất được bỏ phiếu
MIN_KY       = 8      # tối thiểu để chạy được
N_NULL       = 2000
N_BOOTSTRAP  = 300
BO_BOT       = 0.20
TY_LE_TRA    = 95.0

EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")

TEN_TH = ["C1_TanSuatW", "C2_ChuoiNay", "C3_ChuoiMaxW", "C4_DeuDan", "C5_MoHinh"]
MO_TA = {"DB": ("① ĐỀ ĐẶC BIỆT", "2 chữ số CUỐI giải ĐẶC BIỆT"),
         "G1": ("② ĐỀ GIẢI NHẤT", "2 chữ số CUỐI giải NHẤT"),
         "G8": ("③ ĐỀ ĐẦU (G8)", "giải TÁM — đã là số 2 chữ số")}


# ==============================================================================
#  TẦNG DỮ LIỆU
# ==============================================================================

def _nua_doi(ky_moi_tuan=1):
    """Nửa đời (kỳ) sao cho kỳ cách 6 THÁNG còn đúng CON_LAI_6TH trọng số."""
    return max(4.0, 26.0 * ky_moi_tuan) / max(np.log2(1 / CON_LAI_6TH), 0.1)


def _vi_tri_g8(toan_giai):
    """G8 = phần tử cuối của bảng 18 số (MN/MT). Miền Bắc không có -> None."""
    if len(toan_giai[0]) != 18 or len(toan_giai[0][-1]) != 2:
        return None
    return len(toan_giai[0]) - 1


def _nguon_chu_so(toan_giai, vt_g8):
    """Chọn các số làm NGUỒN cầu. GON = ĐB+G1(+G8). DAY = toàn bảng."""
    if NGUON_CAU.upper() == "DAY":
        ten = [f"#{i}[{j+1}]" for i, s in enumerate(toan_giai[0]) for j in range(len(s))]
        M = np.array([[int(c) for s in ky for c in s] for ky in toan_giai], dtype=np.int8)
        return M, ten
    idx = [0, 1] + ([vt_g8] if vt_g8 is not None else [])
    nhan = ["ĐB", "G1", "G8"]
    ten = [f"{nhan[a]}[{j+1}]" for a, i in enumerate(idx)
           for j in range(len(toan_giai[0][i]))]
    M = np.array([[int(c) for i in idx for c in ky[i]] for ky in toan_giai], dtype=np.int8)
    return M, ten


# ==============================================================================
#  QUÉT CẦU — MỌI CHỈ SỐ CÓ TRỌNG SỐ THỜI GIAN  (sửa lỗi 1 + 2)
# ==============================================================================

def quet(toan_giai, muc_tieu, vt_g8, h):
    """muc_tieu: list số 2 chữ số của giải đang dự báo, theo từng kỳ."""
    M, ten = _nguon_chu_so(toan_giai, vt_g8)
    n, V = M.shape
    cham = [{int(m[-2]), int(m[-1])} for m in muc_tieu]

    hit = np.zeros((V, n - 1), dtype=bool)
    w = np.zeros(n - 1)
    for t in range(n - 1):
        hit[:, t] = np.isin(M[t], list(cham[t + 1]))
        w[t] = 0.5 ** ((n - 2 - t) / h)

    trung_w = (hit * w).sum(axis=1) / w.sum()
    so_lan = hit.sum(axis=1)

    chuoi_max_w = np.zeros(V); chuoi_nay = np.zeros(V, dtype=int)
    for v in range(V):
        cur = 0
        for t in range(n - 1):
            cur = cur + 1 if hit[v, t] else 0
            chuoi_max_w[v] = max(chuoi_max_w[v], cur * w[t])   # chuỗi CŨ bị hạ điểm
        chuoi_nay[v] = cur

    # --- ĐỘ ĐỀU ĐẶN: khoảng cách giữa các lần trúng ít biến động ---
    deu = np.zeros(V)
    cua_so_gan = max(3, int(h))
    for v in range(V):
        idx = np.where(hit[v])[0]
        if len(idx) < 3:
            continue
        gap = np.diff(idx)
        deu[v] = 1.0 / (1.0 + gap.std() / max(gap.mean(), 1e-9))
    deu = deu * hit[:, -cua_so_gan:].any(axis=1)   # chỉ thưởng cầu còn hoạt động

    return {"M": M, "ten": ten, "hit": hit, "w": w, "trung_w": trung_w,
            "so_lan": so_lan, "chuoi_max_w": chuoi_max_w, "chuoi_nay": chuoi_nay,
            "deu": deu, "n_test": n - 1, "du_bao": M[n - 1], "h": h}


# ==============================================================================
#  KIỂM ĐỊNH GỐC: CẦU CÓ TỒN TẠI KHÔNG  (sửa lỗi 3)
# ==============================================================================

def kiem_dinh_cau(q, n_lap=N_NULL, seed=2026):
    V, ne = q["hit"].shape
    hr = q["hit"].mean(axis=1)
    p = 1 - 0.81
    ty_so = hr.var() / (p * (1 - p) / ne)
    rng = np.random.default_rng(seed)
    null = (rng.random((n_lap, V, ne)) < p).mean(axis=2).var(axis=1) / (p * (1 - p) / ne)
    return {"ty_so": float(ty_so), "p_value": float((null >= ty_so).mean()),
            "null_tb": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "V": V, "ne": ne}


# ==============================================================================
#  GỘP MÔ HÌNH + CẦU  (sửa lỗi 4)
# ==============================================================================

def _z(v):
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-12 else (v - v.mean()) / sd


def _phan_phoi_cham(muc_tieu, h):
    n = len(muc_tieu)
    d = np.ones(10)
    for t, m in enumerate(muc_tieu):
        w = 0.5 ** ((n - 1 - t) / h)
        for c in {int(m[-2]), int(m[-1])}:
            d[c] += w
    return d / d.sum()


def chon_cham(q, muc_tieu, w_th, k=SO_CHAM, top=TOP_CAU):
    diem_cau = (w_th[0] * _z(q["trung_w"]) + w_th[1] * _z(q["chuoi_nay"])
                + w_th[2] * _z(q["chuoi_max_w"]) + w_th[3] * _z(q["deu"]))
    phieu = np.zeros(10)
    if any(w_th[:4]):
        tot = np.argsort(-diem_cau)[:min(top, len(diem_cau))]
        nen = diem_cau[tot].min()
        for j in tot:
            phieu[q["du_bao"][j]] += max(diem_cau[j] - nen + 0.1, 0.1)
        phieu = _z(phieu)
    mh = _z(np.log(_phan_phoi_cham(muc_tieu, q["h"]))) if w_th[4] else np.zeros(10)
    d = phieu + w_th[4] * mh
    return tuple(int(x) for x in sorted(np.argsort(-d)[:k])), d, diem_cau


def chon_trong_so(toan_giai, muc_tieu, vt_g8, h, k=SO_CHAM):
    """Trọng số do WALK-FORWARD chọn. 31 tổ hợp nhị phân."""
    n = len(muc_tieu); mt = max(4, n // 2)
    best = (None, -1.0, 0)
    for w_th in itertools.product([0.0, 1.0], repeat=5):
        if sum(w_th) == 0:
            continue
        trung = tong = 0
        for t in range(mt, n):
            qq = quet(toan_giai[:t], muc_tieu[:t], vt_g8, h)
            bo, _, _ = chon_cham(qq, muc_tieu[:t], w_th, k)
            s = set(bo); m = muc_tieu[t]
            trung += int(m[-2]) in s or int(m[-1]) in s
            tong += 1
        hr = trung / max(tong, 1)
        if hr > best[1]:
            best = (w_th, hr, tong)
    return best


def do_on_dinh(toan_giai, muc_tieu, vt_g8, h, w_th, k=SO_CHAM,
               n_lap=N_BOOTSTRAP, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(muc_tieu); giu = max(MIN_KY, int(n * (1 - BO_BOT)))
    dem = np.zeros(10)
    for _ in range(n_lap):
        idx = sorted(rng.choice(n, min(giu, n), replace=False))
        tg = [toan_giai[i] for i in idx]; mt = [muc_tieu[i] for i in idx]
        try:
            bo, _, _ = chon_cham(quet(tg, mt, vt_g8, h), mt, w_th, k)
        except Exception:
            continue
        for c in bo:
            dem[c] += 1
    return dem / n_lap


def so_cua_cham(bo):
    s = set(bo)
    return [f"{v:02d}" for v in range(100) if (v // 10 in s) or (v % 10 in s)]


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p, den = k / n, 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / den
    return (max(0., ctr - hw), min(1., ctr + hw))


# ==============================================================================
#  CHẠY 1 ĐÀI
# ==============================================================================

def chay_dai(stt, ngay_moc, so_ky=None, so_cham=None):
    so_ky = so_ky or SO_KY_CHAM
    so_cham = so_cham or SO_CHAM
    ten, ma, mien, nd = E.lay_dai(stt)

    m = E.lay_tu_master(stt, so_ky + 20)
    if m:
        toan_giai, ngay_full, info = m; nguon = "KHO"
    else:
        _, ngay, ngay_full, toan_giai, info = E.lay_du_lieu(stt, min(200, so_ky + 30))
        nguon = "WEB"
    if not toan_giai:
        raise RuntimeError("không bóc được toàn bảng giải")

    tg, ng, that = E.cat_truoc_ngay(toan_giai, ngay_full, ngay_moc, so_ky)
    if len(tg) < MIN_KY:
        raise RuntimeError(f"chỉ còn {len(tg)} kỳ trước {ngay_moc:%d.%m.%Y}, cần >={MIN_KY}")

    # số kỳ mỗi tuần -> nửa đời theo mốc 6 tháng
    gaps = [(ng[i+1] - ng[i]).days for i in range(len(ng)-1)]
    cach = max(set(gaps), key=gaps.count) if gaps else 7
    h = _nua_doi(max(1.0, 7.0 / max(cach, 1)))

    vt_g8 = _vi_tri_g8(tg)
    ds = [("DB", CHAY_DB, 0), ("G1", CHAY_G1, 1)]
    if CHAY_G8 and vt_g8 is not None:
        ds.append(("G8", True, vt_g8))

    mods = []
    for khoa, bat, vi_tri in ds:
        if not bat:
            continue
        muc_tieu = [ky[vi_tri][-2:] for ky in tg]
        w_th, hr_wf, n_wf = chon_trong_so(tg, muc_tieu, vt_g8, h, so_cham)
        q = quet(tg, muc_tieu, vt_g8, h)
        bo, diem, diem_cau = chon_cham(q, muc_tieu, w_th, so_cham)
        on_dinh = do_on_dinh(tg, muc_tieu, vt_g8, h, w_th, so_cham)
        kd = kiem_dinh_cau(q)
        tot = np.argsort(-diem_cau)[:TOP_CAU]
        lo, hi = wilson(int(hr_wf * n_wf), n_wf)
        p_truot = ((10 - so_cham) / 10) ** 2

        mods.append({
            "key": khoa, "ten": MO_TA[khoa][0], "mo_ta": MO_TA[khoa][1],
            "cham": list(bo), "so": so_cua_cham(bo), "n_so": int(100 * (1 - p_truot)),
            "moc": 1 - p_truot, "hoa_von": 100 * (1 - p_truot) / TY_LE_TRA,
            "hr_wf": hr_wf, "n_wf": n_wf, "ktc": (lo, hi),
            "trong_so": [t for t, x in zip(TEN_TH, w_th) if x],
            "on_dinh": {int(c): float(on_dinh[c]) for c in bo},
            "sat_nut": sorted([(int(d), float(on_dinh[d])) for d in range(10)
                               if d not in bo and on_dinh[d] >= .35], key=lambda x: -x[1]),
            "tu_tin": float(np.mean([on_dinh[c] for c in bo])),
            "kiem_dinh": kd,
            "cau": [{"ten": q["ten"][j], "cham": int(q["du_bao"][j]),
                     "so_lan": int(q["so_lan"][j]), "trung_w": float(q["trung_w"][j]),
                     "deu": float(q["deu"][j]), "chuoi_nay": int(q["chuoi_nay"][j])}
                    for j in tot],
            "n_test": q["n_test"],
            "_tong_hit": int(q["hit"].sum()), "_tong_qs": int(q["hit"].size),
        })

    return {"stt": stt, "dai": ten, "mien": mien, "nguon": nguon, "n_ky": len(tg),
            "nua_doi": h, "cach_ngay": cach, "n_vi_tri": len(mods[0]["cau"]) if mods else 0,
            "ngay_ky_truoc": ng[-1].strftime("%d/%m/%Y"),
            "db_ky_truoc": tg[-1][0], "g1_ky_truoc": tg[-1][1],
            "g8_ky_truoc": tg[-1][vt_g8] if vt_g8 is not None else None,
            "modules": mods, "ket_qua_that": that}


# ==============================================================================
#  KIỂM ĐỊNH GỘP LAG-1 — gộp MỌI đài trong ngày
#
#  Với 10 kỳ mỗi đài, riêng lẻ không đủ mẫu. Nhưng gộp 6 đài × 3 giải × 13 vị trí
#  × 9 lần kiểm chứng = ~2.100 quan sát -> ĐỦ để trả lời câu hỏi gốc:
#
#      "Chữ số của kỳ TRƯỚC có làm chạm cho kỳ SAU nhiều hơn 19% không?"
#
#  Đây là MỘT phép kiểm định trên toàn bộ dữ liệu, không phải dò 13 cầu rồi lấy
#  cái tốt nhất -> KHÔNG bị vấn đề so sánh bội.
# ==============================================================================

def kiem_dinh_gop(ket):
    """Gộp mọi (đài × giải) trong ngày, kiểm định lag-1 có tồn tại không."""
    tong_hit = tong_qs = 0
    for r in ket:
        for m in r["modules"]:
            tong_hit += m.get("_tong_hit", 0)
            tong_qs += m.get("_tong_qs", 0)
    if tong_qs == 0:
        return None
    p0 = 1 - 0.81
    hr = tong_hit / tong_qs
    se = (p0 * (1 - p0) / tong_qs) ** 0.5
    z = (hr - p0) / se if se > 0 else 0.0
    from math import erfc
    p = erfc(abs(z) / (2 ** 0.5))
    return {"hit": tong_hit, "qs": tong_qs, "hr": hr, "p0": p0,
            "z": z, "p_value": p, "se": se}


# ==============================================================================
#  EMAIL
# ==============================================================================

def _html(ket, ngay_moc, loi):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    thu = E.THU_VN[ngay_moc.weekday()]
    m0 = ket[0]["modules"][0]
    h = [f'<div style="{css}max-width:700px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">{SO_CHAM} chạm — {thu} {ngay_moc:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài · {SO_KY_CHAM} kỳ · nguồn cầu {NGUON_CAU} · '
             f'mỗi bộ phủ {m0["n_so"]}/100 số</p>')
    h.append(f'<div style="background:#fff8e1;border-left:4px solid #f9a825;'
             f'padding:10px 13px;margin:0 0 22px;font-size:13px;line-height:1.6">'
             f'<b>Đọc theo thứ tự:</b><br>'
             f'1. <b>p cấu trúc cầu</b> — có liên hệ giữa chữ số kỳ trước và chạm kỳ sau '
             f'không. Đây là 1 phép kiểm định, KHÔNG bị so sánh bội. Từ 0,05 trở lên '
             f'nghĩa là bộ chạm ngang bốc bừa.<br>'
             f'2. <b>Hit rate ngoài mẫu</b> — mốc {m0["moc"]:.0%}, hoà vốn {m0["hoa_von"]:.2%}<br>'
             f'3. <b>Độ ổn định</b> — bộ chạm có vững không (KHÔNG phải xác suất trúng)'
             f'</div>')

    for r in ket:
        h.append(f'<div style="margin:26px 0 0;padding:9px 12px;background:#263238;'
                 f'color:#fff;border-radius:5px 5px 0 0">'
                 f'<span style="font-size:17px;font-weight:700">{r["dai"].upper()}</span>'
                 f'<span style="font-size:13px;opacity:.85"> &nbsp;|&nbsp; {r["mien"]}'
                 f' &nbsp;|&nbsp; {thu} {ngay_moc:%d.%m.%Y}</span><br>'
                 f'<span style="font-size:12px;opacity:.7">Kỳ gần nhất {r["ngay_ky_truoc"]}'
                 f' · ĐB {r["db_ky_truoc"]} · G1 {r["g1_ky_truoc"]}'
                 + (f' · G8 {r["g8_ky_truoc"]}' if r.get("g8_ky_truoc") else '')
                 + f' · {r["n_ky"]} kỳ · nửa đời {r["nua_doi"]:.0f} kỳ</span></div>')
        h.append('<div style="border:1px solid #cfd8dc;border-top:0;'
                 'border-radius:0 0 5px 5px;padding:4px 12px 14px">')
        for m in r["modules"]:
            kd = m["kiem_dinh"]
            co_cau = kd["p_value"] < 0.05
            mau = "#2e7d32" if co_cau else "#c62828"
            h.append(f'<div style="margin:14px 0 0">'
                     f'<div style="font-size:14px;font-weight:600">{m["ten"]}</div>'
                     f'<div style="font-size:12px;color:#607d8b;margin:2px 0 6px">'
                     f'{r["dai"]} · {thu} {ngay_moc:%d.%m.%Y} · {m["mo_ta"]}</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:26px;font-weight:700;letter-spacing:8px;'
                     f'background:#e8f5e9;border-left:4px solid #2e7d32;'
                     f'padding:12px 14px">{" ".join(str(c) for c in m["cham"])}</div>')
            h.append(f'<div style="font-size:12px;margin:7px 0 0;color:{mau}">'
                     f'<b>Cấu trúc cầu:</b> tỷ số phương sai {kd["ty_so"]:.2f} '
                     f'(ngưỡng {kd["null_p95"]:.2f}) · <b>p = {kd["p_value"]:.3f}</b> — '
                     f'{"CÓ bằng chứng cấu trúc cầu" if co_cau else "KHÔNG có bằng chứng cấu trúc cầu"}'
                     f'</div>')
            h.append(f'<div style="font-size:12px;color:#546e7a;margin:4px 0 0">'
                     f'Hit rate ngoài mẫu <b>{m["hr_wf"]:.1%}</b> ({m["n_wf"]} kỳ, '
                     f'KTC95 [{m["ktc"][0]:.0%}, {m["ktc"][1]:.0%}]) · mốc {m["moc"]:.0%} · '
                     f'hoà vốn {m["hoa_von"]:.2%}<br>'
                     f'Trọng số: {", ".join(m["trong_so"]) or "(không)"} · '
                     f'Độ ổn định: '
                     + " · ".join(f'chạm {c} {v:.0%}' for c, v in m["on_dinh"].items())
                     + f' → TB <b>{m["tu_tin"]:.0%}</b>'
                     + (f'<br>Sát nút: ' + ", ".join(f'chạm {d} ({v:.0%})'
                                                     for d, v in m["sat_nut"]) if m["sat_nut"] else '')
                     + '</div>')
            h.append('<div style="font-size:11px;color:#78909c;margin:6px 0 0">'
                     '<b>Cầu hàng đầu:</b> '
                     + " · ".join(f'{c["ten"]}→{c["cham"]} ({c["so_lan"]}/{m["n_test"]}, '
                                  f'đều {c["deu"]:.2f}'
                                  + (f', chạy {c["chuoi_nay"]}' if c["chuoi_nay"] else '') + ')'
                                  for c in m["cau"][:4]) + '</div>')
            h.append(f'<div style="font-size:12px;color:#455a64;margin:10px 0 3px;'
                     f'font-weight:600">{m["n_so"]} SỐ QUY RA — bôi đen để copy</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:14px;background:#eceff1;border-left:3px solid #455a64;'
                     f'padding:10px 12px;word-break:break-all;line-height:1.8">'
                     f'{",".join(m["so"])}</div></div>')
        h.append('</div>')

    if loi:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;'
                 'padding:10px 13px;margin:18px 0;font-size:13px"><b>Đài lỗi:</b><ul>')
        for s_, t_, e_ in loi:
            h.append(f'<li>[{s_}] {t_}: <span style="font-family:monospace;font-size:11px">'
                     f'{e_}</span></li>')
        h.append('</ul></div>')

    h.append(f'<hr style="margin:26px 0 10px;border:0;border-top:1px solid #ddd">'
             f'<p style="font-size:12px;color:#888;line-height:1.6">'
             f'Trọng số thời gian: kỳ cách 6 tháng còn {CON_LAI_6TH:.0%}, cách 1 năm còn '
             f'{CON_LAI_6TH**2:.1%} — mọi chỉ số đều giảm dần, không riêng tần suất.<br>'
             f'Độ ổn định đo ĐỘ VỮNG CỦA VIỆC CHỌN, không phải xác suất trúng. '
             f'{SO_CHAM} chạm phủ {m0["n_so"]}/100 số → P(trúng) = {m0["moc"]:.0%} '
             f'trừ khi máy quay lệch thật.</p></div>')
    return "".join(h)


def gui_email(ket, ngay_moc, loi):
    mk = E._lay_mat_khau()
    thu = E.THU_VN[ngay_moc.weekday()]
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[{SO_CHAM} CHẠM] {thu} {ngay_moc:%d.%m.%Y} — {len(ket)} đài"
    msg["From"] = formataddr((f"XSMN {SO_CHAM} Chạm", EMAIL_GUI)); msg["To"] = EMAIL_NHAN
    t = [f"{SO_CHAM} CHẠM {thu} {ngay_moc:%d.%m.%Y}", ""]
    for r in ket:
        t.append("=" * 60)
        t.append(f"{r['dai'].upper()} | {r['mien']} | kỳ trước {r['ngay_ky_truoc']}")
        for m in r["modules"]:
            t.append("")
            t.append(f"{m['ten']} — {r['dai']} — {thu} {ngay_moc:%d.%m.%Y}")
            t.append(f"CHẠM: {' '.join(str(c) for c in m['cham'])}")
            t.append(f"(p cấu trúc cầu {m['kiem_dinh']['p_value']:.3f} | "
                     f"ngoài mẫu {m['hr_wf']:.1%} | ổn định {m['tu_tin']:.0%})")
            t.append(",".join(m["so"]))
        t.append("")
    msg.attach(MIMEText("\n".join(t), "plain", "utf-8"))
    msg.attach(MIMEText(_html(ket, ngay_moc, loi), "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk); sv.send_message(msg)
    print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")


# ==============================================================================
#  PIPELINE
# ==============================================================================

def main(ngay=None, so_ky=None, so_cham=None, gui_mail=True):
    ngay_moc = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    so_ky = so_ky or SO_KY_CHAM
    so_cham = so_cham or SO_CHAM
    thu = E.THU_VN[ngay_moc.weekday()]
    p_truot = ((10 - so_cham) / 10) ** 2

    print("=" * 78)
    print(f"  {so_cham} CHẠM v3  |  {thu} {ngay_moc:%d.%m.%Y}  |  {so_ky} kỳ  |  "
          f"nguồn cầu {NGUON_CAU}")
    print(f"  Phủ {int(100*(1-p_truot))}/100 số | mốc {1-p_truot:.2%} | "
          f"hoà vốn {100*(1-p_truot)/TY_LE_TRA:.2%} | 6 tháng trước còn {CON_LAI_6TH:.0%}")
    print("=" * 78)

    dsach = E.dai_theo_ngay(ngay_moc)
    lich = E.xay_lich()
    print(f"\n  Đài quay hôm nay: {len(dsach)}")

    ket, loi = [], []
    for s in dsach:
        ten = lich[str(s)]["ten"]
        try:
            r = chay_dai(s, ngay_moc, so_ky, so_cham)
            ket.append(r)
            ct = " | ".join(f"{m['ten'][0]}:{''.join(map(str,m['cham']))}"
                            f"(p={m['kiem_dinh']['p_value']:.2f})" for m in r["modules"])
            print(f"     ✓ [{s:>2}] {ten:<20} {r['n_ky']} kỳ ({r['nguon']}) | {ct}")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<20} LỖI: {e}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")

    kdg = kiem_dinh_gop(ket)
    if kdg:
        print(f"\n[0] KIỂM ĐỊNH GỘP LAG-1  —  gộp MỌI đài × giải trong ngày")
        print("-" * 78)
        print(f"  Câu hỏi: chữ số kỳ TRƯỚC có làm chạm kỳ SAU nhiều hơn {kdg['p0']:.0%} không?")
        print(f"  Quan sát: {kdg['hit']:,}/{kdg['qs']:,} = {kdg['hr']:.2%}  "
              f"(mốc độc lập {kdg['p0']:.2%}, sai số {kdg['se']:.3%})")
        print(f"  z = {kdg['z']:+.2f}   p = {kdg['p_value']:.4f}")
        print("  → " + ("CÓ liên hệ lag-1. Đây là bằng chứng cầu tồn tại — đáng đào sâu."
                        if kdg["p_value"] < .05 and kdg["z"] > 0 else
                        "KHÔNG có liên hệ lag-1. Chữ số kỳ trước không dự báo được chạm kỳ sau."))
        print(f"  (1 phép kiểm định trên toàn bộ dữ liệu — KHÔNG bị so sánh bội)")

    print(f"\n[A] CẤU TRÚC CẦU CÓ TỒN TẠI KHÔNG  (1 phép kiểm định/giải, không so sánh bội)")
    print("-" * 78)
    print(f"  {'Đài':<18}{'Giải':<18}{'Tỷ số PS':>10}{'Ngưỡng':>9}{'p':>8}")
    print("  " + "-" * 74)
    for r in ket:
        for i, m in enumerate(r["modules"]):
            kd = m["kiem_dinh"]
            dau = " ← CÓ" if kd["p_value"] < .05 else ""
            print(f"  {r['dai'] if i==0 else '':<18}{m['ten']:<18}"
                  f"{kd['ty_so']:>10.2f}{kd['null_p95']:>9.2f}{kd['p_value']:>8.3f}{dau}")
    print("\n  p < 0,05 = CÓ liên hệ giữa chữ số kỳ trước và chạm kỳ sau.")
    print("  p ≥ 0,05 = mọi vị trí hành xử như nhau, bộ chạm ngang bốc bừa.")

    for r in ket:
        for m in r["modules"]:
            print("\n" + "=" * 78)
            print(f"  {r['dai'].upper()}  —  {m['ten']}")
            print(f"  {thu} {ngay_moc:%d.%m.%Y}  |  {r['mien']}  |  {m['mo_ta']}")
            print("=" * 78)
            print(f"\n     CHẠM:   {'   '.join(str(c) for c in m['cham'])}\n")
            print(f"  p cấu trúc cầu {m['kiem_dinh']['p_value']:.3f}  |  "
                  f"ngoài mẫu {m['hr_wf']:.1%} ({m['n_wf']} kỳ, "
                  f"KTC95 [{m['ktc'][0]:.0%},{m['ktc'][1]:.0%}])  |  mốc {m['moc']:.0%}")
            print(f"  Trọng số: {', '.join(m['trong_so']) or '(không)'}")
            print(f"  Độ ổn định: " + "  ".join(f"chạm {c} {v:.0%}"
                                                 for c, v in m["on_dinh"].items())
                  + f"   → TB {m['tu_tin']:.0%}")
            if m["sat_nut"]:
                print(f"  Sát nút: " + ", ".join(f"chạm {d} ({v:.0%})" for d, v in m["sat_nut"]))
            print(f"\n  Cầu hàng đầu (đã tính trọng số thời gian):")
            for c in m["cau"][:4]:
                song = f"ĐANG CHẠY {c['chuoi_nay']}" if c["chuoi_nay"] else "đã gãy"
                print(f"     {c['ten']:<10}→ chạm {c['cham']}   "
                      f"trúng {c['so_lan']}/{m['n_test']}  tần suất W {c['trung_w']:.0%}  "
                      f"đều {c['deu']:.2f}  {song}")
            print(f"\n  {m['n_so']} SỐ QUY RA:")
            print("  " + ",".join(m["so"]))
    print("\n" + "=" * 78)

    if gui_mail:
        print("\n  Đang gửi email...")
        try:
            gui_email(ket, ngay_moc, loi)
        except Exception as e:
            print(f"  ✗ Không gửi được email: {e}")
    return ket
