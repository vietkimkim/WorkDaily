"""
THONGKE6 — 6 engine XÁC SUẤT THỐNG KÊ thuần tuý cho 64 con ĐB · G1 · G8.

NGUYÊN TẮC:
    Mọi engine ước lượng PHÂN PHỐI XÁC SUẤT P(v) trên 100 ô (tổng = 1).
    Mọi engine có MỘT PHÉP KIỂM ĐỊNH GẮN SẴN — nếu kiểm định nói "không có
    tín hiệu", engine TỰ LÙI về phân phối đều thay vì bịa ra quy luật.

SÁU ENGINE, MỖI CÁI MỘT CÂU HỎI:
    E1 Dirichlet  : ước lượng Bayes chuẩn — α tìm bằng MARGINAL LIKELIHOOD
    E2 DocLap     : 2 chữ số có độc lập? Chi-square 10×10 -> 20 hay 100 tham số
    E3 CoNgot     : chênh lệch là tín hiệu hay nhiễu? Hệ số B James-Stein
    E4 GopGiai    : giải mục tiêu cùng phân phối 17 giải kia? Nếu có -> mượn dữ liệu
    E5 DiemGay    : máy quay có thay đổi? Tỷ số hợp lý tìm điểm gãy
    E6 BIC        : mô hình nào tốt nhất SAU KHI PHẠT độ phức tạp?

CÁCH CHỌN — KHÁC HẲN CÁC BẢN TRƯỚC:
    · Chấm bằng LOG-SCORE, không phải hit rate. Đo ở 30 kỳ: log-score chọn đúng
      engine giỏi 66% số lần, hit rate chỉ 55%. Hit rate vứt bỏ thông tin.
    · Kiểm định DIEBOLD-MARIANO so từng engine với phân phối đều.
    · Chỉ engine VƯỢT ĐỀU CÓ Ý NGHĨA mới được gộp, trọng số theo độ vượt.
    · KHÔNG engine nào vượt -> dùng E1, và BÁO THẲNG điều đó.
      Các bản trước LUÔN chọn top engine dù tất cả vô dụng. Bản này có quyền
      nói "không chọn ai cả".

CON SỐ TRUNG THỰC NHẤT:
    Tổng xác suất của 64 con được chọn = TỶ LỆ TRÚNG MÔ HÌNH DỰ BÁO.
    ≈ 64% nghĩa là mô hình tự thừa nhận không có lợi thế.
"""
import os, math, smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import numpy as np
from scipy import stats
from scipy.special import gammaln

import engine as E

VN = timezone(timedelta(hours=7))

# ==============================================================================
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  BẢNG ĐIỀU KHIỂN                                                     │
#  └──────────────────────────────────────────────────────────────────────┘
# ==============================================================================

SO_KY        = 150    # số kỳ lịch sử
SO_CON       = 64     # 64 ô -> mốc 64,00% · hoà vốn 67,37%
CHAY_DB      = True
CHAY_G1      = True
CHAY_G8      = True   # chỉ MN/MT
MIN_TRAIN    = 40     # số kỳ tối thiểu trước khi chấm walk-forward
MUC_Y_NGHIA  = 0.05   # ngưỡng Diebold-Mariano
TY_LE_TRA    = 95.0

EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")

K = 100
SO = np.arange(K)
A_, B_ = SO // 10, SO % 10
DEU = np.full(K, 1.0 / K)
MO_TA = {"DB": ("① ĐỀ ĐẶC BIỆT", "2 chữ số CUỐI giải ĐẶC BIỆT"),
         "G1": ("② ĐỀ GIẢI NHẤT", "2 chữ số CUỐI giải NHẤT"),
         "G8": ("③ ĐỀ ĐẦU (G8)", "giải TÁM — đã là số 2 chữ số")}


def _dem(seq, k=K):
    c = np.zeros(k)
    for v in seq:
        c[v] += 1
    return c


def _chuan(p):
    p = np.clip(np.asarray(p, float), 1e-12, None)
    return p / p.sum()


# ==============================================================================
#  E1 — DIRICHLET-MULTINOMIAL, α bằng MARGINAL LIKELIHOOD
# ==============================================================================

def E1_Dirichlet(mt, pool):
    """log p(dữ liệu | α) = lnΓ(Kα) − lnΓ(N+Kα) + Σ[lnΓ(n_v+α) − lnΓ(α)]
       α lớn -> dữ liệu gần ĐỀU, engine tự làm phẳng."""
    c = _dem(mt); N = c.sum()
    best_a, best_ml = 1.0, -np.inf
    for a in np.exp(np.linspace(np.log(0.05), np.log(500), 40)):
        ml = (gammaln(K * a) - gammaln(N + K * a)
              + np.sum(gammaln(c + a) - gammaln(a)))
        if ml > best_ml:
            best_a, best_ml = a, ml
    p = (c + best_a) / (N + K * best_a)
    return _chuan(p), {"alpha": float(best_a),
                       "y_nghia": "gần đều" if best_a > 20 else "có cấu trúc"}


# ==============================================================================
#  E2 — KIỂM ĐỊNH ĐỘC LẬP HAI CHỮ SỐ
# ==============================================================================

def E2_DocLap(mt, pool):
    """Chi-square độc lập trên bảng 10×10.
       Độc lập -> P(a)·P(b), 20 tham số (chính xác gấp 5 lần).
       Phụ thuộc -> phân phối đồng thời."""
    T = np.zeros((10, 10))
    for v in mt:
        T[v // 10, v % 10] += 1
    try:
        _, p_dl, _, _ = stats.chi2_contingency(T + 0.5)
    except Exception:
        p_dl = 1.0
    c1 = np.bincount([v // 10 for v in mt], minlength=10) + 1.0
    c2 = np.bincount([v % 10 for v in mt], minlength=10) + 1.0
    p1, p2 = c1 / c1.sum(), c2 / c2.sum()
    if p_dl >= MUC_Y_NGHIA:
        p = p1[A_] * p2[B_]
        mh = "độc lập (20 tham số)"
    else:
        p = (T.flatten() + 0.5) / (T.sum() + 50)
        mh = "đồng thời (100 tham số)"
    return _chuan(p), {"p_doc_lap": float(p_dl), "mo_hinh": mh}


# ==============================================================================
#  E3 — CO NGÓT JAMES-STEIN
# ==============================================================================

def E3_CoNgot(mt, pool):
    """p = u + B·(p̂ − u). B tự tính từ dữ liệu.
       B ≈ 0 -> MỌI CHÊNH LỆCH CHỈ LÀ NHIỄU ĐẾM, engine từ chối đưa ý kiến."""
    c = _dem(mt); N = max(c.sum(), 1)
    ph = c / N
    S = float(((ph - DEU) ** 2).sum())
    sig2 = float(DEU[0] * (1 - DEU[0]) / N)
    B = 0.0 if S <= 1e-12 else float(np.clip(1 - (K - 3) * sig2 / S, 0, 1))
    return _chuan(DEU + B * (ph - DEU)), {
        "B": B, "y_nghia": "mọi chênh lệch là NHIỄU" if B < 0.05 else "có tín hiệu"}


# ==============================================================================
#  E4 — GỘP GIẢI CÓ KIỂM ĐỊNH ĐỒNG NHẤT
# ==============================================================================

def E4_GopGiai(mt, pool):
    """Kiểm định giải mục tiêu có CÙNG phân phối chữ số với 17 giải kia.
       Đồng nhất -> mượn dữ liệu cả 18 giải (gấp 18 lần).
       Kiểm định ở mức CHỮ SỐ (2×10) vì bảng 2×100 quá thưa."""
    khac = [v for ky in pool for v in ky]
    p_min = 1.0
    for pos in (0, 1):
        a = np.bincount([(v // 10 if pos == 0 else v % 10) for v in mt], minlength=10)
        b = np.bincount([(v // 10 if pos == 0 else v % 10) for v in khac], minlength=10)
        try:
            _, p, _, _ = stats.chi2_contingency(np.vstack([a, b]) + 0.5)
        except Exception:
            p = 1.0
        p_min = min(p_min, p)
    p_dn = min(1.0, p_min * 2)        # Bonferroni cho 2 vị trí
    if p_dn >= MUC_Y_NGHIA:
        c = _dem(mt) + _dem(khac)
        nguon = f"GỘP 18 giải ({int(c.sum()):,} quan sát)"
    else:
        c = _dem(mt)
        nguon = f"chỉ giải mục tiêu ({int(c.sum())} quan sát)"
    return _chuan(c + 1.0), {"p_dong_nhat": float(p_dn), "nguon": nguon}


# ==============================================================================
#  E5 — PHÁT HIỆN ĐIỂM GÃY
# ==============================================================================

def _ll_dm(seq):
    """Log-likelihood đa thức theo chữ số hàng chục + đơn vị."""
    if not seq:
        return 0.0
    ll = 0.0
    for pos in (0, 1):
        d = np.bincount([(v // 10 if pos == 0 else v % 10) for v in seq], minlength=10)
        p = d / d.sum()
        ll += float(np.sum(d[d > 0] * np.log(p[d > 0])))
    return ll


def E5_DiemGay(mt, pool):
    """Tỷ số hợp lý: 2 đoạn vs 1 đoạn. Ngưỡng = χ²(0,95; 18) + ln(n) để bù
       việc dò nhiều vị trí gãy. Có gãy -> chỉ dùng dữ liệu SAU điểm gãy."""
    n = len(mt)
    ll0 = _ll_dm(mt)
    best_t, best_lr = None, 0.0
    for t in range(15, n - 15, 5):
        lr = 2 * (_ll_dm(mt[:t]) + _ll_dm(mt[t:]) - ll0)
        if lr > best_lr:
            best_lr, best_t = lr, t
    nguong = stats.chi2.ppf(0.95, 18) + math.log(max(n, 2))
    if best_t is not None and best_lr > nguong:
        seq = mt[best_t:]
        kq = f"GÃY tại kỳ {best_t}/{n}, dùng {len(seq)} kỳ sau"
    else:
        seq = mt
        kq = "không phát hiện"
    c1 = np.bincount([v // 10 for v in seq], minlength=10) + 1.0
    c2 = np.bincount([v % 10 for v in seq], minlength=10) + 1.0
    return _chuan((c1 / c1.sum())[A_] * (c2 / c2.sum())[B_]), {
        "diem_gay": best_t if best_lr > nguong else None,
        "lr": float(best_lr), "nguong": float(nguong), "ket_luan": kq}


# ==============================================================================
#  E6 — CHỌN MÔ HÌNH BẰNG BIC
# ==============================================================================

def E6_BIC(mt, pool):
    """BIC = −2·logL + k·ln(N). Phạt mỗi tham số thêm vào.
       Nếu BIC chọn ĐỀU -> dữ liệu KHÔNG ĐỦ để biện minh cho cấu trúc nào."""
    N = len(mt); c = _dem(mt)
    ung_vien = {}
    ung_vien["ĐỀU"] = (DEU, 0)
    c1 = np.bincount([v // 10 for v in mt], minlength=10) + 0.5
    c2 = np.bincount([v % 10 for v in mt], minlength=10) + 0.5
    ung_vien["ĐỘC LẬP"] = (_chuan((c1 / c1.sum())[A_] * (c2 / c2.sum())[B_]), 18)
    ung_vien["ĐỒNG THỜI"] = (_chuan(c + 0.5), 99)
    w = np.array([0.5 ** ((N - 1 - t) / 30) for t in range(N)])
    cw1 = np.zeros(10); cw2 = np.zeros(10)
    for t, v in enumerate(mt):
        cw1[v // 10] += w[t]; cw2[v % 10] += w[t]
    cw1 += 0.5; cw2 += 0.5
    ung_vien["SUY GIẢM"] = (_chuan((cw1 / cw1.sum())[A_] * (cw2 / cw2.sum())[B_]), 19)

    bic = {}
    for ten, (p, k) in ung_vien.items():
        ll = float(np.sum(c * np.log(np.clip(p, 1e-12, None))))
        bic[ten] = -2 * ll + k * math.log(max(N, 2))
    chon = min(bic, key=bic.get)
    return ung_vien[chon][0], {"chon": chon,
                               "bic": {t: round(v, 1) for t, v in bic.items()}}


ENGINES = [("E1_Dirichlet", E1_Dirichlet), ("E2_DocLap", E2_DocLap),
           ("E3_CoNgot", E3_CoNgot), ("E4_GopGiai", E4_GopGiai),
           ("E5_DiemGay", E5_DiemGay), ("E6_BIC", E6_BIC)]
TEN_E = [n for n, _ in ENGINES]


# ==============================================================================
#  CHỌN ENGINE: WALK-FORWARD LOG-SCORE + DIEBOLD-MARIANO
# ==============================================================================

def danh_gia(mt, pool, min_train=MIN_TRAIN):
    """Walk-forward: tại kỳ t, mỗi engine dự báo P_t(v) CHỈ từ dữ liệu < t,
       rồi chấm log P_t(kết quả thật). Không rò rỉ.

       Diebold-Mariano: d_t = logscore_engine − logscore_đều.
       H0: E[d] ≤ 0. Bác bỏ -> engine VƯỢT đều có ý nghĩa.
    """
    n = len(mt)
    mtr = min(min_train, max(20, n // 3))
    ls = {t: [] for t in TEN_E}
    for t in range(mtr, n):
        for ten, f in ENGINES:
            try:
                p, _ = f(mt[:t], pool[:t])
                ls[ten].append(math.log(max(p[mt[t]], 1e-12)))
            except Exception:
                ls[ten].append(math.log(1.0 / K))
    l_deu = math.log(1.0 / K)
    ra = {}
    for ten in TEN_E:
        d = np.array(ls[ten]) - l_deu
        T = len(d)
        m = float(d.mean())
        s = float(d.std(ddof=1)) if T > 1 else 1.0
        z = m / (s / math.sqrt(T)) if s > 1e-12 else 0.0
        p = float(1 - stats.norm.cdf(z))          # một phía: engine > đều?
        ra[ten] = {"vuot_deu": m, "z": z, "p": p, "T": T,
                   "qua": p < MUC_Y_NGHIA and m > 0}
    return ra


def ket_hop(mt, pool, dg):
    """Gộp P(v) các engine QUA kiểm định, trọng số ∝ độ vượt log-score.
       Không engine nào qua -> dùng E1 và BÁO THẲNG."""
    ds, chan_doan = {}, {}
    for ten, f in ENGINES:
        p, cd = f(mt, pool)
        ds[ten] = p; chan_doan[ten] = cd
    qua = [t for t in TEN_E if dg[t]["qua"]]
    if qua:
        w = np.array([dg[t]["vuot_deu"] for t in qua]); w = w / w.sum()
        P = sum(wi * ds[t] for wi, t in zip(w, qua))
        trong_so = dict(zip(qua, (float(x) for x in w)))
        nguon = "GỘP engine vượt đều"
    else:
        P = ds["E1_Dirichlet"]
        trong_so = {"E1_Dirichlet": 1.0}
        nguon = "KHÔNG engine nào vượt đều — dùng E1 (an toàn nhất)"
    return _chuan(P), trong_so, nguon, chan_doan


def chon_64(P, k=SO_CON):
    idx = np.argsort(-P, kind="stable")[:k]
    return sorted(int(i) for i in idx), float(P[idx].sum())


def wilson(t, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p, den = t / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / den
    return (max(0., c - h), min(1., c + h))


# ==============================================================================
#  CHẠY 1 ĐÀI
# ==============================================================================

def chay_dai(stt, ngay_moc, so_ky=None, k=None):
    so_ky = so_ky or SO_KY; k = k or SO_CON
    ten, ma, mien, nd = E.lay_dai(stt)
    m = E.lay_tu_master(stt, so_ky + 20)
    if m:
        toan_giai, ngay_full, info = m; nguon_dl = "KHO"
    else:
        _, ngay, ngay_full, toan_giai, info = E.lay_du_lieu(stt, min(200, so_ky + 30))
        nguon_dl = "WEB"
    if not toan_giai:
        raise RuntimeError("không bóc được toàn bảng giải")
    tg, ng, that = E.cat_truoc_ngay(toan_giai, ngay_full, ngay_moc, so_ky)
    if len(tg) < 50:
        raise RuntimeError(f"chỉ còn {len(tg)} kỳ, cần >=50")

    vt_g8 = len(tg[0]) - 1 if (len(tg[0]) == 18 and len(tg[0][-1]) == 2) else None
    ds = [("DB", CHAY_DB, 0), ("G1", CHAY_G1, 1)]
    if CHAY_G8 and vt_g8 is not None:
        ds.append(("G8", True, vt_g8))

    mods = []
    for khoa, bat, vi in ds:
        if not bat:
            continue
        mt = [int(ky[vi][-2:]) for ky in tg]
        # pool = 17 giải KHÁC (không gồm giải mục tiêu) để E4 kiểm định đồng nhất
        pool = [[int(s[-2:]) for j, s in enumerate(ky) if j != vi] for ky in tg]
        dg = danh_gia(mt, pool)
        P, trong_so, nguon, cd = ket_hop(mt, pool, dg)
        bo, p_du_bao = chon_64(P, k)
        mods.append({
            "key": khoa, "ten": MO_TA[khoa][0], "mo_ta": MO_TA[khoa][1],
            "so": [f"{v:02d}" for v in bo], "p_du_bao": p_du_bao,
            "trong_so": trong_so, "nguon": nguon, "danh_gia": dg,
            "chan_doan": cd, "moc": k / 100.0, "hoa_von": k / TY_LE_TRA,
        })
    return {"stt": stt, "dai": ten, "mien": mien, "nguon": nguon_dl, "n_ky": len(tg),
            "ngay_ky_truoc": ng[-1].strftime("%d/%m/%Y"),
            "db_ky_truoc": tg[-1][0], "g1_ky_truoc": tg[-1][1],
            "g8_ky_truoc": tg[-1][vt_g8] if vt_g8 is not None else None,
            "modules": mods, "ket_qua_that": that}


def _dong_chan_doan(ten, cd):
    if ten == "E1_Dirichlet":
        return f"α = {cd['alpha']:.1f} ({cd['y_nghia']})"
    if ten == "E2_DocLap":
        return f"p = {cd['p_doc_lap']:.3f} → {cd['mo_hinh']}"
    if ten == "E3_CoNgot":
        return f"B = {cd['B']:.3f} ({cd['y_nghia']})"
    if ten == "E4_GopGiai":
        return f"p = {cd['p_dong_nhat']:.3f} → {cd['nguon']}"
    if ten == "E5_DiemGay":
        return cd["ket_luan"]
    if ten == "E6_BIC":
        return f"chọn mô hình {cd['chon']}"
    return ""


# ==============================================================================
#  EMAIL
# ==============================================================================

def _html(ket, ngay_moc, loi):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    thu = E.THU_VN[ngay_moc.weekday()]
    h = [f'<div style="{css}max-width:760px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">64 con — 6 engine thống kê — '
             f'{thu} {ngay_moc:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài · {SO_KY} kỳ · chọn engine bằng log-score + Diebold-Mariano</p>')
    h.append('<div style="background:#fff8e1;border-left:4px solid #f9a825;'
             'padding:10px 13px;margin:0 0 20px;font-size:13px;line-height:1.6">'
             '<b>Đọc con số "mô hình dự báo":</b> đó là TỔNG XÁC SUẤT của 64 con được '
             'chọn — tức tỷ lệ trúng mà mô hình tự cam kết.<br>'
             '· ≈ 64% → mô hình tự thừa nhận KHÔNG có lợi thế<br>'
             '· ≥ 67,4% → mô hình tuyên bố vượt ngưỡng hoà vốn<br>'
             '<b>Engine chỉ được dùng khi VƯỢT phân phối đều có ý nghĩa thống kê.</b> '
             'Không cái nào vượt thì email nói thẳng điều đó.</div>')

    for r in ket:
        h.append(f'<div style="margin:24px 0 0;padding:9px 12px;background:#263238;'
                 f'color:#fff;border-radius:5px 5px 0 0">'
                 f'<span style="font-size:17px;font-weight:700">{r["dai"].upper()}</span>'
                 f'<span style="font-size:13px;opacity:.85"> &nbsp;|&nbsp; {r["mien"]}'
                 f' &nbsp;|&nbsp; {thu} {ngay_moc:%d.%m.%Y}</span><br>'
                 f'<span style="font-size:12px;opacity:.7">Kỳ gần nhất {r["ngay_ky_truoc"]}'
                 f' · ĐB {r["db_ky_truoc"]} · G1 {r["g1_ky_truoc"]}'
                 + (f' · G8 {r["g8_ky_truoc"]}' if r.get("g8_ky_truoc") else '')
                 + f' · {r["n_ky"]} kỳ</span></div>')
        h.append('<div style="border:1px solid #cfd8dc;border-top:0;'
                 'border-radius:0 0 5px 5px;padding:6px 12px 14px">')
        for m in r["modules"]:
            vuot = m["p_du_bao"] >= m["hoa_von"]
            mau = "#2e7d32" if vuot else "#546e7a"
            h.append(f'<div style="margin:12px 0 0">'
                     f'<div style="font-size:14px;font-weight:600">{m["ten"]}</div>'
                     f'<div style="font-size:12px;color:#607d8b;margin:2px 0 5px">'
                     f'{r["dai"]} · {thu} {ngay_moc:%d.%m.%Y} · {m["mo_ta"]}</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:14px;background:#e8f5e9;border-left:4px solid #2e7d32;'
                     f'padding:10px 12px;word-break:break-all;line-height:1.9">'
                     f'{",".join(m["so"])}</div>')
            h.append(f'<div style="font-size:12px;margin:5px 0 0;color:{mau}">'
                     f'<b>Mô hình dự báo tỷ lệ trúng: {m["p_du_bao"]:.1%}</b> · '
                     f'mốc {m["moc"]:.0%} · hoà vốn {m["hoa_von"]:.1%}</div>')
            h.append(f'<div style="font-size:12px;color:#546e7a;margin:3px 0 0">'
                     f'Nguồn: {m["nguon"]} · '
                     + ", ".join(f'{t} {w:.0%}' for t, w in m["trong_so"].items())
                     + '</div>')
            h.append('<details style="margin:6px 0 0"><summary style="font-size:12px;'
                     'color:#546e7a;cursor:pointer">Chẩn đoán 6 engine</summary>'
                     '<table style="font-size:11px;border-collapse:collapse;margin-top:4px">'
                     '<tr style="background:#eceff1"><th style="padding:3px 7px;text-align:left">Engine</th>'
                     '<th style="padding:3px 7px">Vượt đều</th><th style="padding:3px 7px">p (DM)</th>'
                     '<th style="padding:3px 7px;text-align:left">Tự chẩn đoán</th></tr>')
            for t in TEN_E:
                d = m["danh_gia"][t]
                bg = "#e8f5e9" if d["qua"] else "#fff"
                h.append(f'<tr style="background:{bg}"><td style="padding:3px 7px">{t}</td>'
                         f'<td style="padding:3px 7px;text-align:center">{d["vuot_deu"]:+.4f}</td>'
                         f'<td style="padding:3px 7px;text-align:center">{d["p"]:.3f}</td>'
                         f'<td style="padding:3px 7px">{_dong_chan_doan(t, m["chan_doan"][t])}</td></tr>')
            h.append('</table></details></div>')
        h.append('</div>')

    if loi:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;'
                 'padding:10px 13px;margin:16px 0;font-size:13px"><b>Đài lỗi:</b><ul>')
        for s_, t_, e_ in loi:
            h.append(f'<li>[{s_}] {t_}: {e_}</li>')
        h.append('</ul></div>')
    h.append('<hr style="margin:24px 0 10px;border:0;border-top:1px solid #ddd">'
             '<p style="font-size:12px;color:#888;line-height:1.6">'
             'Mỗi engine có kiểm định gắn sẵn và TỰ LÙI về phân phối đều khi không có '
             'tín hiệu. Nếu E3 báo B≈0 và E6 chọn mô hình ĐỀU, hai engine đang ĐỘC LẬP '
             'nói cùng một điều: dữ liệu không đủ để biện minh cho cấu trúc nào.<br>'
             'EV = 95/100 − 1 = −5,0%, cố định bởi tỷ lệ trả.</p></div>')
    return "".join(h)


def gui_email(ket, ngay_moc, loi):
    mk = E._lay_mat_khau()
    thu = E.THU_VN[ngay_moc.weekday()]
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[64 CON THỐNG KÊ] {thu} {ngay_moc:%d.%m.%Y} — {len(ket)} đài"
    msg["From"] = formataddr(("XSMN Thống Kê", EMAIL_GUI)); msg["To"] = EMAIL_NHAN
    t = [f"64 CON — 6 ENGINE THỐNG KÊ — {thu} {ngay_moc:%d.%m.%Y}", ""]
    for r in ket:
        t.append("=" * 60); t.append(f"{r['dai'].upper()} | {r['mien']}")
        for m in r["modules"]:
            t.append("")
            t.append(f"{m['ten']} — mô hình dự báo {m['p_du_bao']:.1%}")
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

def main(ngay=None, so_ky=None, gui_mail=True):
    ngay_moc = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    so_ky = so_ky or SO_KY
    thu = E.THU_VN[ngay_moc.weekday()]
    print("=" * 80)
    print(f"  64 CON — 6 ENGINE XÁC SUẤT THỐNG KÊ  |  {thu} {ngay_moc:%d.%m.%Y}")
    print(f"  {so_ky} kỳ · log-score walk-forward · Diebold-Mariano α={MUC_Y_NGHIA}")
    print("=" * 80)

    dsach = E.dai_theo_ngay(ngay_moc)
    lich = E.xay_lich()
    print(f"\n  Đài hôm nay: {len(dsach)}")
    ket, loi = [], []
    for s in dsach:
        ten = lich.get(str(s), {}).get("ten", E.lay_dai(s)[0])
        try:
            r = chay_dai(s, ngay_moc, so_ky)
            ket.append(r)
            ct = " | ".join(f"{m['ten'][0]} {m['p_du_bao']:.1%}" for m in r["modules"])
            print(f"     ✓ [{s:>2}] {ten:<20} {r['n_ky']} kỳ ({r['nguon']}) | {ct}")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<20} LỖI: {e}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")

    # ---------- BẢNG TỔNG: engine nào qua kiểm định ----------
    print(f"\n[A] ENGINE NÀO VƯỢT PHÂN PHỐI ĐỀU — gộp mọi đài × giải")
    print("-" * 80)
    dem_qua = {t: 0 for t in TEN_E}; tong = 0
    for r in ket:
        for m in r["modules"]:
            tong += 1
            for t in TEN_E:
                if m["danh_gia"][t]["qua"]:
                    dem_qua[t] += 1
    print(f"  {'Engine':<16}{'Số lần qua':>14}{'Tỷ lệ':>10}")
    print("  " + "-" * 42)
    for t in TEN_E:
        print(f"  {t:<16}{dem_qua[t]:>9}/{tong:<4}{dem_qua[t]/max(tong,1):>10.0%}")
    ky_vong = tong * MUC_Y_NGHIA
    print(f"\n  Kỳ vọng do NGẪU NHIÊN: mỗi engine qua ~{ky_vong:.1f}/{tong} lần "
          f"(α = {MUC_Y_NGHIA})")
    print("  → Engine qua nhiều hơn hẳn mức này mới đáng tin.")

    for r in ket:
        for m in r["modules"]:
            print("\n" + "=" * 80)
            print(f"  {r['dai'].upper()}  —  {m['ten']}")
            print("=" * 80)
            print(f"\n  {','.join(m['so'])}\n")
            print(f"  Mô hình dự báo tỷ lệ trúng: {m['p_du_bao']:.2%}  "
                  f"(mốc {m['moc']:.0%} · hoà vốn {m['hoa_von']:.2%})")
            print(f"  Nguồn: {m['nguon']}")
            print(f"\n  {'Engine':<16}{'Vượt đều':>11}{'p (DM)':>9}   Tự chẩn đoán")
            print("  " + "-" * 76)
            for t in TEN_E:
                d = m["danh_gia"][t]
                dau = " ✓" if d["qua"] else "  "
                print(f"  {t:<16}{d['vuot_deu']:>+11.4f}{d['p']:>9.3f}{dau} "
                      f"{_dong_chan_doan(t, m['chan_doan'][t])}")
    print("\n" + "=" * 80)

    if gui_mail:
        print("\n  Đang gửi email...")
        try:
            gui_email(ket, ngay_moc, loi)
        except Exception as e:
            print(f"  ✗ Không gửi được email: {e}")
    return ket
