"""
ENGINE12 — 12 engine độc lập thi đấu, chọn 4 engine đang thắng để sinh 4 chạm.

KIẾN TRÚC
    100 kỳ mỗi đài
      ├── 90 kỳ xa nhất  : mỗi engine "học" ở đây
      └── 10 kỳ gần nhất : đo engine nào ĐANG thắng
    Chấm điểm = walk-forward trên CẢ 90 kỳ, CỘNG THƯỞNG cho phong độ 10 kỳ gần.
    4 engine hạng cao nhất -> mỗi cái 1 chạm -> 4 chạm.
    Lặp riêng cho ĐB · G1 · G8.

BA ĐIỂM SỬA so với ý tưởng gốc (đo bằng mô phỏng 20.000 lượt):

  ① CHỌN TOP 4/12 CHỈ DỰA VÀO 10 KỲ TẠO ẢO GIÁC +13 ĐIỂM
     Với 12 engine ĐỀU vô dụng (đều đúng 19%), 4 engine "tốt nhất" trong 10 kỳ
     vẫn trúng TB 32%. Sai số ở cửa sổ 10 kỳ là 12,4% — gấp đôi thứ cần đo.
     -> Sửa: chấm walk-forward trên 90 kỳ (sai số 4%), 10 kỳ gần CỘNG THƯỞNG.

  ② 4 ENGINE HAY CHỌN TRÙNG CHẠM
     P(ra đủ 4 chạm khác nhau) chỉ 50,2%. Trung bình ra 3,44 chạm.
     -> Sửa: trùng thì lấy bù từ engine hạng kế tiếp cho tới khi đủ 4.

  ③ KHÔNG THẤY ENGINE NÀO ỔN ĐỊNH
     -> Sửa: in bảng đầy đủ 12 engine (hạng, hit 90 kỳ, hit 10 kỳ, chạm đề xuất).

Sáu engine cuối KHÔNG có cơ chế vật lý — giữ lại để cuộc thi tự phán xử bằng
dữ liệu thay vì loại theo định kiến.
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

SO_KY        = 100    # tổng số kỳ: 90 học + 10 đo phong độ
CUA_SO_GAN   = 10     # số kỳ gần nhất dùng đo phong độ
SO_CHAM      = 4      # số chạm cần ra
CHAY_DB      = True   # ① Đề Đặc Biệt
CHAY_G1      = True   # ② Đề Giải Nhất
CHAY_G8      = True   # ③ Đề Đầu (G8) — chỉ MN/MT, Miền Bắc tự bỏ qua

W_GAN        = 0.6    # trọng số phong độ 10 kỳ (0 = bỏ qua, 1 = ngang 90 kỳ)
HALF_LIFE    = 20     # nửa đời trọng số thời gian trong các engine dùng nó
MIN_TRAIN    = 20     # số kỳ tối thiểu trước khi engine bắt đầu dự báo
TY_LE_TRA    = 95.0

EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")

MO_TA = {"DB": ("① ĐỀ ĐẶC BIỆT", "2 chữ số CUỐI giải ĐẶC BIỆT"),
         "G1": ("② ĐỀ GIẢI NHẤT", "2 chữ số CUỐI giải NHẤT"),
         "G8": ("③ ĐỀ ĐẦU (G8)", "giải TÁM — đã là số 2 chữ số")}


# ==============================================================================
#  12 ENGINE — mỗi engine nhận (chạm quá khứ, ma trận chữ số) -> 1 chữ số
# ==============================================================================

def _dem(cham, w=None):
    d = np.zeros(10); n = len(cham)
    for t, c in enumerate(cham):
        wt = 1.0 if w is None else 0.5 ** ((n - 1 - t) / w)
        for x in set(c):
            d[x] += wt
    return d


def E01_TanSuat(cham, M):
    """Chữ số làm chạm nhiều nhất trong toàn bộ lịch sử."""
    return int(np.argmax(_dem(cham)))


def E02_TanSuatW(cham, M):
    """Như E01 nhưng trọng số giảm dần theo thời gian."""
    return int(np.argmax(_dem(cham, HALF_LIFE)))


def E03_CauCham(cham, M):
    """Vị trí chữ số nào dự báo chạm tốt nhất -> lấy chữ số nó đang chỉ tới."""
    n, V = M.shape
    sc = np.zeros(V)
    for t in range(n - 1):
        sc += np.isin(M[t], list(cham[t + 1]))
    return int(M[-1, int(np.argmax(sc))])


def E04_Markov(cham, M):
    """Chữ số hay làm chạm NGAY SAU khi chạm kỳ trước là x."""
    T = np.ones((10, 10))
    for t in range(len(cham) - 1):
        for a in set(cham[t]):
            for b in set(cham[t + 1]):
                T[a, b] += 1
    v = np.zeros(10)
    for a in set(cham[-1]):
        v += T[a] / T[a].sum()
    return int(np.argmax(v))


def E05_ChuoiNay(cham, M):
    """Chữ số đang có chuỗi làm chạm liên tiếp dài nhất."""
    s = np.zeros(10, dtype=int)
    for d in range(10):
        i = 0
        for t in range(len(cham) - 1, -1, -1):
            if d in cham[t]:
                i += 1
            else:
                break
        s[d] = i
    return int(np.argmax(s)) if s.max() > 0 else int(np.argmax(_dem(cham, HALF_LIFE)))


def E06_DeuDan(cham, M):
    """Chữ số có khoảng cách giữa các lần làm chạm ĐỀU nhất."""
    best, bd = 0, -1.0
    for d in range(10):
        idx = [t for t, c in enumerate(cham) if d in c]
        if len(idx) < 3:
            continue
        g = np.diff(idx)
        sc = 1.0 / (1.0 + g.std() / max(g.mean(), 1e-9))
        if idx[-1] < len(cham) - 3 * max(g.mean(), 1):
            sc *= 0.3                      # đã tắt lâu -> phạt nặng
        if sc > bd:
            best, bd = d, sc
    return int(best)


def E07_Gan(cham, M):
    """Chữ số lâu nhất chưa làm chạm. Phân phối hình học KHÔNG NHỚ -> lý thuyết = 0."""
    g = np.full(10, float(len(cham)))
    for t in range(len(cham) - 1, -1, -1):
        for d in set(cham[t]):
            g[d] = min(g[d], len(cham) - 1 - t)
    return int(np.argmax(g))


def E08_Bet(cham, M):
    """Chạm kỳ trước lặp lại. Không có cơ chế."""
    return int(cham[-1][0])


def E09_BongSo(cham, M):
    """Bóng của chạm kỳ trước: 0<->5, 1<->6... Quy tắc dân gian."""
    return int((cham[-1][0] + 5) % 10)


def E10_TongCham(cham, M):
    """(chạm1 + chạm2) mod 10 của kỳ trước. Không có cơ chế."""
    return int((cham[-1][0] + cham[-1][1]) % 10)


def E11_HieuCham(cham, M):
    """|chạm1 − chạm2| của kỳ trước. Không có cơ chế."""
    return int(abs(cham[-1][0] - cham[-1][1]))


def E12_ChamLanh(cham, M):
    """Chữ số làm chạm ÍT nhất — giả định bù trừ, sai với các kỳ độc lập."""
    return int(np.argmin(_dem(cham, HALF_LIFE)))


ENGINES = [("E01_TanSuat", E01_TanSuat), ("E02_TanSuatW", E02_TanSuatW),
           ("E03_CauCham", E03_CauCham), ("E04_Markov", E04_Markov),
           ("E05_ChuoiNay", E05_ChuoiNay), ("E06_DeuDan", E06_DeuDan),
           ("E07_Gan", E07_Gan), ("E08_Bet", E08_Bet),
           ("E09_BongSo", E09_BongSo), ("E10_TongCham", E10_TongCham),
           ("E11_HieuCham", E11_HieuCham), ("E12_ChamLanh", E12_ChamLanh)]
TEN_ENGINE = [n for n, _ in ENGINES]
CO_CHE = {"E01_TanSuat": "có", "E02_TanSuatW": "có", "E03_CauCham": "k.định",
          "E04_Markov": "k.định", "E05_ChuoiNay": "k.định", "E06_DeuDan": "k.định",
          "E07_Gan": "KHÔNG", "E08_Bet": "KHÔNG", "E09_BongSo": "KHÔNG",
          "E10_TongCham": "KHÔNG", "E11_HieuCham": "KHÔNG", "E12_ChamLanh": "KHÔNG"}


# ==============================================================================
#  CHẤM ĐIỂM 12 ENGINE  (sửa ①)
#
#  Walk-forward trên TOÀN BỘ lịch sử: tại mỗi kỳ t, engine chỉ nhìn dữ liệu < t.
#  Điểm = hit rate 90 kỳ  +  W_GAN × hit rate 10 kỳ gần
#  -> giữ tinh thần "ưu tiên engine đang thắng" nhưng không vứt bỏ 90 kỳ.
# ==============================================================================

def cham_diem_engines(cham, M, cua_so_gan=CUA_SO_GAN, min_train=MIN_TRAIN):
    """Trả list dict: tên, hit toàn kỳ, hit 10 kỳ gần, điểm tổng, chạm đề xuất."""
    n = len(cham)
    mt = min(min_train, max(5, n // 3))
    n_eng = len(ENGINES)
    hit = np.zeros((n_eng, n - mt), dtype=bool)

    for i, t in enumerate(range(mt, n)):
        cx, Mx = cham[:t], M[:t]
        for j, (_, f) in enumerate(ENGINES):
            try:
                d = f(cx, Mx) % 10
            except Exception:
                d = 0
            hit[j, i] = d in cham[t]

    n_test = hit.shape[1]
    g = min(cua_so_gan, n_test)
    ra = []
    for j, (ten, f) in enumerate(ENGINES):
        hr_all = float(hit[j].mean())
        hr_gan = float(hit[j, -g:].mean()) if g else hr_all
        try:
            cham_de_xuat = int(f(cham, M) % 10)
        except Exception:
            cham_de_xuat = 0
        ra.append({"ten": ten, "hr_all": hr_all, "hr_gan": hr_gan,
                   "trung_all": int(hit[j].sum()), "trung_gan": int(hit[j, -g:].sum()),
                   "diem": hr_all + W_GAN * hr_gan, "cham": cham_de_xuat,
                   "co_che": CO_CHE[ten]})
    ra.sort(key=lambda x: -x["diem"])
    return ra, n_test, g, hit


def chon_4_cham(bang, k=SO_CHAM):
    """Lấy chạm từ engine hạng cao. Trùng thì BÙ từ engine kế tiếp  (sửa ②)."""
    cham, nguon, da = [], [], set()
    for e in bang:
        if e["cham"] in da:
            continue
        da.add(e["cham"]); cham.append(e["cham"]); nguon.append(e)
        if len(cham) == k:
            break
    i = 0
    while len(cham) < k:                  # cực hiếm: 12 engine < k chữ số khác nhau
        if i not in da:
            da.add(i); cham.append(i)
            nguon.append({"ten": "(bù)", "hr_all": 0, "hr_gan": 0, "cham": i,
                          "diem": 0, "trung_all": 0, "trung_gan": 0, "co_che": "-"})
        i += 1
    thu_tu = np.argsort(cham)
    return ([cham[i] for i in thu_tu], [nguon[i] for i in thu_tu],
            sum(1 for e in bang[:k] if e["cham"] in cham[:k]))


def null_chon_4(n_test, g, n_lap=2000, seed=2026):
    """Nếu CẢ 12 engine đều vô dụng (19%), 4 engine top đạt bao nhiêu?"""
    rng = np.random.default_rng(seed)
    p = 1 - 0.81
    h = rng.random((n_lap, len(ENGINES), n_test)) < p
    diem = h.mean(axis=2) + W_GAN * h[:, :, -g:].mean(axis=2)
    top = np.argsort(-diem, axis=1)[:, :SO_CHAM]
    ra = np.array([h[i, top[i], -g:].mean() for i in range(n_lap)])
    return ra


def so_cua_cham(bo):
    s = set(bo)
    return [f"{v:02d}" for v in range(100) if (v // 10 in s) or (v % 10 in s)]


# ==============================================================================
#  CHẠY 1 ĐÀI
# ==============================================================================

def _vi_tri_g8(tg):
    if len(tg[0]) != 18 or len(tg[0][-1]) != 2:
        return None
    return len(tg[0]) - 1


def _ma_tran(tg, vt_g8):
    """Nguồn chữ số = ĐB + G1 (+ G8)."""
    idx = [0, 1] + ([vt_g8] if vt_g8 is not None else [])
    return np.array([[int(c) for i in idx for c in ky[i]] for ky in tg], dtype=np.int8)


def chay_dai(stt, ngay_moc, so_ky=None, so_cham=None):
    so_ky = so_ky or SO_KY
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
    if len(tg) < 30:
        raise RuntimeError(f"chỉ còn {len(tg)} kỳ trước {ngay_moc:%d.%m.%Y}, cần >=30")

    vt_g8 = _vi_tri_g8(tg)
    M = _ma_tran(tg, vt_g8)
    ds = [("DB", CHAY_DB, 0), ("G1", CHAY_G1, 1)]
    if CHAY_G8 and vt_g8 is not None:
        ds.append(("G8", True, vt_g8))

    mods = []
    for khoa, bat, vi_tri in ds:
        if not bat:
            continue
        cham = [(int(ky[vi_tri][-2]), int(ky[vi_tri][-1])) for ky in tg]
        bang, n_test, g, hit = cham_diem_engines(cham, M)
        bo, nguon_e, _ = chon_4_cham(bang, so_cham)
        nb = null_chon_4(n_test, g)
        hr4 = float(np.mean([e["hr_gan"] for e in nguon_e if e["ten"] != "(bù)"]))
        p_val = float((nb >= hr4).mean())
        p_truot = ((10 - so_cham) / 10) ** 2

        mods.append({
            "key": khoa, "ten": MO_TA[khoa][0], "mo_ta": MO_TA[khoa][1],
            "cham": bo, "so": so_cua_cham(bo), "n_so": int(100 * (1 - p_truot)),
            "moc": 1 - p_truot, "hoa_von": 100 * (1 - p_truot) / TY_LE_TRA,
            "bang": bang, "nguon_e": nguon_e, "n_test": n_test, "g": g,
            "hr4_gan": hr4, "null_tb": float(nb.mean()), "p_value": p_val,
        })

    return {"stt": stt, "dai": ten, "mien": mien, "nguon": nguon, "n_ky": len(tg),
            "ngay_ky_truoc": ng[-1].strftime("%d/%m/%Y"),
            "db_ky_truoc": tg[-1][0], "g1_ky_truoc": tg[-1][1],
            "g8_ky_truoc": tg[-1][vt_g8] if vt_g8 is not None else None,
            "modules": mods, "ket_qua_that": that}


# ==============================================================================
#  EMAIL  —  có BẢNG 12 ENGINE  (sửa ③)
# ==============================================================================

def _html(ket, ngay_moc, loi):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    thu = E.THU_VN[ngay_moc.weekday()]
    m0 = ket[0]["modules"][0]
    h = [f'<div style="{css}max-width:760px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">{SO_CHAM} chạm — {thu} {ngay_moc:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài · 12 engine thi đấu · {SO_KY} kỳ '
             f'({SO_KY-CUA_SO_GAN} học + {CUA_SO_GAN} đo phong độ) · '
             f'phủ {m0["n_so"]}/100 số</p>')
    h.append(f'<div style="background:#fff8e1;border-left:4px solid #f9a825;'
             f'padding:10px 13px;margin:0 0 22px;font-size:13px;line-height:1.6">'
             f'<b>Đọc theo thứ tự:</b><br>'
             f'1. <b>p-value</b> — nếu CẢ 12 engine đều vô dụng (19%), 4 engine top '
             f'vẫn đạt ~32% do ảo giác chọn lọc. p so 4 engine của bạn với mức ảo '
             f'giác đó. Từ 0,05 trở lên nghĩa là chưa hơn được ảo giác.<br>'
             f'2. <b>Bảng 12 engine</b> — theo dõi engine nào ỔN ĐỊNH qua nhiều ngày, '
             f'không phải engine nào dẫn đầu hôm nay.<br>'
             f'3. Mốc ngẫu nhiên {m0["moc"]:.0%} · hoà vốn {m0["hoa_von"]:.2%}</div>')

    for r in ket:
        h.append(f'<div style="margin:26px 0 0;padding:9px 12px;background:#263238;'
                 f'color:#fff;border-radius:5px 5px 0 0">'
                 f'<span style="font-size:17px;font-weight:700">{r["dai"].upper()}</span>'
                 f'<span style="font-size:13px;opacity:.85"> &nbsp;|&nbsp; {r["mien"]}'
                 f' &nbsp;|&nbsp; {thu} {ngay_moc:%d.%m.%Y}</span><br>'
                 f'<span style="font-size:12px;opacity:.7">Kỳ gần nhất {r["ngay_ky_truoc"]}'
                 f' · ĐB {r["db_ky_truoc"]} · G1 {r["g1_ky_truoc"]}'
                 + (f' · G8 {r["g8_ky_truoc"]}' if r.get("g8_ky_truoc") else '')
                 + f' · {r["n_ky"]} kỳ</span></div>')
        h.append('<div style="border:1px solid #cfd8dc;border-top:0;'
                 'border-radius:0 0 5px 5px;padding:4px 12px 14px">')
        for m in r["modules"]:
            mau = "#2e7d32" if m["p_value"] < .05 else "#c62828"
            h.append(f'<div style="margin:14px 0 0">'
                     f'<div style="font-size:14px;font-weight:600">{m["ten"]}</div>'
                     f'<div style="font-size:12px;color:#607d8b;margin:2px 0 6px">'
                     f'{r["dai"]} · {thu} {ngay_moc:%d.%m.%Y} · {m["mo_ta"]}</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:26px;font-weight:700;letter-spacing:8px;'
                     f'background:#e8f5e9;border-left:4px solid #2e7d32;'
                     f'padding:12px 14px">{" ".join(str(c) for c in m["cham"])}</div>')
            # nguồn từng chạm
            h.append('<div style="font-size:12px;color:#546e7a;margin:6px 0 0">'
                     '<b>Mỗi chạm đến từ engine:</b> '
                     + " · ".join(f'{e["cham"]}←{e["ten"]} ({e["hr_gan"]:.0%} gần)'
                                  for e in m["nguon_e"]) + '</div>')
            h.append(f'<div style="font-size:12px;margin:4px 0 0;color:{mau}">'
                     f'4 engine đạt <b>{m["hr4_gan"]:.0%}</b> trong {m["g"]} kỳ gần · '
                     f'ảo giác chọn lọc {m["null_tb"]:.0%} · <b>p = {m["p_value"]:.3f}</b> — '
                     f'{"VƯỢT mức ảo giác" if m["p_value"] < .05 else "chưa hơn ảo giác"}'
                     f'</div>')
            # BẢNG 12 ENGINE
            h.append('<details style="margin:8px 0 0"><summary style="font-size:12px;'
                     'color:#546e7a;cursor:pointer">Bảng 12 engine</summary>'
                     '<table style="font-size:11px;border-collapse:collapse;margin-top:5px">'
                     '<tr style="background:#eceff1"><th style="padding:3px 7px;text-align:left">Hạng</th>'
                     '<th style="padding:3px 7px;text-align:left">Engine</th>'
                     '<th style="padding:3px 7px">Cơ chế</th>'
                     f'<th style="padding:3px 7px">{m["n_test"]} kỳ</th>'
                     f'<th style="padding:3px 7px">{m["g"]} kỳ gần</th>'
                     '<th style="padding:3px 7px">Chạm</th></tr>')
            for i, e in enumerate(m["bang"], 1):
                bg = "#e8f5e9" if e["cham"] in m["cham"] and i <= 6 else "#fff"
                h.append(f'<tr style="background:{bg}"><td style="padding:3px 7px">{i}</td>'
                         f'<td style="padding:3px 7px">{e["ten"]}</td>'
                         f'<td style="padding:3px 7px;text-align:center;color:#90a4ae">{e["co_che"]}</td>'
                         f'<td style="padding:3px 7px;text-align:center">{e["hr_all"]:.0%}</td>'
                         f'<td style="padding:3px 7px;text-align:center"><b>{e["hr_gan"]:.0%}</b></td>'
                         f'<td style="padding:3px 7px;text-align:center">{e["cham"]}</td></tr>')
            h.append('</table></details>')
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
            h.append(f'<li>[{s_}] {t_}: <span style="font-family:monospace;font-size:11px">{e_}</span></li>')
        h.append('</ul></div>')

    h.append(f'<hr style="margin:26px 0 10px;border:0;border-top:1px solid #ddd">'
             f'<p style="font-size:12px;color:#888;line-height:1.6">'
             f'Sáu engine cuối bảng (Gan, Bệt, Bóng, Tổng, Hiệu, Chạm lạnh) KHÔNG có '
             f'cơ chế vật lý — giữ lại để cuộc thi tự phán xử bằng dữ liệu.<br>'
             f'{SO_CHAM} chạm phủ {m0["n_so"]}/100 số → P(trúng) = {m0["moc"]:.0%} '
             f'trừ khi máy quay lệch thật. Engine dẫn đầu hôm nay KHÔNG có nghĩa '
             f'sẽ dẫn đầu ngày mai — theo dõi qua nhiều ngày mới biết.</p></div>')
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
            t.append(f"(4 engine {m['hr4_gan']:.0%} vs ảo giác {m['null_tb']:.0%}, "
                     f"p {m['p_value']:.3f})")
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
    so_ky = so_ky or SO_KY
    so_cham = so_cham or SO_CHAM
    thu = E.THU_VN[ngay_moc.weekday()]
    p_truot = ((10 - so_cham) / 10) ** 2

    print("=" * 80)
    print(f"  {so_cham} CHẠM — 12 ENGINE THI ĐẤU  |  {thu} {ngay_moc:%d.%m.%Y}")
    print(f"  {so_ky} kỳ ({so_ky-CUA_SO_GAN} học + {CUA_SO_GAN} đo phong độ) | "
          f"trọng số phong độ {W_GAN}")
    print(f"  Phủ {int(100*(1-p_truot))}/100 số | mốc {1-p_truot:.2%} | "
          f"hoà vốn {100*(1-p_truot)/TY_LE_TRA:.2%}")
    print("=" * 80)

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
                            f"(p={m['p_value']:.2f})" for m in r["modules"])
            print(f"     ✓ [{s:>2}] {ten:<20} {r['n_ky']} kỳ ({r['nguon']}) | {ct}")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<20} LỖI: {e}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")

    # --- Bảng tổng hợp engine qua mọi đài ---
    print(f"\n[A] ENGINE NÀO MẠNH — gộp mọi đài × giải hôm nay")
    print("-" * 80)
    gop = {t: {"all": [], "gan": [], "top4": 0} for t in TEN_ENGINE}
    n_bo = 0
    for r in ket:
        for m in r["modules"]:
            n_bo += 1
            top4 = {e["ten"] for e in m["nguon_e"]}
            for e in m["bang"]:
                gop[e["ten"]]["all"].append(e["hr_all"])
                gop[e["ten"]]["gan"].append(e["hr_gan"])
                if e["ten"] in top4:
                    gop[e["ten"]]["top4"] += 1
    print(f"  {'Engine':<15}{'Cơ chế':>9}{'Hit toàn kỳ':>13}{'Hit 10 kỳ gần':>15}"
          f"{'Vào top 4':>12}")
    print("  " + "-" * 76)
    for t in sorted(TEN_ENGINE, key=lambda x: -np.mean(gop[x]["all"])):
        g = gop[t]
        print(f"  {t:<15}{CO_CHE[t]:>9}{np.mean(g['all']):>12.1%}"
              f"{np.mean(g['gan']):>15.1%}{g['top4']:>8}/{n_bo:<3}")
    print(f"\n  Mốc ngẫu nhiên mỗi engine: 19,0%")
    print(f"  Engine dẫn đầu HÔM NAY chưa chắc dẫn đầu ngày mai — theo dõi nhiều ngày.")

    for r in ket:
        for m in r["modules"]:
            print("\n" + "=" * 80)
            print(f"  {r['dai'].upper()}  —  {m['ten']}")
            print(f"  {thu} {ngay_moc:%d.%m.%Y}  |  {r['mien']}  |  {m['mo_ta']}")
            print("=" * 80)
            print(f"\n     CHẠM:   {'   '.join(str(c) for c in m['cham'])}\n")
            print(f"  Mỗi chạm đến từ:")
            for e in m["nguon_e"]:
                print(f"     chạm {e['cham']}  ←  {e['ten']:<15} "
                      f"toàn kỳ {e['hr_all']:.0%} · {m['g']} kỳ gần {e['hr_gan']:.0%}")
            print(f"\n  4 engine đạt {m['hr4_gan']:.0%} trong {m['g']} kỳ gần  |  "
                  f"ảo giác chọn lọc {m['null_tb']:.0%}  |  p = {m['p_value']:.3f}")
            print("  → " + ("VƯỢT mức ảo giác chọn lọc — đáng ghi lại theo dõi."
                            if m["p_value"] < .05 else
                            "CHƯA hơn mức ảo giác. Chọn top 4/12 trên 10 kỳ luôn cho"
                            " con số đẹp kể cả khi 12 engine đều vô dụng."))
            print(f"\n  BẢNG 12 ENGINE")
            print(f"  {'#':>3} {'Engine':<15}{'Cơ chế':>9}{m['n_test']:>7} kỳ"
                  f"{m['g']:>8} kỳ gần{'Chạm':>7}")
            print("  " + "-" * 62)
            for i, e in enumerate(m["bang"], 1):
                dau = " ←" if e["cham"] in m["cham"] and i <= 6 else ""
                print(f"  {i:>3} {e['ten']:<15}{e['co_che']:>9}{e['hr_all']:>9.0%}"
                      f"{e['hr_gan']:>13.0%}{e['cham']:>7}{dau}")
            print(f"\n  {m['n_so']} SỐ QUY RA:")
            print("  " + ",".join(m["so"]))
    print("\n" + "=" * 80)

    if gui_mail:
        print("\n  Đang gửi email...")
        try:
            gui_email(ket, ngay_moc, loi)
        except Exception as e:
            print(f"  ✗ Không gửi được email: {e}")
    return ket
