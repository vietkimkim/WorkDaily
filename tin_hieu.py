"""
TIN_HIEU — engine DUY NHẤT phân biệt TÍN HIỆU với NHIỄU cho ĐB · G1 · G8.

    Ô (đài × giải) nào NHIỄU      -> KHÔNG đề xuất
    Ô nào có TÍN HIỆU ỔN ĐỊNH      -> đề xuất 64 con
    Không ô nào đủ                 -> email ghi "hôm nay không có đề xuất"

TẦNG 1 — PHÁT HIỆN (mỗi sáng, chỉ các đài QUAY HÔM NAY, ~20 ô)
    Chia lịch sử thành HAI ĐOẠN TEST RỜI NHAU:
        Khối A: học [0 : n-100]  -> test [n-100 : n-50]
        Khối B: học [0 : n-50]   -> test [n-50  : n]
    Ô qua khi đủ CẢ BA:
        · Khối A vượt hoà vốn 67,37%
        · Khối B vượt hoà vốn 67,37%
        · Gộp 100 lượt: p < 0,05 / (số ô kiểm hôm nay)   <- Bonferroni
    Cộng 2 phép kiểm dữ liệu: không trùng ngày, không lặp nguyên kỳ.

    Đo trên máy quay CÔNG BẰNG: ~0,3 đề xuất sai MỖI TUẦN.
    Không hiệu chỉnh thì 96,8% số NGÀY có đề xuất sai.

TẦNG 2 — XÁC NHẬN TIẾN CỨU
    Qua tầng 1 CHƯA phải được tin. Ô vào trạng thái "ĐANG THEO DÕI".
    Mọi đề xuất được GHI TRƯỚC giờ quay, commit lên GitHub (dấu thời gian).
    Hôm sau, engine tự chấm đề xuất hôm trước với kết quả thật.
    Đủ N_XAC_NHAN kỳ tiến cứu, tỷ lệ trúng vượt hoà vốn VÀ vượt ngẫu nhiên
    có ý nghĩa -> nâng lên "ĐÃ XÁC NHẬN".

    Lý do cần tầng 2: chạy mỗi ngày, xác suất báo nhầm CỘNG DỒN theo thời gian.
    Chỉ dữ liệu TƯƠNG LAI (chưa tồn tại lúc chọn số) mới loại được điều đó.

CÁCH CHỌN 64 CON: tần suất + làm mượt Laplace (chiến lược thống trị yếu).
"""
import os, sys, json, math, html, smtplib, traceback
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import numpy as np
from scipy import stats

import engine as E

VN = timezone(timedelta(hours=7))

# ==============================================================================
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  BẢNG ĐIỀU KHIỂN                                                     │
#  └──────────────────────────────────────────────────────────────────────┘
# ==============================================================================

SO_KY        = 200    # số kỳ lịch sử mỗi đài (nguồn thường trả tối đa ~200)
MIN_KY       = 180    # ít hơn thì không đủ cho 2 khối test -> bỏ qua ô đó
SO_CON       = 64
TY_LE_TRA    = 95.0
HOA_VON      = SO_CON / TY_LE_TRA          # 67,37%
MOC          = SO_CON / 100.0              # 64,00%
ALPHA        = 0.05
N_XAC_NHAN   = 30     # số kỳ tiến cứu tối thiểu để được "ĐÃ XÁC NHẬN"
CHAY_DB, CHAY_G1, CHAY_G8 = True, True, True

FILE_TT      = "data/tin_hieu_theo_doi.json"
EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")
TEN_GIAI     = {"DB": "Đặc Biệt", "G1": "Giải Nhất", "G8": "Giải 8"}


# ==============================================================================
#  TIỆN ÍCH
# ==============================================================================

def chon_64(ls, k=SO_CON):
    """Tần suất + Laplace. Tie-break ổn định theo thứ tự số, KHÔNG ngẫu nhiên."""
    c = np.ones(100)
    for v in ls:
        c[v] += 1
    return sorted(int(i) for i in np.argsort(-c, kind="stable")[:k])


def p_tren(trung, n, p0=MOC):
    """P(X >= trung) khi X ~ Binomial(n, p0)."""
    return float(1 - stats.binom.cdf(trung - 1, n, p0)) if n > 0 else 1.0


def wilson(t, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p, d = t / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _khoa(stt, giai):
    return f"{stt}|{giai}"


def _vi_tri(tg, giai):
    if giai == "DB":
        return 0
    if giai == "G1":
        return 1
    if giai == "G8" and len(tg[0]) == 18 and len(tg[0][-1]) == 2:
        return len(tg[0]) - 1
    return None


# ==============================================================================
#  TRẠNG THÁI — sổ theo dõi tiến cứu, mang qua từng ngày
# ==============================================================================

def nap_tt():
    if os.path.exists(FILE_TT):
        try:
            return json.load(open(FILE_TT, encoding="utf-8"))
        except Exception:
            pass
    return {"de_xuat": [], "cap_nhat": None}


def luu_tt(tt):
    os.makedirs(os.path.dirname(FILE_TT) or ".", exist_ok=True)
    tt["cap_nhat"] = datetime.now(VN).isoformat(timespec="seconds")
    json.dump(tt, open(FILE_TT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ==============================================================================
#  TẦNG 1 — PHÁT HIỆN
# ==============================================================================

def kiem_du_lieu(tg, ng):
    """Trả (ok, lý do). Bắt 2 nguồn rò rỉ phổ biến nhất."""
    if len(set(str(d) for d in ng)) != len(ng):
        return False, "có ngày bị lưu lặp"
    if any(tg[i] == tg[i - 1] for i in range(1, len(tg))):
        return False, "có kỳ trùng khít kỳ liền trước"
    return True, ""


def kiem_o(mt):
    """Hai khối test rời nhau. Trả dict bằng chứng."""
    n = len(mt)
    boA = set(chon_64(mt[:n - 100]))
    tA = sum(1 for v in mt[n - 100:n - 50] if v in boA)
    boB = set(chon_64(mt[:n - 50]))
    tB = sum(1 for v in mt[n - 50:] if v in boB)
    return {"tA": tA, "tB": tB, "rA": tA / 50, "rB": tB / 50,
            "p": p_tren(tA + tB, 100)}


# ==============================================================================
#  TẦNG 2 — CHẤM ĐỀ XUẤT CŨ + XÉT XÁC NHẬN
# ==============================================================================

def ket_qua_that(stt, giai, ngay_iso):
    """Tìm kết quả thật của (đài, giải) vào ngày cho trước, trong kho dữ liệu."""
    m = E.lay_tu_master(stt, 30)
    if not m:
        return None
    tg, ng, _ = m
    for ky, d in zip(tg, ng):
        if str(d)[:10] == ngay_iso:
            vi = _vi_tri([ky], giai)
            if vi is None:
                return None
            return int(ky[vi][-2:])
    return None


def cham_de_xuat_cu(tt, hom_nay):
    """Đề xuất nào đã qua ngày quay mà chưa chấm -> chấm với kết quả thật."""
    moi_cham = []
    for dx in tt["de_xuat"]:
        if dx.get("ket_qua") is not None:
            continue
        if dx["ngay"] >= hom_nay.isoformat():
            continue                          # chưa tới ngày quay
        so_ve = ket_qua_that(dx["stt"], dx["giai"], dx["ngay"])
        if so_ve is None:
            continue                          # kho chưa có kết quả, để hôm sau chấm
        dx["ket_qua"] = {"so_ve": so_ve, "trung": so_ve in set(dx["so"])}
        moi_cham.append(dx)
    return moi_cham


def thanh_tich(tt):
    """Gom kết quả tiến cứu theo từng ô, xét trạng thái."""
    theo_o = {}
    for dx in tt["de_xuat"]:
        if dx.get("ket_qua") is None:
            continue
        k = _khoa(dx["stt"], dx["giai"])
        o = theo_o.setdefault(k, {"stt": dx["stt"], "dai": dx["dai"],
                                  "giai": dx["giai"], "n": 0, "trung": 0})
        o["n"] += 1
        o["trung"] += int(dx["ket_qua"]["trung"])
    for o in theo_o.values():
        o["ty_le"] = o["trung"] / o["n"]
        o["p"] = p_tren(o["trung"], o["n"])
        o["xac_nhan"] = (o["n"] >= N_XAC_NHAN and o["ty_le"] > HOA_VON
                         and o["p"] < ALPHA)
    return theo_o


def trang_thai_o(tt_o, k):
    o = tt_o.get(k)
    if o and o["xac_nhan"]:
        return "ĐÃ XÁC NHẬN"
    return "ĐANG THEO DÕI"


# ==============================================================================
#  CHẠY 1 NGÀY
# ==============================================================================

def chay(ngay=None, gui_mail=True):
    hom_nay = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    thu = E.THU_VN[hom_nay.weekday()]
    tt = nap_tt()

    print("=" * 78)
    print(f"  ENGINE TÍN HIỆU  |  {thu} {hom_nay:%d.%m.%Y}")
    print("=" * 78)

    # ---- Tầng 2 trước: chấm đề xuất hôm trước ----
    moi_cham = cham_de_xuat_cu(tt, hom_nay)
    tt_o = thanh_tich(tt)
    print(f"\n[TẦNG 2] Chấm {len(moi_cham)} đề xuất cũ vừa có kết quả")
    for dx in moi_cham:
        kq = dx["ket_qua"]
        print(f"     {dx['ngay']} {dx['dai']:<18}{TEN_GIAI[dx['giai']]:<10} "
              f"ra {kq['so_ve']:02d} → {'TRÚNG' if kq['trung'] else 'TRƯỢT'}")

    # ---- Tầng 1: phát hiện trên các đài quay hôm nay ----
    dsach = E.dai_theo_ngay(hom_nay)
    lich = E.xay_lich()
    o_kiem = []
    loi = []
    for s in dsach:
        ten = lich.get(str(s), {}).get("ten", E.lay_dai(s)[0])
        try:
            m = E.lay_tu_master(s, SO_KY + 20)
            if m:
                tg_all, ng_all, _ = m
            else:
                _, _, ng_all, tg_all, _ = E.lay_du_lieu(s, SO_KY + 20)
            tg, ng, _ = E.cat_truoc_ngay(tg_all, ng_all, hom_nay, SO_KY)
        except Exception as e:
            loi.append((s, ten, f"không lấy được dữ liệu: {e}"))
            continue
        for giai, bat in (("DB", CHAY_DB), ("G1", CHAY_G1), ("G8", CHAY_G8)):
            if not bat or not tg:
                continue
            vi = _vi_tri(tg, giai)
            if vi is None:
                continue
            o_kiem.append({"stt": s, "dai": ten, "giai": giai, "tg": tg, "ng": ng,
                           "mt": [int(ky[vi][-2:]) for ky in tg]})

    n_o = max(len(o_kiem), 1)
    nguong = ALPHA / n_o
    print(f"\n[TẦNG 1] {len(dsach)} đài quay hôm nay · {len(o_kiem)} ô kiểm · "
          f"ngưỡng Bonferroni p < {nguong:.5f}")

    ket = []
    for o in o_kiem:
        r = {"stt": o["stt"], "dai": o["dai"], "giai": o["giai"], "n": len(o["mt"])}
        if len(o["mt"]) < MIN_KY:
            r.update(ket_luan="THIẾU DỮ LIỆU", ly_do=f"chỉ {len(o['mt'])} kỳ, cần {MIN_KY}")
        else:
            ok, ly = kiem_du_lieu(o["tg"], o["ng"])
            if not ok:
                r.update(ket_luan="DỮ LIỆU LỖI", ly_do=ly)
            else:
                bc = kiem_o(o["mt"])
                r.update(bc)
                qua = bc["rA"] > HOA_VON and bc["rB"] > HOA_VON and bc["p"] < nguong
                r["ket_luan"] = "TÍN HIỆU" if qua else "NHIỄU"
                if qua:
                    r["so"] = chon_64(o["mt"])
                    r["trang_thai"] = trang_thai_o(tt_o, _khoa(o["stt"], o["giai"]))
        ket.append(r)
        dau = {"TÍN HIỆU": "✓", "NHIỄU": "·"}.get(r["ket_luan"], "✗")
        chi = (f"A {r['tA']}/50 · B {r['tB']}/50 · p {r['p']:.2e}"
               if "p" in r else r.get("ly_do", ""))
        print(f"     {dau} {r['dai']:<18}{TEN_GIAI[r['giai']]:<10} {r['ket_luan']:<14} {chi}")

    # ---- Ghi đề xuất hôm nay (TRƯỚC giờ quay) ----
    de_xuat = [r for r in ket if r["ket_luan"] == "TÍN HIỆU"]
    da_co = {(d["ngay"], d["stt"], d["giai"]) for d in tt["de_xuat"]}
    for r in de_xuat:
        k = (hom_nay.isoformat(), r["stt"], r["giai"])
        if k not in da_co:
            tt["de_xuat"].append({"ngay": hom_nay.isoformat(), "stt": r["stt"],
                                  "dai": r["dai"], "giai": r["giai"], "so": r["so"],
                                  "bang_chung": {"tA": r["tA"], "tB": r["tB"], "p": r["p"]},
                                  "ket_qua": None})
    luu_tt(tt)
    tt_o = thanh_tich(tt)

    print(f"\n  KẾT QUẢ: {len(de_xuat)} ô có tín hiệu / {len(ket)} ô kiểm")
    if not de_xuat:
        print("  → Hôm nay không đài × giải nào đủ ổn định. Không đề xuất.")

    bao_cao = {"ngay": hom_nay, "thu": thu, "ket": ket, "de_xuat": de_xuat,
               "moi_cham": moi_cham, "tt_o": tt_o, "nguong": nguong, "loi": loi}
    if gui_mail:
        print("\n  Đang gửi email...")
        gui_email(bao_cao)
        print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")
    return bao_cao


# ==============================================================================
#  EMAIL
# ==============================================================================

def _html(bc):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    ng, thu = bc["ngay"], bc["thu"]
    dx, ket, tt_o = bc["de_xuat"], bc["ket"], bc["tt_o"]
    n_nhieu = sum(1 for r in ket if r["ket_luan"] == "NHIỄU")
    n_loi = sum(1 for r in ket if r["ket_luan"] in ("THIẾU DỮ LIỆU", "DỮ LIỆU LỖI"))
    h = [f'<div style="{css}max-width:780px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">Engine tín hiệu — {thu} {ng:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:13px">'
             f'{len(ket)} ô kiểm (đài quay hôm nay × ĐB/G1/G8) · '
             f'ngưỡng Bonferroni p &lt; {bc["nguong"]:.5f}</p>')

    # ---- Tóm tắt ----
    h.append('<table style="font-size:13px;border-collapse:collapse;margin:0 0 18px">'
             f'<tr><td style="padding:5px 12px;background:#e8f5e9"><b>{len(dx)}</b> có tín hiệu</td>'
             f'<td style="padding:5px 12px;background:#eceff1"><b>{n_nhieu}</b> nhiễu</td>'
             f'<td style="padding:5px 12px;background:#fff3e0"><b>{n_loi}</b> thiếu/lỗi dữ liệu</td>'
             f'</tr></table>')

    # ---- Đề xuất ----
    if not dx:
        h.append('<div style="border-left:4px solid #546e7a;background:#fafafa;'
                 'padding:12px 14px;margin:0 0 20px;font-size:14px">'
                 '<b>Hôm nay không đài × giải nào đủ ổn định. Không đề xuất.</b><br>'
                 '<span style="font-size:12px;color:#607d8b">Đây là engine đang làm đúng '
                 'việc: từ chối đề xuất khi chỉ thấy nhiễu.</span></div>')
    for r in dx:
        xn = r["trang_thai"] == "ĐÃ XÁC NHẬN"
        mau = "#2e7d32" if xn else "#ef6c00"
        h.append(f'<div style="margin:0 0 18px;border:1px solid #cfd8dc;border-radius:5px">'
                 f'<div style="padding:8px 12px;background:#263238;color:#fff">'
                 f'<b style="font-size:15px">{r["dai"].upper()} — {TEN_GIAI[r["giai"]]}</b>'
                 f'<span style="float:right;background:{mau};padding:1px 8px;'
                 f'border-radius:3px;font-size:12px">{r["trang_thai"]}</span></div>'
                 f'<div style="padding:10px 12px">'
                 f'<div style="font-size:12px;color:#455a64;margin-bottom:5px">'
                 f'Khối A <b>{r["tA"]}/50</b> ({r["rA"]:.0%}) · Khối B <b>{r["tB"]}/50</b> '
                 f'({r["rB"]:.0%}) · p gộp <b>{r["p"]:.2e}</b></div>'
                 f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                 f'font-size:14px;background:#e8f5e9;border-left:4px solid #2e7d32;'
                 f'padding:10px 12px;word-break:break-all;line-height:1.9">'
                 f'{",".join(f"{v:02d}" for v in r["so"])}</div>')
        if not xn:
            h.append('<div style="font-size:12px;color:#ef6c00;margin-top:6px">'
                     f'⚠ Chưa xác nhận tiến cứu — cần {N_XAC_NHAN} kỳ tương lai. '
                     'Nên cược nhỏ.</div>')
        h.append('</div></div>')

    # ---- Thành tích tiến cứu ----
    h.append('<div style="font-size:15px;font-weight:700;margin:22px 0 6px">'
             'THÀNH TÍCH TIẾN CỨU — đề xuất đã chấm với kết quả thật</div>')
    if not tt_o:
        h.append('<p style="font-size:13px;color:#607d8b">Chưa có đề xuất nào được chấm. '
                 'Kết quả bắt đầu tích luỹ từ ngày mai.</p>')
    else:
        T = sum(o["trung"] for o in tt_o.values()); N = sum(o["n"] for o in tt_o.values())
        lo, hi = wilson(T, N)
        h.append(f'<p style="font-size:13px;margin:0 0 6px"><b>Tổng: {T}/{N} = '
                 f'{T/N:.1%}</b> · KTC 95% [{lo:.1%}, {hi:.1%}] · mốc 64% · '
                 f'hoà vốn {HOA_VON:.1%}</p>')
        h.append('<table style="font-size:12px;border-collapse:collapse">'
                 '<tr style="background:#eceff1"><th style="padding:4px 9px;text-align:left">Đài</th>'
                 '<th style="padding:4px 9px">Giải</th><th style="padding:4px 9px">Đúng/Tổng</th>'
                 '<th style="padding:4px 9px">Tỷ lệ</th><th style="padding:4px 9px">Trạng thái</th></tr>')
        for o in sorted(tt_o.values(), key=lambda x: -x["n"]):
            h.append(f'<tr><td style="padding:4px 9px">{o["dai"]}</td>'
                     f'<td style="padding:4px 9px">{TEN_GIAI[o["giai"]]}</td>'
                     f'<td style="padding:4px 9px;text-align:center">{o["trung"]}/{o["n"]}</td>'
                     f'<td style="padding:4px 9px;text-align:center">{o["ty_le"]:.0%}</td>'
                     f'<td style="padding:4px 9px">'
                     f'{"ĐÃ XÁC NHẬN" if o["xac_nhan"] else "theo dõi " + str(o["n"]) + "/" + str(N_XAC_NHAN)}'
                     f'</td></tr>')
        h.append('</table>')

    # ---- Chi tiết mọi ô ----
    h.append('<details style="margin:18px 0 0"><summary style="font-size:13px;'
             'color:#546e7a;cursor:pointer">Chi tiết mọi ô đã kiểm</summary>'
             '<table style="font-size:11px;border-collapse:collapse;margin-top:5px">')
    for r in ket:
        chi = (f'A {r["tA"]}/50 · B {r["tB"]}/50 · p {r["p"]:.2e}'
               if "p" in r else r.get("ly_do", ""))
        h.append(f'<tr><td style="padding:3px 8px">{r["dai"]}</td>'
                 f'<td style="padding:3px 8px">{TEN_GIAI[r["giai"]]}</td>'
                 f'<td style="padding:3px 8px"><b>{r["ket_luan"]}</b></td>'
                 f'<td style="padding:3px 8px;color:#607d8b">{chi}</td></tr>')
    h.append('</table></details>')

    if bc["loi"]:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;'
                 'padding:10px 13px;margin:16px 0;font-size:13px"><b>Đài lỗi:</b><ul>')
        for s_, t_, e_ in bc["loi"]:
            h.append(f'<li>[{s_}] {t_}: {html.escape(str(e_))}</li>')
        h.append('</ul></div>')

    h.append(f'<hr style="margin:22px 0 10px;border:0;border-top:1px solid #ddd">'
             f'<p style="font-size:12px;color:#888;line-height:1.6">'
             f'Tầng 1 phát hiện: 2 khối test rời nhau, cả hai vượt hoà vốn, gộp p qua '
             f'ngưỡng Bonferroni. Máy công bằng cho ~0,3 đề xuất sai mỗi tuần.<br>'
             f'Tầng 2 xác nhận: đề xuất ghi TRƯỚC giờ quay, chấm với kết quả thật. '
             f'Cần {N_XAC_NHAN} kỳ tiến cứu, vượt hoà vốn và vượt ngẫu nhiên có ý nghĩa.'
             f'</p></div>')
    return "".join(h)


def _text(bc):
    t = [f"ENGINE TÍN HIỆU — {bc['thu']} {bc['ngay']:%d.%m.%Y}", ""]
    if not bc["de_xuat"]:
        t.append("Hôm nay không đài × giải nào đủ ổn định. Không đề xuất.")
    for r in bc["de_xuat"]:
        t.append(f"{r['dai'].upper()} — {TEN_GIAI[r['giai']]} [{r['trang_thai']}]")
        t.append(f"  A {r['tA']}/50 · B {r['tB']}/50 · p {r['p']:.2e}")
        t.append("  " + ",".join(f"{v:02d}" for v in r["so"]))
        t.append("")
    return "\n".join(t)


def gui_email(bc):
    mk = E._lay_mat_khau()
    n = len(bc["de_xuat"])
    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"[TÍN HIỆU] {bc['thu']} {bc['ngay']:%d.%m.%Y} — "
                      + (f"{n} đề xuất" if n else "không có đề xuất"))
    msg["From"] = formataddr(("XSMN Tín Hiệu", EMAIL_GUI))
    msg["To"] = EMAIL_NHAN
    msg.attach(MIMEText(_text(bc), "plain", "utf-8"))
    msg.attach(MIMEText(_html(bc), "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk)
        sv.send_message(msg)


if __name__ == "__main__":
    ngay = next((a for a in sys.argv[1:] if "." in a), None)
    try:
        E._lay_mat_khau()
        print("  Mật khẩu ứng dụng: OK")
        kho = E.doc_master()
        if kho is None:
            print("  Chưa có kho — quét lần đầu..."); E.tao_master(so_ky=SO_KY)
        else:
            print(f"  Kho: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
            E.cap_nhat_master(so_ky_moi=20)
        chay(ngay)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
