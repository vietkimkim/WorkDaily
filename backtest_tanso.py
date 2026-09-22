"""
BACKTEST_TANSO — kiểm chứng cách chọn 64 con bằng CHIẾN LƯỢC THỐNG TRỊ YẾU.

CÁCH CHỌN (không engine, không dự đoán):
    1. Đếm mỗi con 00-99 xuất hiện bao nhiêu lần trong lịch sử
    2. Cộng α = 1 cho mọi con (làm mượt Laplace — con chưa từng ra không bị
       loại hẳn chỉ vì thiếu dữ liệu)
    3. Lấy 64 con có số đếm cao nhất

    LÝ DO chọn cách này (chiến lược thống trị yếu, decision theory):
      · Máy quay CÔNG BẰNG  -> mọi bộ 64 con đều đúng 64%. Không mất gì.
      · Máy quay CÓ LỆCH    -> con ra nhiều nhất nằm ở phía được lệch. Có lợi.
    Không bao giờ TỆ HƠN, đôi khi TỐT HƠN -> lựa chọn hợp lý duy nhất.

THIẾT KẾ BACKTEST:
    150 kỳ mỗi đài · 100 kỳ đầu để CHỌN · 50 kỳ sau để KIỂM CHỨNG
    Không rò rỉ: mọi lựa chọn chỉ dùng dữ liệu TRƯỚC kỳ được kiểm.

BA CÁCH CHẠY SONG SONG — đối chứng là phần quan trọng nhất:
    A. CỐ ĐỊNH    : chọn 64 con từ 100 kỳ đầu, GIỮ NGUYÊN suốt 50 kỳ test
    B. CUỐN CHIẾU : tại mỗi kỳ t, chọn lại từ toàn bộ dữ liệu < t (thực tế hơn)
    C. NGẪU NHIÊN : bốc 64 con bất kỳ — ĐỐI CHỨNG

    Nếu A và B không hơn C, cách chọn theo tần suất không mang lại gì.
    Đó là câu trả lời, và nó đáng tin hơn bất kỳ engine nào.

SỐ LIỆU:
    1 đài × 1 giải × 50 kỳ  -> sai số 6,8%  (quá ít, không kết luận được)
    GỘP mọi đài × giải      -> vài nghìn quan sát, sai số dưới 1%  <- ĐÁNG TIN
"""
import os, sys, math, smtplib
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

SO_KY_TONG  = 150    # tổng số kỳ lấy về
SO_KY_HOC   = 100    # 100 kỳ đầu để chọn
SO_CON      = 64     # số con chọn -> mốc ngẫu nhiên 64,00%
ALPHA       = 1.0    # làm mượt Laplace
TY_LE_TRA   = 95.0   # thưởng mỗi lần trúng
SEED        = 2026

EMAIL_NHAN  = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI   = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")

TEN_GIAI = {"DB": "① Đặc Biệt", "G1": "② Giải Nhất", "G8": "③ Giải 8"}


# ==============================================================================
#  CÁCH CHỌN 64 CON
# ==============================================================================

def chon_64(lich_su, k=SO_CON, alpha=ALPHA):
    """Đếm tần suất + làm mượt Laplace -> lấy k con cao nhất.
       Tie-break ổn định theo thứ tự số, KHÔNG ngẫu nhiên."""
    c = np.full(100, alpha)
    for v in lich_su:
        c[v] += 1
    return set(int(i) for i in np.argsort(-c, kind="stable")[:k])


def backtest_1_giai(mt, so_ky_hoc=SO_KY_HOC, k=SO_CON, rng=None):
    """Trả kết quả 3 cách: cố định · cuốn chiếu · ngẫu nhiên."""
    n = len(mt)
    rng = rng or np.random.default_rng(SEED)
    bo_cd = chon_64(mt[:so_ky_hoc], k)

    kq = {"cd": [], "cc": [], "nn": []}
    chi_tiet = []
    for t in range(so_ky_hoc, n):
        that = mt[t]
        bo_cc = chon_64(mt[:t], k)                     # chọn lại mỗi kỳ
        bo_nn = set(rng.choice(100, k, replace=False).tolist())
        a, b, c = that in bo_cd, that in bo_cc, that in bo_nn
        kq["cd"].append(a); kq["cc"].append(b); kq["nn"].append(c)
        chi_tiet.append({"ky": t + 1, "so": that, "cd": a, "cc": b, "nn": c})
    return {t: np.array(v) for t, v in kq.items()}, bo_cd, chi_tiet


def wilson(t, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p, den = t / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / den
    return (max(0., c - h), min(1., c + h))


def tien(trung, tong, k=SO_CON):
    """Lãi/lỗ: mỗi kỳ đặt k điểm, trúng được TY_LE_TRA."""
    von = tong * k
    thu = trung * TY_LE_TRA
    return von, thu, thu - von


# ==============================================================================
#  CHẠY 1 ĐÀI
# ==============================================================================

def chay_dai(stt, so_ky=SO_KY_TONG, so_ky_hoc=SO_KY_HOC, k=SO_CON):
    ten, ma, mien, nd = E.lay_dai(stt)
    m = E.lay_tu_master(stt, so_ky)
    if m:
        tg, ngay_full, info = m; nguon = "KHO"
    else:
        _, ngay, ngay_full, tg, info = E.lay_du_lieu(stt, so_ky)
        nguon = "WEB"
    if not tg or len(tg) < so_ky_hoc + 20:
        raise RuntimeError(f"chỉ có {len(tg) if tg else 0} kỳ, cần >= {so_ky_hoc+20}")
    tg = tg[-so_ky:]

    vt_g8 = len(tg[0]) - 1 if (len(tg[0]) == 18 and len(tg[0][-1]) == 2) else None
    ds = [("DB", 0), ("G1", 1)] + ([("G8", vt_g8)] if vt_g8 is not None else [])

    rng = np.random.default_rng(SEED + stt)
    mods = []
    for khoa, vi in ds:
        mt = [int(ky[vi][-2:]) for ky in tg]
        kq, bo_cd, ct = backtest_1_giai(mt, so_ky_hoc, k, rng)
        n_test = len(kq["cd"])
        m_ = {"key": khoa, "ten": TEN_GIAI[khoa], "n_test": n_test,
              "bo_co_dinh": sorted(bo_cd), "chi_tiet": ct}
        for cach in ("cd", "cc", "nn"):
            trung = int(kq[cach].sum())
            von, thu, lai = tien(trung, n_test, k)
            lo, hi = wilson(trung, n_test)
            m_[cach] = {"trung": trung, "truot": n_test - trung,
                        "hr": trung / n_test, "ktc": (lo, hi),
                        "von": von, "thu": thu, "lai": lai,
                        "roi": lai / von if von else 0.0}
        mods.append(m_)
    return {"stt": stt, "dai": ten, "mien": mien, "nguon": nguon,
            "n_ky": len(tg), "modules": mods}


# ==============================================================================
#  EMAIL BÁO CÁO CHI TIẾT
# ==============================================================================

def _gop(ket, cach):
    tr = sum(m[cach]["trung"] for r in ket for m in r["modules"])
    n = sum(m["n_test"] for r in ket for m in r["modules"])
    lai = sum(m[cach]["lai"] for r in ket for m in r["modules"])
    von = sum(m[cach]["von"] for r in ket for m in r["modules"])
    return {"trung": tr, "truot": n - tr, "n": n, "hr": tr / max(n, 1),
            "ktc": wilson(tr, n), "lai": lai, "von": von,
            "roi": lai / von if von else 0.0}


def _html(ket, loi):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    g = {c: _gop(ket, c) for c in ("cd", "cc", "nn")}
    hoa_von = SO_CON / TY_LE_TRA
    h = [f'<div style="{css}max-width:820px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">Backtest — chọn {SO_CON} con theo tần suất</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài · học {SO_KY_HOC} kỳ · kiểm chứng '
             f'{SO_KY_TONG-SO_KY_HOC} kỳ · tổng {g["cd"]["n"]:,} lượt</p>')

    # ---- KẾT LUẬN GỘP ----
    h.append('<div style="background:#eceff1;border-left:4px solid #263238;'
             'padding:12px 14px;margin:0 0 20px">'
             '<div style="font-size:15px;font-weight:700;margin-bottom:8px">'
             'KẾT QUẢ GỘP TOÀN BỘ — con số đáng tin nhất</div>'
             '<table style="font-size:13px;border-collapse:collapse;width:100%">'
             '<tr style="background:#cfd8dc"><th style="padding:5px 9px;text-align:left">Cách chọn</th>'
             '<th style="padding:5px 9px">Trúng</th><th style="padding:5px 9px">Trượt</th>'
             '<th style="padding:5px 9px">Tỷ lệ</th><th style="padding:5px 9px">KTC 95%</th>'
             '<th style="padding:5px 9px">Lãi/lỗ</th></tr>')
    ten_c = {"cd": "A. Cố định (100 kỳ đầu)", "cc": "B. Cuốn chiếu (chọn lại mỗi kỳ)",
             "nn": "C. NGẪU NHIÊN — đối chứng"}
    for c in ("cd", "cc", "nn"):
        d = g[c]
        bg = "#ffebee" if c == "nn" else "#fff"
        h.append(f'<tr style="background:{bg}"><td style="padding:5px 9px">{ten_c[c]}</td>'
                 f'<td style="padding:5px 9px;text-align:center">{d["trung"]:,}</td>'
                 f'<td style="padding:5px 9px;text-align:center">{d["truot"]:,}</td>'
                 f'<td style="padding:5px 9px;text-align:center"><b>{d["hr"]:.2%}</b></td>'
                 f'<td style="padding:5px 9px;text-align:center">'
                 f'[{d["ktc"][0]:.1%}, {d["ktc"][1]:.1%}]</td>'
                 f'<td style="padding:5px 9px;text-align:center;color:'
                 f'{"#2e7d32" if d["lai"]>0 else "#c62828"}">{d["roi"]:+.1%}</td></tr>')
    h.append(f'</table>'
             f'<div style="font-size:12px;color:#455a64;margin-top:8px">'
             f'Mốc lý thuyết <b>{SO_CON}%</b> · hoà vốn cần <b>{hoa_von:.2%}</b> '
             f'(tỷ lệ trả {TY_LE_TRA:.0f})<br>'
             f'<b>So với đối chứng ngẫu nhiên:</b> '
             f'cố định {g["cd"]["hr"]-g["nn"]["hr"]:+.2%} · '
             f'cuốn chiếu {g["cc"]["hr"]-g["nn"]["hr"]:+.2%}</div></div>')

    # ---- PHÁN XỬ ----
    tot = max(g["cd"]["hr"], g["cc"]["hr"])
    vuot_nn = tot - g["nn"]["hr"]
    lo_tot = max(g["cd"]["ktc"][0], g["cc"]["ktc"][0])
    if lo_tot > hoa_von:
        pxu, mau = ("VƯỢT ngưỡng hoà vốn có ý nghĩa thống kê — phát hiện đáng "
                    "kiểm chứng lại ngay.", "#2e7d32")
    elif vuot_nn > 0 and g["cd"]["ktc"][0] > g["nn"]["ktc"][1]:
        pxu, mau = ("Vượt đối chứng ngẫu nhiên, nhưng chưa tới ngưỡng hoà vốn.",
                    "#ef6c00")
    else:
        pxu, mau = (f"KHÔNG vượt đối chứng ngẫu nhiên. Chọn theo tần suất cho kết quả "
                    f"ngang bốc bừa. Đúng như dự đoán nếu máy quay công bằng — và đây "
                    f"là bằng chứng ngoài mẫu, không phải suy luận.", "#c62828")
    h.append(f'<div style="border-left:4px solid {mau};background:#fafafa;'
             f'padding:11px 14px;margin:0 0 22px;font-size:13px;line-height:1.6">'
             f'<b style="color:{mau}">PHÁN XỬ:</b> {pxu}</div>')

    # ---- CHI TIẾT TỪNG ĐÀI × GIẢI ----
    h.append('<div style="font-size:15px;font-weight:700;margin:0 0 6px">'
             'CHI TIẾT TỪNG ĐÀI × TỪNG GIẢI</div>')
    h.append('<table style="font-size:12px;border-collapse:collapse;width:100%">'
             '<tr style="background:#263238;color:#fff">'
             '<th style="padding:5px 8px;text-align:left">Đài</th>'
             '<th style="padding:5px 8px;text-align:left">Giải</th>'
             '<th style="padding:5px 8px">Kỳ test</th>'
             '<th style="padding:5px 8px">Cố định<br>đúng/sai</th>'
             '<th style="padding:5px 8px">Tỷ lệ</th>'
             '<th style="padding:5px 8px">Cuốn chiếu<br>đúng/sai</th>'
             '<th style="padding:5px 8px">Tỷ lệ</th>'
             '<th style="padding:5px 8px">Ngẫu nhiên<br>đúng/sai</th>'
             '<th style="padding:5px 8px">Lãi/lỗ CĐ</th></tr>')
    for i, r in enumerate(ket):
        for j, m in enumerate(r["modules"]):
            bg = "#fff" if i % 2 == 0 else "#f5f7f8"
            ml = "#2e7d32" if m["cd"]["lai"] > 0 else "#c62828"
            h.append(f'<tr style="background:{bg}">'
                     f'<td style="padding:4px 8px">{r["dai"] if j==0 else ""}</td>'
                     f'<td style="padding:4px 8px">{m["ten"]}</td>'
                     f'<td style="padding:4px 8px;text-align:center">{m["n_test"]}</td>'
                     f'<td style="padding:4px 8px;text-align:center">'
                     f'<b>{m["cd"]["trung"]}</b>/{m["cd"]["truot"]}</td>'
                     f'<td style="padding:4px 8px;text-align:center">{m["cd"]["hr"]:.0%}</td>'
                     f'<td style="padding:4px 8px;text-align:center">'
                     f'<b>{m["cc"]["trung"]}</b>/{m["cc"]["truot"]}</td>'
                     f'<td style="padding:4px 8px;text-align:center">{m["cc"]["hr"]:.0%}</td>'
                     f'<td style="padding:4px 8px;text-align:center;color:#90a4ae">'
                     f'{m["nn"]["trung"]}/{m["nn"]["truot"]}</td>'
                     f'<td style="padding:4px 8px;text-align:center;color:{ml}">'
                     f'{m["cd"]["roi"]:+.0%}</td></tr>')
    h.append('</table>')

    if loi:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;'
                 'padding:10px 13px;margin:16px 0;font-size:13px"><b>Đài lỗi:</b><ul>')
        for s_, t_, e_ in loi:
            h.append(f'<li>[{s_}] {t_}: {e_}</li>')
        h.append('</ul></div>')

    h.append(f'<hr style="margin:22px 0 10px;border:0;border-top:1px solid #ddd">'
             f'<p style="font-size:12px;color:#888;line-height:1.6">'
             f'<b>Cách chọn:</b> đếm tần suất 00-99 trong lịch sử, làm mượt Laplace '
             f'(α={ALPHA}), lấy {SO_CON} con cao nhất. Không engine, không dự đoán.<br>'
             f'<b>Lý do:</b> chiến lược thống trị yếu — máy công bằng thì không mất gì, '
             f'máy có lệch thì có lợi.<br>'
             f'<b>Sai số:</b> 1 đài × 1 giải × {SO_KY_TONG-SO_KY_HOC} kỳ có sai số '
             f'~6,8% — KHÔNG kết luận được. Chỉ dòng GỘP ở đầu mới đáng tin.</p></div>')
    return "".join(h)


def gui_email(ket, loi):
    mk = E._lay_mat_khau()
    hn = datetime.now(VN).date()
    g = {c: _gop(ket, c) for c in ("cd", "cc", "nn")}
    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"[BACKTEST {SO_CON} CON] {len(ket)} đài — "
                      f"cố định {g['cd']['hr']:.1%} vs ngẫu nhiên {g['nn']['hr']:.1%}")
    msg["From"] = formataddr(("XSMN Backtest", EMAIL_GUI)); msg["To"] = EMAIL_NHAN
    t = [f"BACKTEST {SO_CON} CON — {hn:%d.%m.%Y}", "",
         f"Học {SO_KY_HOC} kỳ · kiểm chứng {SO_KY_TONG-SO_KY_HOC} kỳ · "
         f"{len(ket)} đài · {g['cd']['n']:,} lượt", ""]
    for c, ten in (("cd", "A. Cố định"), ("cc", "B. Cuốn chiếu"),
                   ("nn", "C. Ngẫu nhiên (đối chứng)")):
        d = g[c]
        t.append(f"{ten:<28} {d['trung']:>5} đúng / {d['truot']:>5} sai  "
                 f"= {d['hr']:.2%}  (lãi {d['roi']:+.1%})")
    t.append("")
    t.append(f"Mốc lý thuyết {SO_CON}% · hoà vốn {SO_CON/TY_LE_TRA:.2%}")
    t.append("")
    for r in ket:
        t.append(f"{r['dai'].upper()}")
        for m in r["modules"]:
            t.append(f"   {m['ten']:<14} CĐ {m['cd']['trung']:>2}/{m['cd']['truot']:<2} "
                     f"({m['cd']['hr']:.0%})  CC {m['cc']['trung']:>2}/{m['cc']['truot']:<2} "
                     f"({m['cc']['hr']:.0%})  NN {m['nn']['trung']:>2}/{m['nn']['truot']:<2}")
    msg.attach(MIMEText("\n".join(t), "plain", "utf-8"))
    msg.attach(MIMEText(_html(ket, loi), "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk); sv.send_message(msg)
    print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")


# ==============================================================================
#  PIPELINE
# ==============================================================================

def main(gui_mail=True, so_ky=SO_KY_TONG, so_ky_hoc=SO_KY_HOC, k=SO_CON):
    print("=" * 80)
    print(f"  BACKTEST — chọn {k} con theo TẦN SUẤT (chiến lược thống trị yếu)")
    print(f"  Học {so_ky_hoc} kỳ · kiểm chứng {so_ky-so_ky_hoc} kỳ · TẤT CẢ các đài")
    print("=" * 80)

    lich = E.xay_lich()
    dsach = sorted(int(s) for s in lich)
    print(f"\n  Tổng số đài: {len(dsach)}\n")

    ket, loi = [], []
    for s in dsach:
        ten = lich.get(str(s), {}).get("ten", E.lay_dai(s)[0])
        try:
            r = chay_dai(s, so_ky, so_ky_hoc, k)
            ket.append(r)
            ct = " ".join(f"{m['ten'][0]}{m['cd']['trung']}/{m['n_test']}"
                          for m in r["modules"])
            print(f"     ✓ [{s:>2}] {ten:<20} {r['n_ky']} kỳ ({r['nguon']}) | {ct}")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<20} {e}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")

    g = {c: _gop(ket, c) for c in ("cd", "cc", "nn")}
    hoa_von = k / TY_LE_TRA
    print(f"\n{'='*80}")
    print(f"  KẾT QUẢ GỘP — {g['cd']['n']:,} lượt từ {len(ket)} đài")
    print("=" * 80)
    print(f"  {'Cách chọn':<34}{'Đúng':>7}{'Sai':>7}{'Tỷ lệ':>9}{'KTC 95%':>18}{'Lãi':>9}")
    print("  " + "-" * 78)
    for c, ten in (("cd", "A. Cố định (100 kỳ đầu)"),
                   ("cc", "B. Cuốn chiếu (chọn lại mỗi kỳ)"),
                   ("nn", "C. NGẪU NHIÊN — đối chứng")):
        d = g[c]
        print(f"  {ten:<34}{d['trung']:>7,}{d['truot']:>7,}{d['hr']:>9.2%}"
              f"   [{d['ktc'][0]:>5.1%},{d['ktc'][1]:>5.1%}]{d['roi']:>9.1%}")
    print("  " + "-" * 78)
    print(f"  Mốc lý thuyết {k}%  ·  hoà vốn cần {hoa_von:.2%}")
    print(f"  Vượt đối chứng: cố định {g['cd']['hr']-g['nn']['hr']:+.2%} · "
          f"cuốn chiếu {g['cc']['hr']-g['nn']['hr']:+.2%}")
    lo_tot = max(g["cd"]["ktc"][0], g["cc"]["ktc"][0])
    print(f"\n  PHÁN XỬ")
    print("  " + "-" * 78)
    if lo_tot > hoa_von:
        print(f"  → VƯỢT ngưỡng hoà vốn có ý nghĩa. Kiểm chứng lại ngay.")
    elif g["cd"]["ktc"][0] > g["nn"]["ktc"][1]:
        print(f"  → Vượt đối chứng ngẫu nhiên nhưng chưa tới ngưỡng hoà vốn.")
    else:
        print(f"  → KHÔNG vượt đối chứng ngẫu nhiên.")
        print(f"    Chọn theo tần suất cho kết quả NGANG bốc bừa — đúng như dự đoán")
        print(f"    nếu máy quay công bằng. Đây là bằng chứng NGOÀI MẪU thật sự.")
    print(f"\n  ⚠ 1 đài × 1 giải × {so_ky-so_ky_hoc} kỳ có sai số ~6,8% — không kết luận")
    print(f"    được. Chỉ dòng GỘP ở trên mới đáng tin.")

    if gui_mail:
        print("\n  Đang gửi email...")
        try:
            gui_email(ket, loi)
        except Exception as e:
            print(f"  ✗ Không gửi được email: {e}")
    return ket


if __name__ == "__main__":
    import traceback
    try:
        E._lay_mat_khau(); print("  Mật khẩu ứng dụng: OK\n")
        kho = E.doc_master()
        if kho is None:
            print("  Chưa có kho — quét lần đầu (~4 phút)..."); E.tao_master(so_ky=200)
        else:
            print(f"  Kho: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
            E.cap_nhat_master(so_ky_moi=20)
        print()
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
