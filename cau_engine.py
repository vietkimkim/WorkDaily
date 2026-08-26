"""
CẦU ENGINE — dự báo 100% bằng phương pháp SOI CẦU VỊ TRÍ.

Độc lập hoàn toàn với engine.py. Chỉ mượn phần hạ tầng: danh sách đài, tải dữ
liệu, lịch quay, gửi email.

CẦU VỊ TRÍ:
    Chọn 2 vị trí chữ số CỐ ĐỊNH trong bảng kết quả kỳ trước, ghép thành số
    2 chữ số, đánh số đó ở kỳ sau.
    Miền Bắc  : 107 vị trí -> 11.342 cầu
    Miền Nam  :  82 vị trí ->  6.642 cầu

CÁCH CHẤM ĐIỂM CẦU (theo yêu cầu: nhiều lần trúng + chuỗi duy trì dài):
    điểm = W_SO_LAN × z(số lần trúng)
         + W_CHUOI_MAX × z(chuỗi dài nhất)
         + W_CHUOI_NAY × z(chuỗi đang chạy tính đến kỳ mới nhất)

CÁCH RA SỐ:
    Xếp cầu theo điểm giảm dần. Lấy dự báo của từng cầu (áp lên kỳ mới nhất),
    gom số duy nhất cho tới khi đủ N con. Cầu điểm cao được ưu tiên trước.

CẢNH BÁO TRUNG THỰC — in kèm mọi báo cáo:
    Với 50 kỳ, mỗi cầu chỉ được kiểm chứng ~39 lần. Xác suất trúng của đề là 1%,
    nên số lần trúng kỳ vọng của MỘT cầu chỉ là 0,39. Khi dò 11.342 cầu, việc
    tìm được vài cầu trúng 3-4 lần là điều CHẮC CHẮN xảy ra kể cả với dữ liệu
    hoàn toàn ngẫu nhiên. Con số "kỳ vọng do ngẫu nhiên" được in ra để đối chiếu.
"""
import os, json, smtplib
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

SO_KY_CAU    = 75     # số kỳ dò cầu. 75 = cân đối nhất với 64 con:
                      #   50 kỳ -> cầu yếu nhất chỉ trúng 2 lần (717 cầu đạt mức này
                      #            do ngẫu nhiên) — quá loãng
                      #   75 kỳ -> cầu yếu nhất vẫn trúng 4 lần, mà mới 1,4 năm
                      #   150+  -> phủ gần 3 năm, rủi ro máy đã thay bi/bảo trì
SO_CON_DB    = 64     # ① ĐỀ ĐẶC BIỆT  : bao nhiêu con?  (0 = tắt)
SO_CON_G1    = 64     # ② ĐỀ GIẢI NHẤT : bao nhiêu con?  (0 = tắt)
MIN_TRAIN    = 10     # số kỳ tối thiểu trước khi bắt đầu chấm cầu

# Trọng số chấm điểm cầu
W_SO_LAN     = 1.0    # số lần trúng trong toàn bộ lịch sử
W_CHUOI_MAX  = 1.0    # chuỗi trúng liên tiếp DÀI NHẤT từng đạt
W_CHUOI_NAY  = 1.5    # chuỗi ĐANG chạy tính đến kỳ mới nhất (ưu tiên cầu còn sống)

TOP_CAU_BAO_CAO = 8   # số cầu hàng đầu in ra trong email

EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")


# ==============================================================================
#  QUÉT CẦU
# ==============================================================================

def _bang_vi_tri(toan_giai):
    """Ma trận (n_ky, n_vi_tri) mọi chữ số + bản đồ vị trí -> (giải, chữ số)."""
    vt = [(i, j) for i, x in enumerate(toan_giai[0]) for j in range(len(x))]
    M = np.zeros((len(toan_giai), len(vt)), dtype=np.int16)
    for t, ky in enumerate(toan_giai):
        for k, (i, j) in enumerate(vt):
            M[t, k] = int(ky[i][j])
    return M, vt


def quet_cau(toan_giai, vi_tri_giai, min_train=MIN_TRAIN):
    """Chấm điểm MỌI cầu cho một giải cụ thể (0 = ĐB, 1 = giải Nhất).

    Trả dict: so_lan, chuoi_max, chuoi_nay, du_bao (số mỗi cầu đề xuất cho kỳ tới),
              n_test, cap (danh sách cặp vị trí), vt (bản đồ vị trí).
    """
    M, vt = _bang_vi_tri(toan_giai)
    n, V = M.shape
    muc_tieu = [int(ky[vi_tri_giai][-2:]) for ky in toan_giai]

    A = np.repeat(np.arange(V), V)
    B = np.tile(np.arange(V), V)
    giu = A != B
    A, B = A[giu], B[giu]

    so_lan = np.zeros(len(A), dtype=np.int32)
    chuoi_max = np.zeros(len(A), dtype=np.int16)
    chuoi_nay = np.zeros(len(A), dtype=np.int16)
    n_test = 0
    for t in range(min_train, n - 1):
        trung = (M[t, A] * 10 + M[t, B]) == muc_tieu[t + 1]
        so_lan += trung
        chuoi_nay = np.where(trung, chuoi_nay + 1, 0)
        chuoi_max = np.maximum(chuoi_max, chuoi_nay)
        n_test += 1

    du_bao = M[n - 1, A] * 10 + M[n - 1, B]      # cầu áp lên kỳ MỚI NHẤT
    return {"so_lan": so_lan, "chuoi_max": chuoi_max, "chuoi_nay": chuoi_nay,
            "du_bao": du_bao, "n_test": n_test, "A": A, "B": B, "vt": vt}


def _z(v):
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-12 else (v - v.mean()) / sd


def chon_so(q, K):
    """Xếp cầu theo điểm, gom số duy nhất cho tới khi đủ K con.

    Trả (danh sách số, danh sách cầu đã dùng, mảng điểm).
    """
    diem = (W_SO_LAN * _z(q["so_lan"])
            + W_CHUOI_MAX * _z(q["chuoi_max"])
            + W_CHUOI_NAY * _z(q["chuoi_nay"]))
    thu_tu = np.argsort(-diem, kind="stable")
    so, dung = [], []
    da_co = set()
    for j in thu_tu:
        v = int(q["du_bao"][j])
        if v in da_co:
            continue
        da_co.add(v); so.append(v); dung.append(j)
        if len(so) == K:
            break
    return sorted(f"{v:02d}" for v in so), dung, diem


def mo_ta_cau(q, j, toan_giai):
    a, b = int(q["A"][j]), int(q["B"][j])
    ga, ja = q["vt"][a]; gb, jb = q["vt"][b]
    return (f"giải#{ga}[{ja+1}]×giải#{gb}[{jb+1}] "
            f"({toan_giai[-1][ga]}→{toan_giai[-1][ga][ja]}, "
            f"{toan_giai[-1][gb]}→{toan_giai[-1][gb][jb]}) = {int(q['du_bao'][j]):02d}")


def chan_doan(q):
    """So số cầu đạt mốc với KỲ VỌNG do ngẫu nhiên. p = 1% cho đề."""
    n_cau, n_test = len(q["so_lan"]), q["n_test"]
    ra = []
    for k in range(2, 7):
        thuc = int((q["so_lan"] >= k).sum())
        kv = n_cau * (1 - stats.binom.cdf(k - 1, n_test, 0.01))
        ra.append((k, thuc, kv))
    return ra


# ==============================================================================
#  CHẠY 1 ĐÀI
# ==============================================================================

MO_TA = {0: ("① ĐỀ ĐẶC BIỆT", "2 chữ số CUỐI của giải ĐẶC BIỆT"),
         1: ("② ĐỀ GIẢI NHẤT", "2 chữ số CUỐI của giải NHẤT")}


def chay_dai(stt, ngay_moc, so_ky=SO_KY_CAU, Ks=None):
    Ks = Ks or {0: SO_CON_DB, 1: SO_CON_G1}
    ten, ma, mien, nd = E.lay_dai(stt)

    m = E.lay_tu_master(stt, so_ky + 20)
    if m:
        toan_giai, ngay_full, info = m
        nguon = "KHO"
    else:
        _, ngay, ngay_full, toan_giai, info = E.lay_du_lieu(stt, min(200, so_ky + 30))
        nguon = "WEB"
    if not toan_giai:
        raise RuntimeError("không bóc được toàn bảng giải")

    tg, ng, that = E.cat_truoc_ngay(toan_giai, ngay_full, ngay_moc, so_ky)
    if len(tg) < 25:
        raise RuntimeError(f"chỉ còn {len(tg)} kỳ trước {ngay_moc:%d.%m.%Y}, cần >=25")

    mods = []
    for vi_tri, K in Ks.items():
        if not K:
            continue
        q = quet_cau(tg, vi_tri)
        so, dung, diem = chon_so(q, K)
        top = [{"mo_ta": mo_ta_cau(q, j, tg), "so_lan": int(q["so_lan"][j]),
                "chuoi_max": int(q["chuoi_max"][j]), "chuoi_nay": int(q["chuoi_nay"][j]),
                "diem": float(diem[j])}
               for j in dung[:TOP_CAU_BAO_CAO]]
        mods.append({"vi_tri": vi_tri, "ten": MO_TA[vi_tri][0], "mo_ta": MO_TA[vi_tri][1],
                     "K": K, "so": so, "top_cau": top, "n_cau": len(q["so_lan"]),
                     "n_test": q["n_test"], "chan_doan": chan_doan(q),
                     "max_so_lan": int(q["so_lan"].max()),
                     "max_chuoi": int(q["chuoi_max"].max()),
                     "dang_chay": int(q["chuoi_nay"].max())})

    return {"stt": stt, "dai": ten, "mien": mien, "nguon": nguon, "n_ky": len(tg),
            "ngay_ky_truoc": ng[-1].strftime("%d/%m/%Y"),
            "db_ky_truoc": tg[-1][0], "g1_ky_truoc": tg[-1][1],
            "modules": mods, "ket_qua_that": that}


# ==============================================================================
#  EMAIL
# ==============================================================================

def _html(ket, ngay_moc, loi):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    thu = E.THU_VN[ngay_moc.weekday()]
    h = [f'<div style="{css}max-width:700px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">Soi cầu — {thu} {ngay_moc:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài · dò cầu trên {SO_KY_CAU} kỳ gần nhất</p>')

    n_cau = ket[0]["modules"][0]["n_cau"] if ket and ket[0]["modules"] else 0
    n_test = ket[0]["modules"][0]["n_test"] if ket and ket[0]["modules"] else 0
    h.append(f'<div style="background:#fff8e1;border-left:4px solid #f9a825;'
             f'padding:10px 13px;margin:0 0 22px;font-size:13px;line-height:1.6">'
             f'<b>Đọc kỹ trước khi dùng.</b><br>'
             f'Mỗi đài dò ~{n_cau:,} cầu trên chỉ {n_test} lần kiểm chứng. Xác suất '
             f'trúng của một cầu là 1%, nên số lần trúng kỳ vọng của MỘT cầu chỉ là '
             f'{n_test*0.01:.2f}. Khi dò hàng vạn cầu, tìm được cầu trúng 3–4 lần là '
             f'điều CHẮC CHẮN xảy ra kể cả với dữ liệu ngẫu nhiên.<br>'
             f'Bảng chẩn đoán dưới mỗi đài so số cầu thực tế với KỲ VỌNG do ngẫu nhiên '
             f'— đó là con số đáng đọc nhất.</div>')

    for r in ket:
        h.append(f'<div style="margin:26px 0 0;padding:9px 12px;background:#263238;'
                 f'color:#fff;border-radius:5px 5px 0 0">'
                 f'<span style="font-size:17px;font-weight:700">{r["dai"].upper()}</span>'
                 f'<span style="font-size:13px;opacity:.85"> &nbsp;|&nbsp; {r["mien"]}'
                 f' &nbsp;|&nbsp; {thu} {ngay_moc:%d.%m.%Y}</span><br>'
                 f'<span style="font-size:12px;opacity:.7">Kỳ gần nhất '
                 f'{r["ngay_ky_truoc"]} · ĐB {r["db_ky_truoc"]} · G1 {r["g1_ky_truoc"]}'
                 f' · {r["n_ky"]} kỳ dò cầu</span></div>')
        h.append('<div style="border:1px solid #cfd8dc;border-top:0;'
                 'border-radius:0 0 5px 5px;padding:4px 12px 12px">')
        for m in r["modules"]:
            h.append(f'<div style="margin:12px 0 0">'
                     f'<div style="font-size:14px;font-weight:600">{m["ten"]} '
                     f'&mdash; {m["K"]} con</div>'
                     f'<div style="font-size:12px;color:#607d8b;margin:2px 0 5px">'
                     f'{r["dai"]} · {thu} {ngay_moc:%d.%m.%Y} · {m["mo_ta"]}</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:14px;background:#eceff1;border-left:3px solid #455a64;'
                     f'padding:9px 11px;word-spacing:2px;line-height:1.75;'
                     f'word-break:break-all">{",".join(m["so"])}</div>')
            h.append('<div style="font-size:12px;color:#546e7a;margin:6px 0 0">'
                     '<b>Cầu hàng đầu:</b><ul style="margin:3px 0;padding-left:18px">')
            for c in m["top_cau"][:5]:
                h.append(f'<li>{c["mo_ta"]} — trúng {c["so_lan"]} lần, '
                         f'chuỗi dài nhất {c["chuoi_max"]}, đang chạy {c["chuoi_nay"]}</li>')
            h.append('</ul>')
            h.append('<b>Chẩn đoán:</b> <table style="font-size:12px;border-collapse:collapse">'
                     '<tr><th style="padding:2px 8px;text-align:left">Trúng ≥</th>'
                     '<th style="padding:2px 8px">Số cầu thực tế</th>'
                     '<th style="padding:2px 8px">Kỳ vọng ngẫu nhiên</th></tr>')
            for k, thuc, kv in m["chan_doan"]:
                mau = "#c62828" if thuc > kv * 2 and kv >= 1 else "#546e7a"
                h.append(f'<tr><td style="padding:2px 8px">{k} lần</td>'
                         f'<td style="padding:2px 8px;text-align:center;color:{mau}">{thuc:,}</td>'
                         f'<td style="padding:2px 8px;text-align:center">{kv:,.1f}</td></tr>')
            h.append('</table></div></div>')
        h.append('</div>')

    if loi:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;'
                 'padding:10px 13px;margin:18px 0;font-size:13px"><b>Đài lỗi:</b><ul>')
        for s_, t_, e_ in loi:
            h.append(f'<li>[{s_}] {t_}: <span style="font-family:monospace;font-size:11px">'
                     f'{e_}</span></li>')
        h.append('</ul></div>')

    h.append('<hr style="margin:26px 0 10px;border:0;border-top:1px solid #ddd">'
             '<p style="font-size:12px;color:#888;line-height:1.6">'
             'Bộ số này dựa 100% vào soi cầu vị trí, KHÔNG dùng mô hình thống kê nào khác. '
             'Cột "Kỳ vọng ngẫu nhiên" cho biết bao nhiêu cầu đạt mốc đó chỉ do may rủi — '
             'nếu số thực tế xấp xỉ kỳ vọng, các cầu tìm được không mang thông tin.</p></div>')
    return "".join(h)


def gui_email(ket, ngay_moc, loi):
    mk = E._lay_mat_khau()
    thu = E.THU_VN[ngay_moc.weekday()]
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[SOI CẦU] {thu} {ngay_moc:%d.%m.%Y} — {len(ket)} đài"
    msg["From"] = formataddr(("XSMN Soi Cầu", EMAIL_GUI)); msg["To"] = EMAIL_NHAN
    t = [f"SOI CẦU {thu} {ngay_moc:%d.%m.%Y}", ""]
    for r in ket:
        t.append("=" * 60)
        t.append(f"{r['dai'].upper()} | {r['mien']} | kỳ trước {r['ngay_ky_truoc']}")
        for m in r["modules"]:
            t.append("")
            t.append(f"{m['ten']} — {m['K']} con — {r['dai']} — {thu} {ngay_moc:%d.%m.%Y}")
            t.append(",".join(m["so"]))
        t.append("")
    msg.attach(MIMEText("\n".join(t), "plain", "utf-8"))
    msg.attach(MIMEText(_html(ket, ngay_moc, loi), "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk); sv.send_message(msg)
    print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")


# ==============================================================================
#  PIPELINE CHÍNH
# ==============================================================================

def main(ngay=None, so_ky=None, so_con_db=None, so_con_g1=None, gui_mail=True):
    ngay_moc = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    so_ky = so_ky or SO_KY_CAU
    Ks = {0: so_con_db if so_con_db is not None else SO_CON_DB,
          1: so_con_g1 if so_con_g1 is not None else SO_CON_G1}
    if not any(Ks.values()):
        raise ValueError("Phải bật ít nhất một giải")

    thu = E.THU_VN[ngay_moc.weekday()]
    print("=" * 78)
    print(f"  SOI CẦU  |  {thu} {ngay_moc:%d.%m.%Y} (giờ VN)  |  {so_ky} kỳ dò cầu")
    print(f"  Trọng số: số lần={W_SO_LAN}  chuỗi max={W_CHUOI_MAX}  đang chạy={W_CHUOI_NAY}")
    print("=" * 78)

    dsach = E.dai_theo_ngay(ngay_moc)
    lich = E.xay_lich()
    print(f"\n  Đài quay hôm nay: {len(dsach)}")

    ket, loi = [], []
    for s in dsach:
        ten = lich[str(s)]["ten"]
        try:
            r = chay_dai(s, ngay_moc, so_ky, Ks)
            ket.append(r)
            m0 = r["modules"][0]
            print(f"     ✓ [{s:>2}] {ten:<22} {r['n_ky']} kỳ ({r['nguon']}) | "
                  f"{m0['n_cau']:,} cầu | max {m0['max_so_lan']} lần trúng, "
                  f"chuỗi {m0['max_chuoi']}")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<22} LỖI: {e}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")

    # ---------- Chẩn đoán gộp ----------
    print(f"\n[A] CHẨN ĐOÁN — số cầu đạt mốc so với KỲ VỌNG do ngẫu nhiên")
    print("-" * 78)
    for r in ket[:3]:
        for m in r["modules"]:
            cd = " | ".join(f"≥{k}: {t:,} (kv {kv:.1f})" for k, t, kv in m["chan_doan"][:4])
            print(f"  {r['dai']:<16}{m['ten']:<17}{cd}")
    if len(ket) > 3:
        print(f"  ... và {len(ket)-3} đài nữa (xem đầy đủ trong email)")
    print("\n  Nếu số thực tế xấp xỉ kỳ vọng, các cầu tìm được KHÔNG mang thông tin.")

    # ---------- Các vùng số ----------
    for r in ket:
        for m in r["modules"]:
            print("\n" + "=" * 78)
            print(f"  {r['dai'].upper()}  —  {m['ten']}  ({m['K']} con)")
            print(f"  {thu} {ngay_moc:%d.%m.%Y}  |  {r['mien']}  |  {m['mo_ta']}")
            print(f"  Kỳ trước {r['ngay_ky_truoc']} · ĐB {r['db_ky_truoc']} · "
                  f"G1 {r['g1_ky_truoc']}")
            print("=" * 78)
            for i in range(0, len(m["so"]), 10):
                print("   " + "  ".join(m["so"][i:i+10]))
            print("\n  Cầu hàng đầu đã dùng:")
            for c in m["top_cau"][:5]:
                print(f"     {c['mo_ta']}")
                print(f"        trúng {c['so_lan']} lần | chuỗi dài nhất {c['chuoi_max']}"
                      f" | đang chạy {c['chuoi_nay']}")
    print("\n" + "=" * 78)

    if gui_mail:
        print("\n  Đang gửi email...")
        try:
            gui_email(ket, ngay_moc, loi)
        except Exception as e:
            print(f"  ✗ Không gửi được email: {e}")
    return ket
