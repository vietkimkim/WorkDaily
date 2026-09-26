"""
DO_NI_20 — ĐO NI 130 + 20 KỲ, PHƯƠNG ÁN A.
            Chỉ đề xuất giải nào có tỷ lệ trúng 20 kỳ gần nhất VƯỢT 64%.

═══════════════════════════════════════════════════════════════════════════════
QUY TẮC (người dùng chọn — phương án A)
    Mỗi ngày, với từng giải của từng đài quay hôm đó, lấy 150 kỳ gần nhất:
      ├── Kỳ 1–130  : thử 10 họ logic, chọn họ PHÙ HỢP NHẤT (theo thứ hạng)
      └── Kỳ 131–150: đo TỶ LỆ TRÚNG của họ đó trên 20 kỳ gần nhất
    Tỷ lệ > 64% (tức >= 13/20) -> ĐỀ XUẤT 64 con hôm nay bằng họ đó
    Tỷ lệ <= 64%               -> ĐỂ TRỐNG

ĐIỀU ĐÃ ĐO TRƯỚC KHI NGƯỜI DÙNG CHỌN (400 giải mô phỏng)
    Giải HOÀN TOÀN ngẫu nhiên : có đề xuất 56% số ngày, kỳ hôm nay trúng ~64%
    Giải có logic THẬT        : có đề xuất 100%,        kỳ hôm nay trúng ~94%
    -> Không bỏ sót logic thật, nhưng hơn nửa giải ngẫu nhiên cũng lọt qua.
       Cửa sổ 20 kỳ có sai số ±10,7%.
    Ngưỡng 64% là mốc BỐC BỪA; hoà vốn ở tỷ lệ trả 95 là 67,37%.

SỔ TIẾN CỨU — để dữ liệu tự trả lời quy tắc có giá trị không
    Mỗi ngày ghi cho MỌI giải: bộ 64 con của họ logic được chọn (kể cả giải để
    trống — gọi là "bóng") + 1 bộ 64 con ngẫu nhiên đối chứng. Hôm sau tự chấm.
    Báo cáo tách 3 nhóm: ĐỀ XUẤT · BỊ ĐỂ TRỐNG · NGẪU NHIÊN.
    Nếu nhóm ĐỀ XUẤT không trúng nhiều hơn 2 nhóm kia qua nhiều tuần, quy tắc
    lọc 20 kỳ không mang lại lợi thế.

10 HỌ LOGIC: Tần suất · Cửa sổ 10/20/30/50 · Gần đây nửa đời 5/10/20 · Độc lập
    chữ số · Markov · Tổng chữ số · Gộp giải cùng đài · Lịch (cùng thứ, đài quay
    >= 2 thứ/tuần) · Gan · Đều (đối chứng).
    Chọn họ bằng THỨ HẠNG của con thật sự ra, không bằng trúng/trượt: trúng/trượt
    hay chọn nhầm họ học nhanh (bắt được logic thật 33% so với 38%).

GIỚI HẠN: máy quay công bằng thì mọi bộ 64 con trúng 64%, kỳ vọng -5%.
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, json, math, html, zlib, smtplib, traceback
from datetime import datetime, timedelta, timezone, date
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

SO_KY        = 150    # số kỳ gần nhất
N_DO         = 20     # số kỳ cuối dùng ĐO tỷ lệ trúng
T_BD         = 30     # khám phá bắt đầu dự báo từ kỳ thứ 31
N_MIN        = 80     # ít hơn -> để trống "chưa đủ dữ liệu"
NGUONG       = 0.64   # đề xuất khi tỷ lệ trúng 20 kỳ > ngưỡng (>= 13/20)
SO_CON       = 64
TY_LE_TRA    = 95.0
HOA_VON      = SO_CON / TY_LE_TRA          # 67,37%

CHAY_DB, CHAY_G1, CHAY_G8 = True, True, True
FILE_TT      = "data/do_ni_20.json"
EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")
TEN_GIAI     = {"DB": "① Đặc Biệt", "G1": "② Giải Nhất", "G8": "③ Giải 8"}
A_, B_ = np.arange(100) // 10, np.arange(100) % 10


# ==============================================================================
#  10 HỌ LOGIC — mảng dồn, hàng t = thống kê của mt[:t] (không nhìn tương lai)
# ==============================================================================

def _thu(d):
    return d.weekday() if hasattr(d, "weekday") else date.fromisoformat(str(d)[:10]).weekday()


def _chuan_bi(mt, pool, thu_list):
    n = len(mt); mt = np.asarray(mt)
    X = {"mt": mt, "thu": thu_list}
    one = np.zeros((n, 100)); one[np.arange(n), mt] = 1
    X["c"] = np.vstack([np.zeros(100), np.cumsum(one, 0)])
    pk = np.zeros((n, 100))
    for t, ky in enumerate(pool):
        np.add.at(pk[t], ky, 1)
    X["pool"] = np.vstack([np.zeros(100), np.cumsum(pk, 0)])
    for ten, v in (("d1", mt // 10), ("d2", mt % 10), ("tong", (mt // 10 + mt % 10) % 10)):
        m = np.zeros((n, 10)); m[np.arange(n), v] = 1
        X[ten] = np.vstack([np.zeros(10), np.cumsum(m, 0)])
    for H in (5, 10, 20):
        R = np.zeros((n + 1, 100)); dec = 0.5 ** (1.0 / H)
        for t in range(1, n + 1):
            R[t] = R[t - 1] * dec; R[t, mt[t - 1]] += 1
        X[f"gd{H}"] = R
    T1 = np.zeros((n + 1, 10, 10)); T2 = np.zeros((n + 1, 10, 10))
    L = np.full((n + 1, 100), -1.0); W = np.zeros((n + 1, 7, 100))
    for t in range(1, n + 1):
        T1[t] = T1[t - 1]; T2[t] = T2[t - 1]; L[t] = L[t - 1]; W[t] = W[t - 1]
        if t >= 2:
            T1[t, mt[t - 2] // 10, mt[t - 1] // 10] += 1
            T2[t, mt[t - 2] % 10, mt[t - 1] % 10] += 1
        L[t, mt[t - 1]] = t - 1
        W[t, thu_list[t - 1], mt[t - 1]] += 1
    X.update(T1=T1, T2=T2, last=L, wd=W)
    return X


def cau_hinh(co_lich):
    ds = ["TẦN SUẤT", "CỬA SỔ 10", "CỬA SỔ 20", "CỬA SỔ 30", "CỬA SỔ 50",
          "GẦN ĐÂY 5", "GẦN ĐÂY 10", "GẦN ĐÂY 20", "ĐỘC LẬP", "MARKOV",
          "TỔNG CS", "GỘP GIẢI", "GAN", "ĐỀU"]
    if co_lich:
        ds.insert(12, "LỊCH")
    return ds


def diem(cfg, X, t, thu_t):
    mt = X["mt"]
    if cfg == "TẦN SUẤT":
        return X["c"][t]
    if cfg.startswith("CỬA SỔ"):
        w = int(cfg.split()[-1]); return X["c"][t] - X["c"][max(0, t - w)]
    if cfg.startswith("GẦN ĐÂY"):
        return X[f"gd{int(cfg.split()[-1])}"][t]
    if cfg == "ĐỘC LẬP":
        return np.log(X["d1"][t] + 1)[A_] + np.log(X["d2"][t] + 1)[B_]
    if cfg == "MARKOV":
        if t < 2:
            return np.zeros(100)
        a, b = mt[t - 1] // 10, mt[t - 1] % 10
        return np.log(X["T1"][t, a] + 1)[A_] + np.log(X["T2"][t, b] + 1)[B_]
    if cfg == "TỔNG CS":
        return X["tong"][t][(A_ + B_) % 10]
    if cfg == "GỘP GIẢI":
        return X["pool"][t] + X["c"][t]
    if cfg == "LỊCH":
        return X["wd"][t, thu_t]
    if cfg == "GAN":
        return t - X["last"][t]
    return np.zeros(100)                                   # ĐỀU


def _thu_tu(s):
    return np.argsort(-np.asarray(s, float), kind="stable")


def top64(s):
    return sorted(int(i) for i in _thu_tu(s)[:SO_CON])


def _trung(cfg, X, t0, t1):
    mt, thu = X["mt"], X["thu"]
    return sum(1 for t in range(t0, t1) if mt[t] in set(_thu_tu(diem(cfg, X, t, thu[t]))[:SO_CON].tolist()))


def _hang(cfg, X, t0, t1):
    mt, thu = X["mt"], X["thu"]
    r = []
    for t in range(t0, t1):
        pos = np.empty(100); pos[_thu_tu(diem(cfg, X, t, thu[t]))] = np.arange(100)
        r.append(pos[mt[t]])
    return float(np.mean(r)) if r else 99.0


def quyet_dinh(mt, pool, thu_list, thu_hn):
    """Quy tắc A cho 1 giải. Trả dict: họ logic, trúng k/20, có đề xuất không, 64 con."""
    n = len(mt)
    if n < N_MIN:
        return {"ket_luan": "THIẾU", "ly_do": f"chỉ {n} kỳ, cần >= {N_MIN}"}
    X = _chuan_bi(mt, pool, thu_list)
    ds = cau_hinh(len(set(thu_list)) >= 2)
    t_do = n - N_DO
    hang = {c: _hang(c, X, T_BD, t_do) for c in ds}
    ho = min(ds, key=lambda c: (hang[c], ds.index(c)))
    k = _trung(ho, X, t_do, n)
    so = top64(diem(ho, X, n, thu_hn))
    return {"ket_luan": "ĐỀ XUẤT" if k / N_DO > NGUONG else "ĐỂ TRỐNG",
            "ho": ho, "k": k, "ty_le": k / N_DO, "so": so}


def ngau_nhien_64(ngay, stt, giai):
    rng = np.random.default_rng(zlib.crc32(f"{ngay}|{stt}|{giai}".encode()))
    return sorted(int(x) for x in rng.choice(100, SO_CON, replace=False))


# ==============================================================================
#  SỔ TIẾN CỨU
# ==============================================================================

def _vi_tri(tg, giai):
    if giai == "DB":
        return 0
    if giai == "G1":
        return 1
    if giai == "G8" and len(tg[0]) == 18 and len(tg[0][-1]) == 2:
        return len(tg[0]) - 1
    return None


def _chuoi(so):
    return ",".join(f"{v:02d}" for v in so)


def wilson(t, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p, d = t / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def nap_tt():
    if os.path.exists(FILE_TT):
        try:
            tt = json.load(open(FILE_TT, encoding="utf-8")); tt.setdefault("so", [])
            return tt
        except Exception:
            pass
    return {"so": [], "cap_nhat": None}


def luu_tt(tt):
    os.makedirs(os.path.dirname(FILE_TT) or ".", exist_ok=True)
    tt["cap_nhat"] = datetime.now(VN).isoformat(timespec="seconds")
    json.dump(tt, open(FILE_TT, "w", encoding="utf-8"), ensure_ascii=False)


def cham_cu(tt, hom_nay):
    cache = {}

    def tra(stt, giai, ngay):
        if stt not in cache:
            m = E.lay_tu_master(stt, 60)
            cache[stt] = {str(d)[:10]: ky for ky, d in zip(m[0], m[1])} if m else {}
        ky = cache[stt].get(ngay)
        vi = _vi_tri([ky], giai) if ky else None
        return None if vi is None else int(ky[vi][-2:])

    moi = []
    for r in tt["so"]:
        if r.get("ket_qua") is not None or r["ngay"] >= hom_nay.isoformat():
            continue
        y = tra(r["stt"], r["giai"], r["ngay"])
        if y is None:
            continue
        r["ket_qua"] = {"so_ve": y,
                        "trung": y in set(int(x) for x in r["so"].split(",")),
                        "trung_ngau": y in set(int(x) for x in r["ngau"].split(","))}
        moi.append(r)
    return moi


def thanh_tich(tt):
    da = [r for r in tt["so"] if r.get("ket_qua")]
    nhom = {"ĐỀ XUẤT": [0, 0], "ĐỂ TRỐNG (bóng)": [0, 0], "NGẪU NHIÊN": [0, 0]}
    for r in da:
        k = "ĐỀ XUẤT" if r["de_xuat"] else "ĐỂ TRỐNG (bóng)"
        nhom[k][0] += int(r["ket_qua"]["trung"]); nhom[k][1] += 1
        nhom["NGẪU NHIÊN"][0] += int(r["ket_qua"]["trung_ngau"]); nhom["NGẪU NHIÊN"][1] += 1
    return nhom


# ==============================================================================
#  CHẠY 1 NGÀY
# ==============================================================================

def chay(ngay=None, gui_mail=True):
    hom_nay = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    thu = E.THU_VN[hom_nay.weekday()]
    hn = hom_nay.isoformat()
    tt = nap_tt()

    print("=" * 80)
    print(f"  ĐO NI 130+20 — PHƯƠNG ÁN A  |  {thu} {hom_nay:%d.%m.%Y}")
    print(f"  Đề xuất khi tỷ lệ trúng {N_DO} kỳ gần nhất > {NGUONG:.0%}  ·  hoà vốn {HOA_VON:.2%}")
    print("=" * 80)

    moi_cham = cham_cu(tt, hom_nay)
    if moi_cham:
        print(f"\n[CHẤM HÔM TRƯỚC] {len(moi_cham)} bộ vừa có kết quả")

    dsach = E.dai_theo_ngay(hom_nay)
    lich = E.xay_lich()
    dai_list = []
    for s in dsach:
        ten = lich.get(str(s), {}).get("ten", E.lay_dai(s)[0])
        mien = lich.get(str(s), {}).get("mien", E.lay_dai(s)[2])
        d = {"stt": s, "dai": ten, "mien": mien, "o": [], "loi": ""}
        try:
            m = E.lay_tu_master(s, SO_KY + 30)
            if m:
                tg_all, ng_all, _ = m
            else:
                _, _, ng_all, tg_all, _ = E.lay_du_lieu(s, SO_KY + 30)
            tg, ng, _ = E.cat_truoc_ngay(tg_all, ng_all, hom_nay, SO_KY)
        except Exception as e:
            tg, ng = [], []
            d["loi"] = f"không lấy được dữ liệu: {e}"
        print(f"\n  {ten.upper()}" + (f"   ⚠ {d['loi']}" if d["loi"] else ""))
        for giai, bat in (("DB", CHAY_DB), ("G1", CHAY_G1), ("G8", CHAY_G8)):
            if not bat:
                continue
            if not tg:
                if giai == "G8" and E.lay_dai(s)[1] == "xsmb":
                    continue
                o = {"giai": giai, "ket_luan": "LỖI", "ly_do": d["loi"] or "không có dữ liệu"}
            else:
                vi = _vi_tri(tg, giai)
                if vi is None:
                    continue
                mt = [int(ky[vi][-2:]) for ky in tg]
                pool = [[int(x[-2:]) for j, x in enumerate(ky) if j != vi] for ky in tg]
                try:
                    o = {"giai": giai, **quyet_dinh(mt, pool, [_thu(x) for x in ng], hom_nay.weekday())}
                except Exception as e:
                    o = {"giai": giai, "ket_luan": "LỖI", "ly_do": f"lỗi tính toán: {e}"}
            if "so" in o:
                o["ngau"] = ngau_nhien_64(hn, s, giai)
            d["o"].append(o)
            chi = (f"{o['ho']:<11} trúng {o['k']}/{N_DO} = {o['ty_le']:.0%}" if "ho" in o
                   else o.get("ly_do", ""))
            print(f"     {TEN_GIAI[giai]:<14} {o['ket_luan']:<10} {chi}")
        dai_list.append(d)

    da_co = {(r["ngay"], r["stt"], r["giai"]) for r in tt["so"]}
    for d in dai_list:
        for o in d["o"]:
            if "so" in o and (hn, d["stt"], o["giai"]) not in da_co:
                tt["so"].append({"ngay": hn, "stt": d["stt"], "dai": d["dai"], "giai": o["giai"],
                                 "de_xuat": o["ket_luan"] == "ĐỀ XUẤT", "ho": o["ho"], "k": o["k"],
                                 "so": _chuoi(o["so"]), "ngau": _chuoi(o["ngau"]), "ket_qua": None})
    luu_tt(tt)
    tk = thanh_tich(tt)
    tat = [o for d in dai_list for o in d["o"]]
    n_dx = sum(1 for o in tat if o["ket_luan"] == "ĐỀ XUẤT")
    print(f"\n  {n_dx} đề xuất / {len(tat)} giải")
    for k, (a, b) in tk.items():
        if b:
            print(f"  Tiến cứu {k:<16} {a}/{b} = {a/b:.1%}")

    bc = {"ngay": hom_nay, "thu": thu, "dai": dai_list, "tk": tk, "n_dx": n_dx, "n_o": len(tat)}
    if gui_mail:
        print("\n  Đang gửi email...")
        gui_email(bc)
        print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")
    return bc


# ==============================================================================
#  EMAIL
# ==============================================================================

def _html(bc):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    ng, thu, tk = bc["ngay"], bc["thu"], bc["tk"]
    h = [f'<div style="{css}max-width:820px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">Đo ni 130+20 — {thu} {ng:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 10px;font-size:13px">{len(bc["dai"])} đài · '
             f'<b>{bc["n_dx"]}</b> đề xuất / {bc["n_o"]} giải · đề xuất khi tỷ lệ trúng '
             f'{N_DO} kỳ gần nhất &gt; {NGUONG:.0%}</p>')
    h.append(f'<div style="background:#fff8e1;border-left:4px solid #f9a825;padding:9px 12px;'
             f'margin:0 0 16px;font-size:12px;line-height:1.6">Ba mốc: bốc bừa <b>64%</b> · hoà vốn '
             f'<b>{HOA_VON:.2%}</b> · kỳ vọng <b>−5%</b> (tỷ lệ trả 95).<br>Cửa sổ 20 kỳ có sai số '
             f'±10,7% — một giải hoàn toàn ngẫu nhiên vẫn vượt 64% trong khoảng 56% số ngày.</div>')

    h.append('<div style="font-size:15px;font-weight:700;margin:0 0 6px">THÀNH TÍCH TIẾN CỨU</div>')
    if not any(b for _, b in tk.values()):
        h.append('<p style="font-size:13px;color:#607d8b;margin:0 0 16px">Chưa có bộ số nào được '
                 'chấm. Sổ bắt đầu tích luỹ từ ngày mai.</p>')
    else:
        h.append('<table style="font-size:12px;border-collapse:collapse;margin:0 0 6px">'
                 '<tr style="background:#eceff1"><th style="padding:4px 10px;text-align:left">Nhóm</th>'
                 '<th style="padding:4px 10px">Đúng/Tổng</th><th style="padding:4px 10px">Tỷ lệ</th>'
                 '<th style="padding:4px 10px">KTC 95%</th></tr>')
        for k, (a, b) in tk.items():
            if b:
                lo, hi = wilson(a, b)
                h.append(f'<tr><td style="padding:4px 10px">{k}</td>'
                         f'<td style="padding:4px 10px;text-align:center">{a}/{b}</td>'
                         f'<td style="padding:4px 10px;text-align:center"><b>{a/b:.1%}</b></td>'
                         f'<td style="padding:4px 10px;text-align:center">[{lo:.0%}, {hi:.0%}]</td></tr>')
        h.append('</table><p style="font-size:12px;color:#607d8b;margin:0 0 16px">Nhóm ĐỀ XUẤT phải '
                 'trúng nhiều hơn cả nhóm ĐỂ TRỐNG lẫn NGẪU NHIÊN qua nhiều tuần thì quy tắc lọc 20 kỳ '
                 'mới có giá trị.</p>')

    for d in bc["dai"]:
        h.append(f'<div style="margin:14px 0 0;padding:8px 12px;background:#263238;color:#fff;'
                 f'border-radius:5px 5px 0 0"><b style="font-size:16px">{html.escape(d["dai"].upper())}</b>'
                 f'<span style="font-size:12px;opacity:.8"> &nbsp;|&nbsp; {html.escape(str(d["mien"]))}'
                 f' &nbsp;|&nbsp; {thu} {ng:%d.%m.%Y}</span></div>'
                 '<div style="border:1px solid #cfd8dc;border-top:0;border-radius:0 0 5px 5px;'
                 'padding:4px 12px 12px">')
        for o in d["o"]:
            dx = o["ket_luan"] == "ĐỀ XUẤT"
            mau = "#2e7d32" if dx else "#90a4ae"
            if "ho" in o:
                ly = (f'Họ logic <b>{html.escape(o["ho"])}</b> · trúng {o["k"]}/{N_DO} = '
                      f'<b>{o["ty_le"]:.0%}</b> trong {N_DO} kỳ gần nhất')
                if not dx:
                    ly += f' (≤ {NGUONG:.0%})'
            else:
                ly = html.escape(o.get("ly_do", ""))
            h.append(f'<div style="margin:9px 0 0"><b style="font-size:14px">{TEN_GIAI[o["giai"]]}</b>'
                     f' <span style="background:{mau};color:#fff;padding:1px 7px;border-radius:3px;'
                     f'font-size:11px">{o["ket_luan"]}</span>'
                     f'<div style="font-size:11px;color:#546e7a;margin-top:3px">{ly}</div>')
            if dx:
                h.append(f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;'
                         f'background:#e8f5e9;border-left:4px solid {mau};padding:8px 10px;margin-top:4px;'
                         f'word-break:break-all;line-height:1.85">{_chuoi(o["so"])}</div>')
            h.append('</div>')
        h.append('</div>')

    h.append('<hr style="margin:22px 0 10px;border:0;border-top:1px solid #ddd">'
             '<p style="font-size:12px;color:#888;line-height:1.6">Quy tắc: 130 kỳ đầu chọn họ logic '
             'phù hợp nhất (10 họ, chọn theo thứ hạng) → 20 kỳ gần nhất đo tỷ lệ trúng → vượt 64% thì '
             'đề xuất, không thì để trống. Máy quay công bằng thì mọi bộ 64 con trúng 64% và kỳ vọng '
             '−5%; bảng tiến cứu sẽ cho biết quy tắc lọc có mang lại lợi thế hay không.</p></div>')
    return "".join(h)


def _text(bc):
    t = [f"ĐO NI 130+20 — {bc['thu']} {bc['ngay']:%d.%m.%Y}",
         f"{bc['n_dx']} đề xuất / {bc['n_o']} giải", ""]
    for d in bc["dai"]:
        t.append(d["dai"].upper())
        for o in d["o"]:
            if o["ket_luan"] == "ĐỀ XUẤT":
                t.append(f"  {TEN_GIAI[o['giai']]} [ĐỀ XUẤT · {o['ho']} · {o['k']}/{N_DO}]")
                t.append("  " + _chuoi(o["so"]))
            else:
                chi = f"{o['ho']} {o['k']}/{N_DO}" if "ho" in o else o.get("ly_do", "")
                t.append(f"  {TEN_GIAI[o['giai']]} [{o['ket_luan']} · {chi}]")
        t.append("")
    return "\n".join(t)


def gui_email(bc):
    mk = E._lay_mat_khau()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"[ĐO NI 20] {bc['thu']} {bc['ngay']:%d.%m.%Y} — "
                      + (f"{bc['n_dx']} đề xuất / {bc['n_o']} giải" if bc["n_dx"] else "không có đề xuất"))
    msg["From"] = formataddr(("XSMN Đo Ni 20", EMAIL_GUI))
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
            print("  Chưa có kho — quét lần đầu..."); E.tao_master(so_ky=200)
        else:
            print(f"  Kho: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
            E.cap_nhat_master(so_ky_moi=20)
        chay(ngay)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
