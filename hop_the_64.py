"""
HOP_THE_64 — ENGINE HỢP THỂ PHIÊN BẢN TRÙM CUỐI
             Mỗi đài × mỗi giải (ĐB · G1 · G8) × mỗi ngày -> LUÔN CÓ 64 CON.

═══════════════════════════════════════════════════════════════════════════════
KHÁC BẢN HỢP THỂ TRƯỚC (hop_the.py)
    · BỎ cổng lọc nhiễu. Không ô nào bị từ chối đề xuất.
    · Cổng lọc cũ chuyển thành THANG ĐỘ TIN CẬY — chỉ để tham khảo, KHÔNG chặn.
    · Sổ tiến cứu ghi MỌI bộ số, kèm 1 BỘ NGẪU NHIÊN ĐỐI CHỨNG cho từng ô.
    · Đài lỗi hoặc thiếu dữ liệu VẪN có 64 con, kèm cảnh báo và lý do.

LÕI CHỌN SỐ — giữ nguyên, vì đây là cách chọn có lý do tốt nhất
    7 mô hình ước lượng xác suất từng con 00-99:
        ĐỀU · TẦN SUẤT · DIRICHLET · JAMES-STEIN · ĐỘC LẬP · GẦN ĐÂY · GỘP GIẢI
    Gộp bằng BMA tiến cứu có nhiệt độ (= Hedge với log-score).
    64 con = 64 con có xác suất gộp cao nhất.

    LÝ DO chọn 64 con cao nhất (chiến lược thống trị yếu):
      · Máy quay CÔNG BẰNG -> mọi bộ 64 con đều trúng đúng 64%. Không mất gì.
      · Máy quay CÓ LỆCH   -> các con xác suất cao nằm ở phía được lệch. Có lợi.
    Không bao giờ tệ hơn cách chọn nào khác, đôi khi tốt hơn.

THANG ĐỘ TIN CẬY — kiểm ngoài mẫu trên 2 khối rời nhau, KHÔNG chặn bộ số
    CAO        : cả 2 khối vượt hoà vốn 67,37% VÀ p gộp < 0,05 / số ô hôm nay
    TRUNG BÌNH : tỷ lệ gộp 2 khối vượt hoà vốn, nhưng chưa đủ ý nghĩa thống kê
    THẤP       : tỷ lệ gộp 2 khối không vượt hoà vốn
    Backtest 5.350 lượt thật: 106/107 ô về đúng mốc 64% -> đa số ô sẽ là THẤP.
    Đó là sự thật về dữ liệu, không phải engine chọn kém.

KỶ LUẬT CÒN GIỮ
    · Mọi bộ số ghi vào sổ TRƯỚC giờ quay, commit có dấu thời gian.
    · Hôm sau tự chấm cả bộ ENGINE lẫn bộ NGẪU NHIÊN trên cùng kết quả.
    · Báo cáo tách theo độ tin cậy -> sau vài tuần biết thang có giá trị không.

GIỚI HẠN KHÔNG VƯỢT ĐƯỢC
    Máy quay công bằng: 64 con bất kỳ trúng 64%, tỷ lệ trả 95 -> kỳ vọng -5%
    trên MỌI bộ số, kể cả bộ của engine này. Engine chọn số có lý do nhất có
    thể, nhưng không đổi được định lý đó.
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, json, math, html, zlib, smtplib, traceback
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

SO_KY        = 200    # số kỳ lịch sử mỗi đài
MIN_KY_DANH_GIA = 180 # ít hơn -> vẫn ra 64 con, nhưng không chấm được độ tin cậy
SO_CON       = 64
TY_LE_TRA    = 95.0
HOA_VON      = SO_CON / TY_LE_TRA          # 67,37%
MOC          = SO_CON / 100.0              # 64,00%
ALPHA        = 0.05

TAU          = 0.5    # nhiệt độ BMA — không để mô hình nào thắng tuyệt đối
SAN_W        = 0.02   # sàn trọng số
HALF_LIFE    = 30     # nửa đời (kỳ) cho mô hình GẦN ĐÂY
T0           = 20     # bắt đầu chấm log-score từ kỳ thứ T0
K_TIEN       = 100    # tiên nghiệm hoài nghi: dự báo = tỷ lệ ngoài mẫu co 1 nửa về 64%

CHAY_DB, CHAY_G1, CHAY_G8 = True, True, True
FILE_TT      = "data/hop_the_64_theo_doi.json"
EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")
TEN_GIAI     = {"DB": "① Đặc Biệt", "G1": "② Giải Nhất", "G8": "③ Giải 8"}
MAU_DO_TIN   = {"CAO": "#2e7d32", "TRUNG BÌNH": "#ef6c00", "THẤP": "#78909c",
                "CHƯA ĐỦ DỮ LIỆU": "#90a4ae", "DỰ PHÒNG": "#c62828"}

MO_HINH = ["ĐỀU", "TẦN SUẤT", "DIRICHLET", "JAMES-STEIN", "ĐỘC LẬP", "GẦN ĐÂY", "GỘP GIẢI"]
M = len(MO_HINH)
A_, B_ = np.arange(100) // 10, np.arange(100) % 10
DEU = np.full(100, 0.01)
LUOI_ALPHA = np.exp(np.linspace(np.log(0.05), np.log(500), 30))


# ==============================================================================
#  LÕI — 7 MÔ HÌNH XÁC SUẤT, CẬP NHẬT TĂNG DẦN
# ==============================================================================

def _p_dong_nhat(a, b):
    T = np.vstack([a, b]) + 0.5
    R = T.sum(1, keepdims=True); C = T.sum(0, keepdims=True)
    Ex = R @ C / T.sum()
    return float(stats.chi2.sf(((T - Ex) ** 2 / Ex).sum(), 9))


class ThongKe:
    def __init__(self):
        self.c = np.zeros(100); self.n = 0
        self.d1 = np.zeros(10); self.d2 = np.zeros(10)
        self.pc = np.zeros(100); self.pd1 = np.zeros(10); self.pd2 = np.zeros(10)
        self.cw = np.zeros(100); self.decay = 0.5 ** (1.0 / HALF_LIFE)

    def them(self, y, pool_ky):
        self.c[y] += 1; self.n += 1
        self.d1[y // 10] += 1; self.d2[y % 10] += 1
        self.cw *= self.decay; self.cw[y] += 1
        for v in pool_ky:
            self.pc[v] += 1; self.pd1[v // 10] += 1; self.pd2[v % 10] += 1

    def mo_hinh(self):
        n, c = self.n, self.c
        P = np.empty((M, 100))
        P[0] = DEU
        P[1] = (c + 1) / (n + 100)
        a = LUOI_ALPHA
        ml = (gammaln(100 * a) - gammaln(n + 100 * a)
              + gammaln(c[None, :] + a[:, None]).sum(1) - 100 * gammaln(a))
        al = float(a[int(np.argmax(ml))])
        P[2] = (c + al) / (n + 100 * al)
        if n > 0:
            ph = c / n; S = float(((ph - DEU) ** 2).sum())
            B = 0.0 if S <= 1e-12 else float(np.clip(1 - 97 * (0.01 * 0.99 / n) / S, 0, 1))
            q = np.clip(DEU + B * (ph - DEU), 1e-9, None); P[3] = q / q.sum()
        else:
            B = 0.0; P[3] = DEU
        p1 = (self.d1 + 1) / (n + 10); p2 = (self.d2 + 1) / (n + 10)
        q = p1[A_] * p2[B_]; P[4] = q / q.sum()
        P[5] = (self.cw + 0.5) / (self.cw.sum() + 50)
        if self.pc.sum() > 0 and n >= 10:
            p_dn = min(1.0, 2 * min(_p_dong_nhat(self.d1, self.pd1),
                                    _p_dong_nhat(self.d2, self.pd2)))
        else:
            p_dn = 1.0
        P[6] = (c + self.pc + 1) / (n + self.pc.sum() + 100) if p_dn >= ALPHA else P[1]
        return P, {"alpha": al, "B": B, "p_dong_nhat": p_dn}


def trong_so(S):
    z = TAU * (S - S.max())
    w = np.exp(z); w /= w.sum()
    return (1 - SAN_W) * w + SAN_W / len(w)


def khop(mt, pool):
    """Học 7 mô hình + trọng số CHỈ từ dữ liệu đưa vào. Log-score tiến cứu."""
    st = ThongKe()
    S = np.zeros(M)
    for t, y in enumerate(mt):
        if t >= T0:
            Ps, _ = st.mo_hinh()
            S += np.log(np.clip(Ps[:, y], 1e-12, None))
        st.them(y, pool[t])
    Ps, chan_doan = st.mo_hinh()
    w = trong_so(S)
    P = w @ Ps
    return {"P": P / P.sum(), "w": w, "chan_doan": chan_doan}


def top64(P, k=SO_CON):
    return sorted(int(i) for i in np.argsort(-P, kind="stable")[:k])


# ==============================================================================
#  TIỆN ÍCH
# ==============================================================================

def p_tren(trung, n, p0=MOC):
    return float(1 - stats.binom.cdf(trung - 1, n, p0)) if n > 0 else 1.0


def wilson(t, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p, d = t / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


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


def canh_bao_du_lieu(tg, ng, mt):
    """Không chặn — chỉ trả cảnh báo nếu dữ liệu đáng ngờ."""
    if len(set(str(d)[:10] for d in ng)) != len(ng):
        return "có ngày bị lưu lặp"
    if any(tg[i] == tg[i - 1] for i in range(1, len(tg))):
        return "có kỳ trùng khít kỳ liền trước"
    ky_vong = 100 * (1 - 0.99 ** len(mt))
    if len(mt) >= 30 and len(set(mt)) < 0.6 * ky_vong:
        return f"chỉ {len(set(mt))} giá trị khác nhau (kỳ vọng ~{ky_vong:.0f}) — nghi bóc sai ô"
    return ""


def ngau_nhien_64(ngay, stt, giai):
    """Bộ đối chứng: 64 con ngẫu nhiên, TÁI LẬP ĐƯỢC (hạt giống cố định theo ô + ngày)."""
    rng = np.random.default_rng(zlib.crc32(f"{ngay}|{stt}|{giai}".encode()))
    return sorted(int(x) for x in rng.choice(100, SO_CON, replace=False))


# ==============================================================================
#  THANG ĐỘ TIN CẬY — kiểm ngoài mẫu 2 khối rời nhau, KHÔNG chặn bộ số
# ==============================================================================

def danh_gia_do_tin(mt, pool, n_o):
    n = len(mt)
    if n < MIN_KY_DANH_GIA:
        return {"muc": "CHƯA ĐỦ DỮ LIỆU"}
    fA = khop(mt[:n - 100], pool[:n - 100]); sA = set(top64(fA["P"]))
    tA = sum(1 for v in mt[n - 100:n - 50] if v in sA)
    fB = khop(mt[:n - 50], pool[:n - 50]); sB = set(top64(fB["P"]))
    tB = sum(1 for v in mt[n - 50:] if v in sB)
    p = p_tren(tA + tB, 100)
    if tA / 50 > HOA_VON and tB / 50 > HOA_VON and p < ALPHA / max(n_o, 1):
        muc = "CAO"
    elif (tA + tB) / 100 > HOA_VON:
        muc = "TRUNG BÌNH"
    else:
        muc = "THẤP"
    return {"muc": muc, "tA": tA, "tB": tB, "p": p,
            "du_bao": (tA + tB + K_TIEN * MOC) / (100 + K_TIEN)}


# ==============================================================================
#  SỔ TIẾN CỨU — ghi mọi bộ số + bộ ngẫu nhiên, hôm sau tự chấm
# ==============================================================================

def nap_tt():
    if os.path.exists(FILE_TT):
        try:
            tt = json.load(open(FILE_TT, encoding="utf-8"))
            tt.setdefault("so", [])
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
        if ky is None:
            return None
        vi = _vi_tri([ky], giai)
        return None if vi is None else int(ky[vi][-2:])

    moi = []
    for r in tt["so"]:
        if r.get("ket_qua") is not None or r["ngay"] >= hom_nay.isoformat():
            continue
        y = tra(r["stt"], r["giai"], r["ngay"])
        if y is None:
            continue
        eng = set(int(x) for x in r["so"].split(","))
        nn = set(int(x) for x in r["ngau"].split(","))
        r["ket_qua"] = {"so_ve": y, "trung": y in eng, "trung_ngau": y in nn}
        moi.append(r)
    return moi


def thanh_tich(tt):
    da = [r for r in tt["so"] if r.get("ket_qua")]
    kq = {"N": len(da),
          "eng": sum(r["ket_qua"]["trung"] for r in da),
          "ngau": sum(r["ket_qua"]["trung_ngau"] for r in da),
          "theo_muc": {}, "theo_giai": {}}
    for r in da:
        for nhom, khoa in (("theo_muc", r.get("muc", "?")), ("theo_giai", r["giai"])):
            x = kq[nhom].setdefault(khoa, {"n": 0, "eng": 0, "ngau": 0})
            x["n"] += 1
            x["eng"] += int(r["ket_qua"]["trung"])
            x["ngau"] += int(r["ket_qua"]["trung_ngau"])
    return kq


# ==============================================================================
#  CHẠY 1 NGÀY — MỌI Ô ĐỀU RA 64 CON
# ==============================================================================

def chay(ngay=None, gui_mail=True):
    hom_nay = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    thu = E.THU_VN[hom_nay.weekday()]
    hn = hom_nay.isoformat()
    tt = nap_tt()

    print("=" * 80)
    print(f"  ENGINE HỢP THỂ 64 — TRÙM CUỐI  |  {thu} {hom_nay:%d.%m.%Y}")
    print(f"  Mọi đài × mọi giải đều có 64 con · 7 mô hình · BMA τ={TAU}")
    print("=" * 80)

    moi_cham = cham_cu(tt, hom_nay)
    if moi_cham:
        te = sum(r["ket_qua"]["trung"] for r in moi_cham)
        tn = sum(r["ket_qua"]["trung_ngau"] for r in moi_cham)
        print(f"\n[CHẤM HÔM TRƯỚC] {len(moi_cham)} bộ · engine trúng {te} · ngẫu nhiên trúng {tn}")

    # ---- Dựng danh sách đài × giải hôm nay ----
    dsach = E.dai_theo_ngay(hom_nay)
    lich = E.xay_lich()
    dai_list = []
    for s in dsach:
        ten = lich.get(str(s), {}).get("ten", E.lay_dai(s)[0])
        mien = lich.get(str(s), {}).get("mien", E.lay_dai(s)[2])
        d = {"stt": s, "dai": ten, "mien": mien, "o": [], "loi": ""}
        try:
            m = E.lay_tu_master(s, SO_KY + 20)
            if m:
                tg_all, ng_all, _ = m
            else:
                _, _, ng_all, tg_all, _ = E.lay_du_lieu(s, SO_KY + 20)
            tg, ng, _ = E.cat_truoc_ngay(tg_all, ng_all, hom_nay, SO_KY)
        except Exception as e:
            tg, ng = [], []
            d["loi"] = f"không lấy được dữ liệu: {e}"
        cac_giai = [g for g, b in (("DB", CHAY_DB), ("G1", CHAY_G1), ("G8", CHAY_G8)) if b]
        for giai in cac_giai:
            if tg:
                vi = _vi_tri(tg, giai)
                if vi is None:
                    continue                     # Miền Bắc không có G8
                d["o"].append({"giai": giai,
                               "mt": [int(ky[vi][-2:]) for ky in tg],
                               "pool": [[int(x[-2:]) for j, x in enumerate(ky) if j != vi]
                                        for ky in tg],
                               "canh_bao": canh_bao_du_lieu(tg, ng, [int(ky[vi][-2:]) for ky in tg])})
            else:
                # Đài lỗi dữ liệu: MB không có G8, MN/MT có đủ 3 giải
                if giai == "G8" and E.lay_dai(s)[1] == "xsmb":
                    continue
                d["o"].append({"giai": giai, "mt": [], "pool": [], "canh_bao": d["loi"]})
        dai_list.append(d)

    n_o = sum(len(d["o"]) for d in dai_list)
    print(f"\n  {len(dai_list)} đài · {n_o} bộ số cần ra · ngưỡng độ tin CAO p < {ALPHA/max(n_o,1):.5f}")

    # ---- Tần suất gộp theo giải, làm dự phòng cho ô không có dữ liệu ----
    du_phong = {}
    for d in dai_list:
        for o in d["o"]:
            du_phong.setdefault(o["giai"], []).extend(o["mt"])

    for d in dai_list:
        print(f"\n  {d['dai'].upper()}" + (f"   ⚠ {d['loi']}" if d["loi"] else ""))
        for o in d["o"]:
            if len(o["mt"]) >= 1:
                try:
                    f = khop(o["mt"], o["pool"])
                    o["so"] = top64(f["P"]); o["w"] = f["w"]
                    o["ly_do"] = f"BMA 7 mô hình trên {len(o['mt'])} kỳ"
                    o.update(danh_gia_do_tin(o["mt"], o["pool"], n_o))
                except Exception as e:
                    o["mt"] = []; o["canh_bao"] = f"lỗi tính toán: {e}"
            if "so" not in o:                     # DỰ PHÒNG — vẫn phải có 64 con
                goc = du_phong.get(o["giai"], [])
                if goc:
                    c = np.ones(100)
                    for v in goc:
                        c[v] += 1
                    o["so"] = top64(c)
                    o["ly_do"] = (f"DỰ PHÒNG: tần suất gộp {TEN_GIAI[o['giai']]} "
                                  f"của các đài khác hôm nay ({len(goc)} kỳ)")
                else:
                    o["so"] = list(range(SO_CON))
                    o["ly_do"] = "DỰ PHÒNG: không có dữ liệu nào để lập luận"
                o["muc"] = "DỰ PHÒNG"
            o["ngau"] = ngau_nhien_64(hn, d["stt"], o["giai"])
            chi = (f"A {o['tA']}/50 · B {o['tB']}/50 · dự báo {o['du_bao']:.1%}"
                   if "tA" in o else o["ly_do"])
            print(f"     {TEN_GIAI[o['giai']]:<14} độ tin {o['muc']:<16} {chi}"
                  + (f"   ⚠ {o['canh_bao']}" if o.get("canh_bao") else ""))

    # ---- Ghi sổ TRƯỚC giờ quay ----
    da_co = {(r["ngay"], r["stt"], r["giai"]) for r in tt["so"]}
    for d in dai_list:
        for o in d["o"]:
            k = (hn, d["stt"], o["giai"])
            if k not in da_co:
                tt["so"].append({"ngay": hn, "stt": d["stt"], "dai": d["dai"], "giai": o["giai"],
                                 "muc": o["muc"], "so": _chuoi(o["so"]),
                                 "ngau": _chuoi(o["ngau"]), "ket_qua": None})
    luu_tt(tt)
    tk = thanh_tich(tt)
    if tk["N"]:
        print(f"\n  TIẾN CỨU: engine {tk['eng']}/{tk['N']} = {tk['eng']/tk['N']:.1%} · "
              f"ngẫu nhiên {tk['ngau']}/{tk['N']} = {tk['ngau']/tk['N']:.1%}")

    bc = {"ngay": hom_nay, "thu": thu, "dai": dai_list, "tk": tk,
          "moi_cham": moi_cham, "n_o": n_o}
    if gui_mail:
        print("\n  Đang gửi email...")
        gui_email(bc)
        print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")
    return bc


# ==============================================================================
#  EMAIL
# ==============================================================================

def _dong_tien_cuu(ten, n, e, g):
    if not n:
        return ""
    lo, hi = wilson(e, n)
    mau = "#2e7d32" if e > g else ("#c62828" if e < g else "#455a64")
    return (f'<tr><td style="padding:4px 9px">{ten}</td>'
            f'<td style="padding:4px 9px;text-align:center">{n}</td>'
            f'<td style="padding:4px 9px;text-align:center"><b>{e/n:.1%}</b></td>'
            f'<td style="padding:4px 9px;text-align:center">{g/n:.1%}</td>'
            f'<td style="padding:4px 9px;text-align:center;color:{mau}">{(e-g)/n:+.1%}</td>'
            f'<td style="padding:4px 9px;text-align:center">[{lo:.0%}, {hi:.0%}]</td></tr>')


def _html(bc):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    ng, thu, tk = bc["ngay"], bc["thu"], bc["tk"]
    dem = {}
    for d in bc["dai"]:
        for o in d["o"]:
            dem[o["muc"]] = dem.get(o["muc"], 0) + 1
    h = [f'<div style="{css}max-width:820px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">64 con — {thu} {ng:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 12px;font-size:13px">{len(bc["dai"])} đài · '
             f'{bc["n_o"]} bộ số · 7 mô hình gộp bằng BMA tiến cứu · mọi ô đều có 64 con</p>')
    h.append('<div style="font-size:12px;margin:0 0 16px">'
             + " ".join(f'<span style="background:{MAU_DO_TIN.get(k,"#999")};color:#fff;'
                        f'padding:2px 8px;border-radius:3px;margin-right:4px">{k}: {v}</span>'
                        for k, v in sorted(dem.items(), key=lambda x: -x[1]))
             + '</div>')

    # ---- Tiến cứu: engine vs ngẫu nhiên ----
    h.append('<div style="font-size:15px;font-weight:700;margin:0 0 6px">'
             'THÀNH TÍCH TIẾN CỨU — engine vs bốc ngẫu nhiên, cùng kết quả thật</div>')
    if not tk["N"]:
        h.append('<p style="font-size:13px;color:#607d8b;margin:0 0 18px">Chưa có bộ số nào '
                 'được chấm. Sổ bắt đầu tích luỹ từ ngày mai.</p>')
    else:
        h.append('<table style="font-size:12px;border-collapse:collapse;margin:0 0 6px">'
                 '<tr style="background:#eceff1"><th style="padding:4px 9px;text-align:left">Nhóm</th>'
                 '<th style="padding:4px 9px">Số bộ</th><th style="padding:4px 9px">Engine</th>'
                 '<th style="padding:4px 9px">Ngẫu nhiên</th><th style="padding:4px 9px">Chênh</th>'
                 '<th style="padding:4px 9px">KTC 95% engine</th></tr>')
        h.append(_dong_tien_cuu("<b>TỔNG</b>", tk["N"], tk["eng"], tk["ngau"]))
        for k in ("CAO", "TRUNG BÌNH", "THẤP", "CHƯA ĐỦ DỮ LIỆU", "DỰ PHÒNG"):
            x = tk["theo_muc"].get(k)
            if x:
                h.append(_dong_tien_cuu(f"Độ tin {k}", x["n"], x["eng"], x["ngau"]))
        for k in ("DB", "G1", "G8"):
            x = tk["theo_giai"].get(k)
            if x:
                h.append(_dong_tien_cuu(TEN_GIAI[k], x["n"], x["eng"], x["ngau"]))
        h.append(f'</table><p style="font-size:12px;color:#607d8b;margin:0 0 18px">Mốc 64% · hoà '
                 f'vốn {HOA_VON:.1%}. Cột "Chênh" dương đều đặn qua nhiều tuần mới là dấu hiệu '
                 f'engine hơn bốc bừa.</p>')

    # ---- Bộ số từng đài ----
    for d in bc["dai"]:
        h.append(f'<div style="margin:16px 0 0;padding:8px 12px;background:#263238;color:#fff;'
                 f'border-radius:5px 5px 0 0"><b style="font-size:16px">{html.escape(d["dai"].upper())}</b>'
                 f'<span style="font-size:12px;opacity:.8"> &nbsp;|&nbsp; {html.escape(str(d["mien"]))}'
                 f' &nbsp;|&nbsp; {thu} {ng:%d.%m.%Y}</span></div>'
                 '<div style="border:1px solid #cfd8dc;border-top:0;border-radius:0 0 5px 5px;'
                 'padding:4px 12px 12px">')
        for o in d["o"]:
            mau = MAU_DO_TIN.get(o["muc"], "#999")
            chi = (f'A {o["tA"]}/50 · B {o["tB"]}/50 · dự báo thận trọng {o["du_bao"]:.1%}'
                   if "tA" in o else html.escape(o["ly_do"]))
            h.append(f'<div style="margin:10px 0 0"><b style="font-size:14px">{TEN_GIAI[o["giai"]]}</b>'
                     f' <span style="background:{mau};color:#fff;padding:1px 7px;border-radius:3px;'
                     f'font-size:11px">{o["muc"]}</span>'
                     f' <span style="font-size:11px;color:#607d8b">{chi}</span>'
                     + (f'<div style="font-size:11px;color:#c62828">⚠ {html.escape(o["canh_bao"])}</div>'
                        if o.get("canh_bao") else '')
                     + f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;'
                     f'background:#e8f5e9;border-left:4px solid {mau};padding:8px 10px;margin-top:4px;'
                     f'word-break:break-all;line-height:1.85">{_chuoi(o["so"])}</div></div>')
        h.append('</div>')

    h.append('<hr style="margin:22px 0 10px;border:0;border-top:1px solid #ddd">'
             '<p style="font-size:12px;color:#888;line-height:1.6">'
             '<b>Lý do chọn 64 con:</b> 64 con có xác suất cao nhất theo 7 mô hình gộp bằng BMA '
             'tiến cứu (chiến lược thống trị yếu — máy công bằng thì không mất gì, máy lệch thì có lợi).'
             '<br><b>Độ tin cậy</b> chỉ để tham khảo, không chặn bộ số: CAO = 2 khối ngoài mẫu vượt '
             'hoà vốn và qua Bonferroni · TRUNG BÌNH = gộp vượt hoà vốn nhưng chưa đủ ý nghĩa · '
             'THẤP = không vượt hoà vốn.<br>Máy quay công bằng thì mọi bộ 64 con trúng 64%, tỷ lệ '
             'trả 95 cho kỳ vọng −5% — kể cả bộ của engine này.</p></div>')
    return "".join(h)


def _text(bc):
    t = [f"64 CON — {bc['thu']} {bc['ngay']:%d.%m.%Y}", ""]
    for d in bc["dai"]:
        t.append(d["dai"].upper())
        for o in d["o"]:
            t.append(f"  {TEN_GIAI[o['giai']]} [{o['muc']}]")
            t.append("  " + _chuoi(o["so"]))
        t.append("")
    tk = bc["tk"]
    if tk["N"]:
        t.append(f"Tiến cứu: engine {tk['eng']}/{tk['N']} · ngẫu nhiên {tk['ngau']}/{tk['N']}")
    return "\n".join(t)


def gui_email(bc):
    mk = E._lay_mat_khau()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"[64 CON] {bc['thu']} {bc['ngay']:%d.%m.%Y} — "
                      f"{len(bc['dai'])} đài · {bc['n_o']} bộ số")
    msg["From"] = formataddr(("XSMN Hợp Thể 64", EMAIL_GUI))
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
