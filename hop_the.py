"""
HOP_THE — ENGINE HỢP THỂ: bản tổng hợp cuối cùng cho 64 con ĐB · G1 · G8.

═══════════════════════════════════════════════════════════════════════════════
NGUYÊN TẮC HỢP THỂ
    Không cộng dồn mọi engine. GIỮ phần đã được CHỨNG MINH có ích, LOẠI phần đã
    được CHỨNG MINH có hại, rồi ghép theo MỘT nguyên lý toán học thống nhất.

TRỤC XƯƠNG SỐNG — một phát hiện khi thiết kế:
    Hedge (engine_thich_nghi) cập nhật  w ← w·exp(η·thưởng).
    Khi "thưởng" = log-score (thongke6), sau T kỳ:  w ∝ exp(η · Σ log P(kết quả))
    Đó CHÍNH LÀ Bayes Model Averaging tiến cứu có nhiệt độ η.
    -> Hai engine tưởng khác nhau thực ra là MỘT thuật toán.

═══════════════════════════════════════════════════════════════════════════════
BẢN ĐỒ HỢP THỂ — mỗi thành phần đến từ đâu

  TẦNG 0  KIỂM TRA DỮ LIỆU ...................... từ kiem_chung_mb.py
          trùng ngày · trùng nguyên kỳ · số giá trị khác nhau bất thường

  TẦNG 1  BẢY MÔ HÌNH XÁC SUẤT (mềm mỏng)
          M0 ĐỀU         mô hình trung thực ....... de_engine12 D12
          M1 TẦN SUẤT    thống trị yếu ............ backtest_tanso.py
          M2 DIRICHLET   α bằng marginal likelihood  thongke6 E1 / D01
          M3 JAMES-STEIN co ngót về đều ........... thongke6 E3 / D02
          M4 ĐỘC LẬP     P(a)·P(b), 20 tham số .... thongke6 E2 / D08
          M5 GẦN ĐÂY     trọng số thời gian ....... de_engine12 D04 / thích nghi
          M6 GỘP GIẢI    mượn 17-26 giải khác nếu đồng nhất .. thongke6 E4 / D06

  GỘP     BMA tiến cứu có nhiệt độ = Hedge log-loss ... thích nghi + thongke6
          · NHIỆT ĐỘ τ < 1: không để một mô hình thắng tuyệt đối (mềm mỏng)
          · SÀN trọng số: không mô hình nào chết hẳn (mềm mỏng)
          · M0 ĐỀU cộng một hằng số cho mọi ô -> KHÔNG đổi thứ hạng 64 con.
            Nó là mô hình trung thực: trên dữ liệu công bằng giữ ~40% trọng số.
          · DỰ BÁO THẬN TRỌNG lấy tỷ lệ trúng NGOÀI MẪU (2 khối) rồi co một nửa
            về 64% — KHÔNG dùng tổng xác suất trong mẫu vì nó luôn thiên cao.

  TẦNG 2  CỔNG TÍN HIỆU (kỷ luật) ............... từ tin_hieu.py
          Hai khối test RỜI NHAU, toàn bộ quy trình (kể cả học trọng số) chỉ
          dùng dữ liệu TRƯỚC mỗi khối -> không rò rỉ.
          Qua khi: khối A > hoà vốn · khối B > hoà vốn · p gộp < 0,05/số ô hôm nay

  TẦNG 3  XÁC NHẬN TIẾN CỨU (kỷ luật) ........... từ tin_hieu.py
          Đề xuất ghi TRƯỚC giờ quay, commit có dấu thời gian, hôm sau tự chấm.
          ≥30 kỳ, vượt hoà vốn và vượt ngẫu nhiên có ý nghĩa -> ĐÃ XÁC NHẬN.
          ≥30 kỳ mà dưới mốc 64% -> BỊ LOẠI, ngừng đề xuất ô đó.

  TẦNG 4  SỔ BÓNG — đối chứng tự nhiên ........... bài học từ ml_nghien_cuu.py
          Ghi 64 con của MỌI ô mỗi ngày (kể cả ô không đề xuất) rồi chấm.
          Nếu ô được đề xuất KHÔNG trúng nhiều hơn ô bóng, cổng tín hiệu
          không thêm giá trị gì — và bạn sẽ thấy điều đó bằng số liệu.

LOẠI BỎ CÓ CHỦ ĐÍCH — kèm bằng chứng
    · Cầu vị trí (cau_engine) ....... cầu tốt nhất 41,8% < mức nhiễu 43,4%
    · Lập trình di truyền (tu_hoc) ... học 81,1% -> ngoài mẫu 58,3%
    · Học máy phức tạp .............. không vượt được nhãn xáo trộn (71,8%)
    · Markov 100 trạng thái (D05) ... 10.000 ô chuyển từ ~200 quan sát
    · Điểm gãy (E5) ................. chưa phát hiện gì, tốn tính toán
    · Quy tắc dân gian (gan, bệt, bóng, tổng, hiệu) — không có cơ chế

GIỚI HẠN KHÔNG VƯỢT ĐƯỢC
    Máy quay công bằng -> 64 con BẤT KỲ trúng đúng 64%. Engine hợp thể không
    phá được định lý đó. Giá trị của nó: im lặng đúng lúc, bắt được độ lệch
    MẠNH nếu có, và tự chấm điểm công khai bằng dữ liệu tương lai.
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, json, math, html, smtplib, traceback
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

SO_KY        = 200    # số kỳ lịch sử (nguồn thường trả tối đa ~200)
MIN_KY       = 180    # ít hơn -> không đủ 2 khối test
SO_CON       = 64
TY_LE_TRA    = 95.0
HOA_VON      = SO_CON / TY_LE_TRA          # 67,37%
MOC          = SO_CON / 100.0              # 64,00%
ALPHA        = 0.05
N_XAC_NHAN   = 30     # kỳ tiến cứu tối thiểu để xác nhận hoặc loại

# --- Mềm mỏng ---
TAU          = 0.5    # nhiệt độ BMA: 1 = Bayes thuần, <1 = không để mô hình nào thắng tuyệt đối
SAN_W        = 0.02   # sàn trọng số: mọi mô hình luôn giữ ít nhất phần này
HALF_LIFE    = 30     # nửa đời (kỳ) cho mô hình GẦN ĐÂY
T0           = 20     # bắt đầu chấm log-score từ kỳ thứ T0 trong mỗi lần học
K_TIEN       = 100    # sức mạnh tiên nghiệm hoài nghi cho "dự báo thận trọng":
                      #   dự báo = (trúng A+B ngoài mẫu + 100·64%) / (100 + 100)
                      #   -> tỷ lệ ngoài mẫu bị CO MỘT NỬA về 64%.
                      # Vì sao không dùng tổng P của 64 ô được chọn? Đo trên máy công bằng:
                      #   tổng đó TB 65,0%, cao nhất 69,1%, trong khi trúng thật luôn 64,0%.
                      #   Chọn ô cao nhất rồi cộng xác suất của CHÍNH chúng luôn thiên cao
                      #   ("lời nguyền người tối ưu") -> không phải dự báo trung thực.

CHAY_DB, CHAY_G1, CHAY_G8 = True, True, True
FILE_TT      = "data/hop_the_theo_doi.json"
EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")
TEN_GIAI     = {"DB": "Đặc Biệt", "G1": "Giải Nhất", "G8": "Giải 8"}

MO_HINH = ["ĐỀU", "TẦN SUẤT", "DIRICHLET", "JAMES-STEIN", "ĐỘC LẬP", "GẦN ĐÂY", "GỘP GIẢI"]
M = len(MO_HINH)
A_, B_ = np.arange(100) // 10, np.arange(100) % 10
DEU = np.full(100, 0.01)
LUOI_ALPHA = np.exp(np.linspace(np.log(0.05), np.log(500), 30))


# ==============================================================================
#  TẦNG 1 — BẢY MÔ HÌNH, CẬP NHẬT TĂNG DẦN (mỗi kỳ O(100), không tính lại từ đầu)
# ==============================================================================

def _p_dong_nhat(a, b):
    """Chi-square đồng nhất 2×10 (tự tính, nhanh hơn scipy)."""
    T = np.vstack([a, b]) + 0.5
    R = T.sum(1, keepdims=True); C = T.sum(0, keepdims=True)
    Ex = R @ C / T.sum()
    return float(stats.chi2.sf(((T - Ex) ** 2 / Ex).sum(), 9))


class ThongKe:
    """Giữ số liệu đếm dồn. them() thêm 1 kỳ, mo_hinh() trả 7 phân phối."""

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
        P[0] = DEU                                            # M0 ĐỀU
        P[1] = (c + 1) / (n + 100)                            # M1 TẦN SUẤT
        # M2 DIRICHLET — α tối đa hoá marginal likelihood (vector hoá)
        a = LUOI_ALPHA
        ml = (gammaln(100 * a) - gammaln(n + 100 * a)
              + gammaln(c[None, :] + a[:, None]).sum(1) - 100 * gammaln(a))
        al = float(a[int(np.argmax(ml))])
        P[2] = (c + al) / (n + 100 * al)
        # M3 JAMES-STEIN — B≈0 nghĩa là mọi chênh lệch là nhiễu đếm
        if n > 0:
            ph = c / n; S = float(((ph - DEU) ** 2).sum())
            B = 0.0 if S <= 1e-12 else float(np.clip(1 - 97 * (0.01 * 0.99 / n) / S, 0, 1))
            q = np.clip(DEU + B * (ph - DEU), 1e-9, None); P[3] = q / q.sum()
        else:
            B = 0.0; P[3] = DEU
        # M4 ĐỘC LẬP — P(a)·P(b)
        p1 = (self.d1 + 1) / (n + 10); p2 = (self.d2 + 1) / (n + 10)
        q = p1[A_] * p2[B_]; P[4] = q / q.sum()
        # M5 GẦN ĐÂY — trọng số giảm dần theo thời gian
        P[5] = (self.cw + 0.5) / (self.cw.sum() + 50)
        # M6 GỘP GIẢI — chỉ mượn khi giải mục tiêu ĐỒNG NHẤT với các giải khác
        if self.pc.sum() > 0 and n >= 10:
            p_dn = min(1.0, 2 * min(_p_dong_nhat(self.d1, self.pd1),
                                    _p_dong_nhat(self.d2, self.pd2)))
        else:
            p_dn = 1.0
        if p_dn >= ALPHA:
            P[6] = (c + self.pc + 1) / (n + self.pc.sum() + 100)
        else:
            P[6] = P[1]
        return P, {"alpha": al, "B": B, "p_dong_nhat": p_dn}


def trong_so(S):
    """BMA tiến cứu có nhiệt độ τ + sàn. S = tổng log-score mỗi mô hình."""
    z = TAU * (S - S.max())
    w = np.exp(z); w /= w.sum()
    return (1 - SAN_W) * w + SAN_W / len(w)


def khop(mt, pool):
    """Học 7 mô hình + trọng số CHỈ từ (mt, pool) được đưa vào. Không nhìn tương lai.
       Chấm log-score tiến cứu: tại kỳ t, mô hình chỉ biết dữ liệu < t."""
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
    return {"P": P / P.sum(), "w": w, "S": S, "chan_doan": chan_doan}


def top64(P, k=SO_CON):
    return sorted(int(i) for i in np.argsort(-P, kind="stable")[:k])


# ==============================================================================
#  TIỆN ÍCH THỐNG KÊ
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


# ==============================================================================
#  TẦNG 0 — KIỂM TRA DỮ LIỆU
# ==============================================================================

def kiem_du_lieu(tg, ng, mt):
    if len(set(str(d)[:10] for d in ng)) != len(ng):
        return False, "có ngày bị lưu lặp"
    if any(tg[i] == tg[i - 1] for i in range(1, len(tg))):
        return False, "có kỳ trùng khít kỳ liền trước"
    ky_vong = 100 * (1 - 0.99 ** len(mt))
    if len(set(mt)) < 0.6 * ky_vong:
        return False, f"chỉ {len(set(mt))} giá trị khác nhau (kỳ vọng ~{ky_vong:.0f}) — nghi bóc sai ô"
    return True, ""


# ==============================================================================
#  TẦNG 2 — CỔNG TÍN HIỆU (toàn bộ quy trình học lại TRƯỚC mỗi khối)
# ==============================================================================

def cong_tin_hieu(mt, pool):
    n = len(mt)
    fA = khop(mt[:n - 100], pool[:n - 100])
    sA = set(top64(fA["P"]))
    tA = sum(1 for v in mt[n - 100:n - 50] if v in sA)
    fB = khop(mt[:n - 50], pool[:n - 50])
    sB = set(top64(fB["P"]))
    tB = sum(1 for v in mt[n - 50:] if v in sB)
    return {"tA": tA, "tB": tB, "rA": tA / 50, "rB": tB / 50, "p": p_tren(tA + tB, 100)}


# ==============================================================================
#  TRẠNG THÁI — sổ đề xuất + sổ bóng
# ==============================================================================

def nap_tt():
    if os.path.exists(FILE_TT):
        try:
            tt = json.load(open(FILE_TT, encoding="utf-8"))
            tt.setdefault("de_xuat", []); tt.setdefault("bong", [])
            return tt
        except Exception:
            pass
    return {"de_xuat": [], "bong": [], "cap_nhat": None}


def luu_tt(tt):
    os.makedirs(os.path.dirname(FILE_TT) or ".", exist_ok=True)
    tt["cap_nhat"] = datetime.now(VN).isoformat(timespec="seconds")
    json.dump(tt, open(FILE_TT, "w", encoding="utf-8"), ensure_ascii=False)


def _chuoi(so):
    return ",".join(f"{v:02d}" for v in so)


# ==============================================================================
#  TẦNG 3 + 4 — CHẤM ĐỀ XUẤT CŨ VÀ SỔ BÓNG
# ==============================================================================

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
    for loai in ("de_xuat", "bong"):
        for r in tt[loai]:
            if r.get("ket_qua") is not None or r["ngay"] >= hom_nay.isoformat():
                continue
            y = tra(r["stt"], r["giai"], r["ngay"])
            if y is None:
                continue
            r["ket_qua"] = {"so_ve": y, "trung": y in set(int(x) for x in r["so"].split(","))}
            if loai == "de_xuat":
                moi.append(r)
    return moi


def thanh_tich(tt):
    o = {}
    for r in tt["de_xuat"]:
        if r.get("ket_qua") is None:
            continue
        k = f"{r['stt']}|{r['giai']}"
        x = o.setdefault(k, {"stt": r["stt"], "dai": r["dai"], "giai": r["giai"], "n": 0, "trung": 0})
        x["n"] += 1; x["trung"] += int(r["ket_qua"]["trung"])
    for x in o.values():
        x["ty_le"] = x["trung"] / x["n"]
        x["p"] = p_tren(x["trung"], x["n"])
        x["xac_nhan"] = x["n"] >= N_XAC_NHAN and x["ty_le"] > HOA_VON and x["p"] < ALPHA
        x["bi_loai"] = x["n"] >= N_XAC_NHAN and x["ty_le"] < MOC
    dx = [r for r in tt["de_xuat"] if r.get("ket_qua")]
    bg = [r for r in tt["bong"] if r.get("ket_qua")]
    return {"o": o,
            "dx": (sum(r["ket_qua"]["trung"] for r in dx), len(dx)),
            "bong": (sum(r["ket_qua"]["trung"] for r in bg), len(bg))}


# ==============================================================================
#  CHẠY 1 NGÀY
# ==============================================================================

def chay(ngay=None, gui_mail=True):
    hom_nay = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    thu = E.THU_VN[hom_nay.weekday()]
    tt = nap_tt()

    print("=" * 80)
    print(f"  ENGINE HỢP THỂ  |  {thu} {hom_nay:%d.%m.%Y}")
    print(f"  7 mô hình · BMA tiến cứu τ={TAU} · cổng 2 khối + Bonferroni · sổ bóng")
    print("=" * 80)

    moi_cham = cham_cu(tt, hom_nay)
    tk = thanh_tich(tt)
    print(f"\n[TẦNG 3] Chấm {len(moi_cham)} đề xuất cũ vừa có kết quả")
    for r in moi_cham:
        kq = r["ket_qua"]
        print(f"     {r['ngay']} {r['dai']:<18}{TEN_GIAI[r['giai']]:<10} ra {kq['so_ve']:02d} "
              f"→ {'TRÚNG' if kq['trung'] else 'TRƯỢT'}")

    # ---- Dựng danh sách ô hôm nay ----
    dsach = E.dai_theo_ngay(hom_nay)
    lich = E.xay_lich()
    o_kiem, loi = [], []
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
        if not tg:
            loi.append((s, ten, "không có kỳ nào trước hôm nay"))
            continue
        for giai, bat in (("DB", CHAY_DB), ("G1", CHAY_G1), ("G8", CHAY_G8)):
            vi = _vi_tri(tg, giai) if bat else None
            if vi is None:
                continue
            o_kiem.append({"stt": s, "dai": ten, "giai": giai, "tg": tg, "ng": ng,
                           "mt": [int(ky[vi][-2:]) for ky in tg],
                           "pool": [[int(x[-2:]) for j, x in enumerate(ky) if j != vi] for ky in tg]})

    n_o = max(len(o_kiem), 1)
    nguong = ALPHA / n_o
    print(f"\n[TẦNG 0-2] {len(dsach)} đài · {len(o_kiem)} ô · ngưỡng Bonferroni p < {nguong:.5f}")

    ket = []
    for o in o_kiem:
        k = f"{o['stt']}|{o['giai']}"
        r = {"stt": o["stt"], "dai": o["dai"], "giai": o["giai"], "n": len(o["mt"])}
        if len(o["mt"]) < MIN_KY:
            r.update(ket_luan="THIẾU DỮ LIỆU", ly_do=f"chỉ {len(o['mt'])} kỳ, cần {MIN_KY}")
        else:
            ok, ly = kiem_du_lieu(o["tg"], o["ng"], o["mt"])
            if not ok:
                r.update(ket_luan="DỮ LIỆU LỖI", ly_do=ly)
            else:
                r.update(cong_tin_hieu(o["mt"], o["pool"]))
                f = khop(o["mt"], o["pool"])
                so = top64(f["P"])
                du_bao = (r["tA"] + r["tB"] + K_TIEN * MOC) / (100 + K_TIEN)
                r.update(so=so, du_bao=du_bao, w=f["w"], chan_doan=f["chan_doan"])
                qua = r["rA"] > HOA_VON and r["rB"] > HOA_VON and r["p"] < nguong
                if qua and tk["o"].get(k, {}).get("bi_loai"):
                    r["ket_luan"] = "BỊ LOẠI"
                    r["ly_do"] = "tiến cứu ≥30 kỳ dưới mốc 64% — kỷ luật ngừng đề xuất"
                else:
                    r["ket_luan"] = "TÍN HIỆU" if qua else "NHIỄU"
                if r["ket_luan"] == "TÍN HIỆU":
                    r["trang_thai"] = ("ĐÃ XÁC NHẬN" if tk["o"].get(k, {}).get("xac_nhan")
                                       else "ĐANG THEO DÕI")
        ket.append(r)
        dau = {"TÍN HIỆU": "✓", "NHIỄU": "·"}.get(r["ket_luan"], "✗")
        chi = (f"A {r['tA']}/50 · B {r['tB']}/50 · p {r['p']:.1e} · dự báo {r['du_bao']:.1%}"
               if "p" in r else r.get("ly_do", ""))
        print(f"     {dau} {r['dai']:<18}{TEN_GIAI[r['giai']]:<10}{r['ket_luan']:<14}{chi}")

    # ---- Ghi sổ TRƯỚC giờ quay: đề xuất + bóng cho MỌI ô có số ----
    hn = hom_nay.isoformat()
    co_dx = {(x["ngay"], x["stt"], x["giai"]) for x in tt["de_xuat"]}
    co_bg = {(x["ngay"], x["stt"], x["giai"]) for x in tt["bong"]}
    for r in ket:
        if "so" not in r:
            continue
        khoa = (hn, r["stt"], r["giai"])
        if khoa not in co_bg:
            tt["bong"].append({"ngay": hn, "stt": r["stt"], "giai": r["giai"],
                               "so": _chuoi(r["so"]), "ket_qua": None})
        if r["ket_luan"] == "TÍN HIỆU" and khoa not in co_dx:
            tt["de_xuat"].append({"ngay": hn, "stt": r["stt"], "dai": r["dai"],
                                  "giai": r["giai"], "so": _chuoi(r["so"]),
                                  "du_bao": r["du_bao"],
                                  "bang_chung": {"tA": r["tA"], "tB": r["tB"], "p": r["p"]},
                                  "ket_qua": None})
    luu_tt(tt)
    tk = thanh_tich(tt)

    de_xuat = [r for r in ket if r["ket_luan"] == "TÍN HIỆU"]
    print(f"\n  KẾT QUẢ: {len(de_xuat)} đề xuất / {len(ket)} ô")
    if not de_xuat:
        print("  → Hôm nay không đài × giải nào đủ ổn định. Không đề xuất.")
    T, N = tk["dx"]; Tb, Nb = tk["bong"]
    if N:
        print(f"  Tiến cứu đề xuất: {T}/{N} = {T/N:.1%}")
    if Nb:
        print(f"  Tiến cứu sổ bóng: {Tb}/{Nb} = {Tb/Nb:.1%}  (đối chứng tự nhiên)")

    bc = {"ngay": hom_nay, "thu": thu, "ket": ket, "de_xuat": de_xuat,
          "moi_cham": moi_cham, "tk": tk, "nguong": nguong, "loi": loi}
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
    ng, thu, ket, dx, tk = bc["ngay"], bc["thu"], bc["ket"], bc["de_xuat"], bc["tk"]
    dem = {k: sum(1 for r in ket if r["ket_luan"] == k)
           for k in ("NHIỄU", "BỊ LOẠI", "THIẾU DỮ LIỆU", "DỮ LIỆU LỖI")}
    h = [f'<div style="{css}max-width:800px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">Engine hợp thể — {thu} {ng:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:13px">{len(ket)} ô kiểm · '
             f'7 mô hình xác suất · BMA tiến cứu · ngưỡng Bonferroni p &lt; {bc["nguong"]:.5f}</p>')
    h.append('<table style="font-size:13px;border-collapse:collapse;margin:0 0 18px"><tr>'
             f'<td style="padding:5px 11px;background:#e8f5e9"><b>{len(dx)}</b> đề xuất</td>'
             f'<td style="padding:5px 11px;background:#eceff1"><b>{dem["NHIỄU"]}</b> nhiễu</td>'
             f'<td style="padding:5px 11px;background:#ffebee"><b>{dem["BỊ LOẠI"]}</b> bị loại</td>'
             f'<td style="padding:5px 11px;background:#fff3e0">'
             f'<b>{dem["THIẾU DỮ LIỆU"] + dem["DỮ LIỆU LỖI"]}</b> thiếu/lỗi dữ liệu</td>'
             '</tr></table>')

    if not dx:
        h.append('<div style="border-left:4px solid #546e7a;background:#fafafa;padding:12px 14px;'
                 'margin:0 0 20px;font-size:14px"><b>Hôm nay không đài × giải nào đủ ổn định. '
                 'Không đề xuất.</b><br><span style="font-size:12px;color:#607d8b">'
                 'Engine đang làm đúng việc: từ chối khi chỉ thấy nhiễu.</span></div>')
    for r in dx:
        xn = r["trang_thai"] == "ĐÃ XÁC NHẬN"
        mau = "#2e7d32" if xn else "#ef6c00"
        tw = sorted(zip(MO_HINH, r["w"]), key=lambda x: -x[1])[:3]
        h.append(f'<div style="margin:0 0 18px;border:1px solid #cfd8dc;border-radius:5px">'
                 f'<div style="padding:8px 12px;background:#263238;color:#fff">'
                 f'<b style="font-size:15px">{html.escape(r["dai"].upper())} — {TEN_GIAI[r["giai"]]}</b>'
                 f'<span style="float:right;background:{mau};padding:1px 8px;border-radius:3px;'
                 f'font-size:12px">{r["trang_thai"]}</span></div><div style="padding:10px 12px">'
                 f'<div style="font-size:12px;color:#455a64;margin-bottom:4px">'
                 f'Khối A <b>{r["tA"]}/50</b> ({r["rA"]:.0%}) · Khối B <b>{r["tB"]}/50</b> '
                 f'({r["rB"]:.0%}) · p gộp <b>{r["p"]:.2e}</b></div>'
                 f'<div style="font-size:12px;color:#455a64;margin-bottom:6px">'
                 f'Dự báo thận trọng <b>{r["du_bao"]:.1%}</b> · trọng số: '
                 + " · ".join(f"{t} {w:.0%}" for t, w in tw) + '</div>'
                 f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px;'
                 f'background:#e8f5e9;border-left:4px solid #2e7d32;padding:10px 12px;'
                 f'word-break:break-all;line-height:1.9">{_chuoi(r["so"])}</div>')
        if not xn:
            h.append(f'<div style="font-size:12px;color:#ef6c00;margin-top:6px">⚠ Chưa xác nhận '
                     f'tiến cứu — cần {N_XAC_NHAN} kỳ tương lai. Nên cược nhỏ.</div>')
        h.append('</div></div>')

    # ---- Tiến cứu: đề xuất vs bóng ----
    T, N = tk["dx"]; Tb, Nb = tk["bong"]
    h.append('<div style="font-size:15px;font-weight:700;margin:22px 0 6px">THÀNH TÍCH TIẾN CỨU</div>')
    if not N and not Nb:
        h.append('<p style="font-size:13px;color:#607d8b">Chưa có kết quả nào được chấm. '
                 'Sổ bắt đầu tích luỹ từ ngày mai.</p>')
    else:
        h.append('<table style="font-size:13px;border-collapse:collapse;margin:0 0 8px">'
                 '<tr style="background:#eceff1"><th style="padding:5px 10px;text-align:left">Sổ</th>'
                 '<th style="padding:5px 10px">Đúng/Tổng</th><th style="padding:5px 10px">Tỷ lệ</th>'
                 '<th style="padding:5px 10px">KTC 95%</th></tr>')
        for ten, (a, b) in (("Đề xuất (qua cổng tín hiệu)", (T, N)),
                            ("Bóng — mọi ô, đối chứng tự nhiên", (Tb, Nb))):
            if b:
                lo, hi = wilson(a, b)
                h.append(f'<tr><td style="padding:5px 10px">{ten}</td>'
                         f'<td style="padding:5px 10px;text-align:center">{a}/{b}</td>'
                         f'<td style="padding:5px 10px;text-align:center"><b>{a/b:.1%}</b></td>'
                         f'<td style="padding:5px 10px;text-align:center">[{lo:.1%}, {hi:.1%}]</td></tr>')
        h.append(f'</table><p style="font-size:12px;color:#607d8b;margin:0 0 8px">Mốc 64% · hoà vốn '
                 f'{HOA_VON:.1%}. Nếu dòng đề xuất KHÔNG cao hơn dòng bóng, cổng tín hiệu không '
                 f'thêm giá trị.</p>')
        if tk["o"]:
            h.append('<table style="font-size:12px;border-collapse:collapse">'
                     '<tr style="background:#eceff1"><th style="padding:4px 9px;text-align:left">Đài</th>'
                     '<th style="padding:4px 9px">Giải</th><th style="padding:4px 9px">Đúng/Tổng</th>'
                     '<th style="padding:4px 9px">Tỷ lệ</th><th style="padding:4px 9px">Trạng thái</th></tr>')
            for x in sorted(tk["o"].values(), key=lambda z: -z["n"]):
                tt_ = ("ĐÃ XÁC NHẬN" if x["xac_nhan"] else "BỊ LOẠI" if x["bi_loai"]
                       else f"theo dõi {x['n']}/{N_XAC_NHAN}")
                h.append(f'<tr><td style="padding:4px 9px">{html.escape(x["dai"])}</td>'
                         f'<td style="padding:4px 9px">{TEN_GIAI[x["giai"]]}</td>'
                         f'<td style="padding:4px 9px;text-align:center">{x["trung"]}/{x["n"]}</td>'
                         f'<td style="padding:4px 9px;text-align:center">{x["ty_le"]:.0%}</td>'
                         f'<td style="padding:4px 9px">{tt_}</td></tr>')
            h.append('</table>')

    # ---- Chi tiết mọi ô ----
    h.append('<details style="margin:18px 0 0"><summary style="font-size:13px;color:#546e7a;'
             'cursor:pointer">Chi tiết mọi ô đã kiểm</summary>'
             '<table style="font-size:11px;border-collapse:collapse;margin-top:5px">'
             '<tr style="background:#eceff1"><th style="padding:3px 7px;text-align:left">Đài</th>'
             '<th style="padding:3px 7px">Giải</th><th style="padding:3px 7px">Kết luận</th>'
             '<th style="padding:3px 7px">A · B · p</th><th style="padding:3px 7px">Dự báo</th>'
             '<th style="padding:3px 7px">Trọng số ĐỀU</th></tr>')
    for r in ket:
        abp = (f'{r["tA"]}/50 · {r["tB"]}/50 · {r["p"]:.1e}' if "p" in r
               else html.escape(r.get("ly_do", "")))
        h.append(f'<tr><td style="padding:3px 7px">{html.escape(r["dai"])}</td>'
                 f'<td style="padding:3px 7px">{TEN_GIAI[r["giai"]]}</td>'
                 f'<td style="padding:3px 7px"><b>{r["ket_luan"]}</b></td>'
                 f'<td style="padding:3px 7px;color:#607d8b">{abp}</td>'
                 f'<td style="padding:3px 7px;text-align:center">'
                 f'{(format(r["du_bao"], ".1%") if "du_bao" in r else "—")}</td>'
                 f'<td style="padding:3px 7px;text-align:center">'
                 f'{(format(r["w"][0], ".0%") if "w" in r else "—")}</td></tr>')
    h.append('</table></details>')

    if bc["loi"]:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;padding:10px 13px;'
                 'margin:16px 0;font-size:13px"><b>Đài lỗi:</b><ul>')
        for s_, t_, e_ in bc["loi"]:
            h.append(f'<li>[{s_}] {html.escape(t_)}: {html.escape(str(e_))}</li>')
        h.append('</ul></div>')

    h.append('<hr style="margin:22px 0 10px;border:0;border-top:1px solid #ddd">'
             '<p style="font-size:12px;color:#888;line-height:1.6">'
             '<b>Mềm mỏng:</b> 7 mô hình được gộp mềm bằng BMA tiến cứu có nhiệt độ — không mô '
             'hình nào thắng tuyệt đối. <b>Dự báo thận trọng</b> = tỷ lệ trúng ngoài mẫu co '
             'một nửa về 64%.<br><b>Kỷ luật:</b> kiểm tra dữ liệu · 2 khối test rời nhau · '
             'Bonferroni · xác nhận bằng 30 kỳ tương lai · loại ô thất bại tiến cứu · sổ bóng '
             'đối chứng.<br>Máy quay công bằng thì 64 con bất kỳ trúng đúng 64% — engine không '
             'phá được định lý đó, chỉ im lặng đúng lúc và bắt được độ lệch mạnh nếu có.</p></div>')
    return "".join(h)


def _text(bc):
    t = [f"ENGINE HỢP THỂ — {bc['thu']} {bc['ngay']:%d.%m.%Y}", ""]
    if not bc["de_xuat"]:
        t.append("Hôm nay không đài × giải nào đủ ổn định. Không đề xuất.")
    for r in bc["de_xuat"]:
        t.append(f"{r['dai'].upper()} — {TEN_GIAI[r['giai']]} [{r['trang_thai']}]")
        t.append(f"  A {r['tA']}/50 · B {r['tB']}/50 · p {r['p']:.2e} · dự báo {r['du_bao']:.1%}")
        t.append("  " + _chuoi(r["so"]))
        t.append("")
    T, N = bc["tk"]["dx"]; Tb, Nb = bc["tk"]["bong"]
    if N:
        t.append(f"Tiến cứu đề xuất: {T}/{N} = {T/N:.1%}")
    if Nb:
        t.append(f"Tiến cứu sổ bóng: {Tb}/{Nb} = {Tb/Nb:.1%}")
    return "\n".join(t)


def gui_email(bc):
    mk = E._lay_mat_khau()
    n = len(bc["de_xuat"])
    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"[HỢP THỂ] {bc['thu']} {bc['ngay']:%d.%m.%Y} — "
                      + (f"{n} đề xuất" if n else "không có đề xuất"))
    msg["From"] = formataddr(("XSMN Hợp Thể", EMAIL_GUI))
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
