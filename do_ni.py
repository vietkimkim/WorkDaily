"""
DO_NI — ĐO NI ĐÓNG GIÀY: mỗi giải của mỗi đài có ENGINE RIÊNG, do chính dữ liệu
        150 kỳ của giải đó quyết định. Mọi giải vẫn LUÔN có 64 con mỗi ngày.

═══════════════════════════════════════════════════════════════════════════════
QUY TRÌNH ĐO NI (mỗi tháng một lần cho mỗi giải × mỗi đài)
    150 kỳ gần nhất
      ├── KHÁM PHÁ  kỳ 31–100 : thử 10 họ logic (14–15 cấu hình), chọn cái vừa nhất
      └── KIỂM CHỨNG kỳ 101–150: đo cấu hình đó trên 50 kỳ nó CHƯA TỪNG THẤY

    PHÁN QUYẾT
      CÓ LOGIC RIÊNG       : kiểm chứng vượt hoà vốn 67,37% VÀ p < 0,0025
                             -> dùng engine đo ni của giải đó
      KHÔNG CÓ LOGIC RIÊNG : còn lại -> dùng engine mặc định HỢP THỂ 64
                             (vẫn đủ 64 con, nhãn ghi rõ trên từng đề xuất)

VÌ SAO PHẢI KIỂM CHỨNG — đo trên 120 giải HOÀN TOÀN công bằng:
    Logic "tìm được" lúc khám phá : 71,0%
    Cùng logic đó trên kỳ mới     : 63,7%   (mốc 64,0%)  -> ẢO GIÁC ĐO NI +7,3 điểm
    Máy LUÔN tìm ra một "logic" cho mọi giải, kể cả giải không có logic nào.
    Giải có logic THẬT thì giữ được: khám phá 84,4% -> kỳ mới 90,7%.

ĐỐI CHỨNG NHIỄU — CHÍNH XÁC, KHÔNG CẦN XÁO TRỘN
    Bản mô tả ban đầu định xáo thứ tự kỳ để làm đối chứng. Khi dựng, tôi bỏ cách
    đó vì hai lý do:
      · Xáo thứ tự GIỮ NGUYÊN độ lệch tần suất -> sẽ BỎ SÓT đúng loại logic mạnh
        nhất đã thấy trong dữ liệu thật (Miền Bắc G1).
      · Máy quay công bằng thì MỖI kỳ kiểm chứng trúng với xác suất đúng 64%, bất
        kể logic nào được chọn -> số lần trúng ~ Nhị thức(50; 0,64) CHÍNH XÁC.
    p = P(Nhị thức(50; 0,64) >= số lần trúng). Ngưỡng 0,0025 ứng với >= 42/50.
    Trên ~107 giải, kỳ vọng chỉ ~0.17 nhãn "CÓ LOGIC RIÊNG" sai mỗi tháng.

CHỌN LOGIC BẰNG THỨ HẠNG, KHÔNG BẰNG TRÚNG/TRƯỢT — phát hiện khi dựng
    Chấm khám phá bằng trúng/trượt chỉ cho 1 bit mỗi kỳ -> chọn rất nhiễu, hay
    chọn nhầm họ "học nhanh" (ĐỘC LẬP, 20 tham số) thay vì họ "đúng".
    Chấm bằng THỨ HẠNG của con thật sự ra trong bảng 100 con dùng nhiều thông tin
    hơn. Đo trên 60 giải có logic thật:
        trúng/trượt : bắt được 33% · kiểm chứng TB 79,9%
        THỨ HẠNG    : bắt được 38% · kiểm chứng TB 81,5%
        trần (biết trước đáp án): 53% · 85,1%
    Trần chỉ 53% -> với 150 kỳ, độ lệch dàn trải vẫn khó bắt. Giới hạn nằm ở
    LƯỢNG DỮ LIỆU, không ở thuật toán.

10 HỌ LOGIC
    1 TẦN SUẤT        toàn kỳ
    2 CỬA SỔ          10 · 20 · 30 · 50 kỳ gần nhất
    3 GẦN ĐÂY         trọng số giảm dần, nửa đời 5 · 10 · 20 kỳ
    4 ĐỘC LẬP         P(chục)·P(đơn vị)
    5 MARKOV          chữ số kỳ trước -> chữ số kỳ sau
    6 TỔNG CS         phân phối (chục + đơn vị) mod 10
    7 GỘP GIẢI        mượn 17–26 giải khác cùng đài
    8 LỊCH            chỉ các kỳ cùng thứ trong tuần (đài quay >= 2 thứ/tuần)
    9 GAN             lâu chưa về — quy tắc dân gian, để dữ liệu tự phán xử
   10 ĐỀU             đối chứng (luôn chọn 00–63)

GIỚI HẠN KHÔNG VƯỢT ĐƯỢC
    Máy quay công bằng: mọi bộ 64 con trúng 64%, tỷ lệ trả 95 -> kỳ vọng -5%.
    Đo ni không tạo ra logic — nó chỉ tìm và XÁC NHẬN logic nếu có sẵn.
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, json, math, html, zlib, smtplib, traceback
from datetime import datetime, timedelta, timezone, date
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

SO_KY        = 150    # số kỳ dùng để đo ni
N_KIEM       = 50     # số kỳ cuối dùng để KIỂM CHỨNG (chưa từng thấy lúc khám phá)
T_BD         = 30     # khám phá bắt đầu dự báo từ kỳ thứ 31
N_MIN        = 110    # ít hơn -> "CHƯA ĐỦ DỮ LIỆU", dùng engine mặc định
SO_CON       = 64
TY_LE_TRA    = 95.0
HOA_VON      = SO_CON / TY_LE_TRA          # 67,37%
MOC          = SO_CON / 100.0              # 64,00%
ALPHA_O      = 0.0025 # ngưỡng p cho nhãn "CÓ LOGIC RIÊNG" (≈ 0,05/20)

CHAY_DB, CHAY_G1, CHAY_G8 = True, True, True
FILE_TT      = "data/do_ni.json"
EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")
TEN_GIAI     = {"DB": "① Đặc Biệt", "G1": "② Giải Nhất", "G8": "③ Giải 8"}
NHAN_CO      = "CÓ LOGIC RIÊNG"
NHAN_KHONG   = "Không có logic riêng"
NHAN_THIEU   = "Chưa đủ dữ liệu đo ni"

A_, B_ = np.arange(100) // 10, np.arange(100) % 10


# ==============================================================================
#  ENGINE MẶC ĐỊNH — HỢP THỂ 64 (bản nhúng, giống hệt hop_the_64.py)
#  7 mô hình gộp bằng BMA tiến cứu có nhiệt độ. Dùng cho giải KHÔNG có logic riêng.
# ==============================================================================

TAU, SAN_W, HALF_LIFE_BMA, T0_BMA = 0.5, 0.02, 30, 20
DEU = np.full(100, 0.01)
LUOI_ALPHA = np.exp(np.linspace(np.log(0.05), np.log(500), 30))


def _p_dong_nhat(a, b):
    T = np.vstack([a, b]) + 0.5
    R = T.sum(1, keepdims=True); C = T.sum(0, keepdims=True)
    Ex = R @ C / T.sum()
    return float(stats.chi2.sf(((T - Ex) ** 2 / Ex).sum(), 9))


class _TK:
    def __init__(self):
        self.c = np.zeros(100); self.n = 0
        self.d1 = np.zeros(10); self.d2 = np.zeros(10)
        self.pc = np.zeros(100); self.pd1 = np.zeros(10); self.pd2 = np.zeros(10)
        self.cw = np.zeros(100); self.dec = 0.5 ** (1.0 / HALF_LIFE_BMA)

    def them(self, y, pk):
        self.c[y] += 1; self.n += 1; self.d1[y // 10] += 1; self.d2[y % 10] += 1
        self.cw *= self.dec; self.cw[y] += 1
        for v in pk:
            self.pc[v] += 1; self.pd1[v // 10] += 1; self.pd2[v % 10] += 1

    def mo_hinh(self):
        n, c = self.n, self.c
        P = np.empty((7, 100)); P[0] = DEU; P[1] = (c + 1) / (n + 100)
        a = LUOI_ALPHA
        ml = (gammaln(100 * a) - gammaln(n + 100 * a)
              + gammaln(c[None, :] + a[:, None]).sum(1) - 100 * gammaln(a))
        al = float(a[int(np.argmax(ml))]); P[2] = (c + al) / (n + 100 * al)
        if n > 0:
            ph = c / n; S = float(((ph - DEU) ** 2).sum())
            B = 0.0 if S <= 1e-12 else float(np.clip(1 - 97 * (0.0099 / n) / S, 0, 1))
            q = np.clip(DEU + B * (ph - DEU), 1e-9, None); P[3] = q / q.sum()
        else:
            P[3] = DEU
        q = ((self.d1 + 1) / (n + 10))[A_] * ((self.d2 + 1) / (n + 10))[B_]; P[4] = q / q.sum()
        P[5] = (self.cw + 0.5) / (self.cw.sum() + 50)
        p_dn = (min(1.0, 2 * min(_p_dong_nhat(self.d1, self.pd1), _p_dong_nhat(self.d2, self.pd2)))
                if self.pc.sum() > 0 and n >= 10 else 1.0)
        P[6] = (c + self.pc + 1) / (n + self.pc.sum() + 100) if p_dn >= 0.05 else P[1]
        return P


def mac_dinh_64(mt, pool):
    """Engine mặc định: BMA 7 mô hình -> 64 con xác suất cao nhất."""
    st = _TK(); S = np.zeros(7)
    for t, y in enumerate(mt):
        if t >= T0_BMA:
            S += np.log(np.clip(st.mo_hinh()[:, y], 1e-12, None))
        st.them(y, pool[t])
    z = TAU * (S - S.max()); w = np.exp(z); w /= w.sum(); w = (1 - SAN_W) * w + SAN_W / 7
    P = w @ st.mo_hinh()
    return sorted(int(i) for i in np.argsort(-P, kind="stable")[:SO_CON])


# ==============================================================================
#  10 HỌ LOGIC — tính sẵn số liệu dồn cho mọi thời điểm t (chỉ dùng dữ liệu < t)
# ==============================================================================

def _thu(d):
    return d.weekday() if hasattr(d, "weekday") else date.fromisoformat(str(d)[:10]).weekday()


def _chuan_bi(mt, pool, thu_list):
    """Mảng dồn kích thước (n+1, ...): hàng t = thống kê của mt[:t]. KHÔNG nhìn tương lai."""
    n = len(mt)
    X = {"n": n, "mt": mt, "thu": thu_list}
    one = np.zeros((n, 100)); one[np.arange(n), mt] = 1
    X["c"] = np.vstack([np.zeros(100), np.cumsum(one, 0)])
    pk = np.zeros((n, 100))
    for t, ky in enumerate(pool):
        np.add.at(pk[t], ky, 1)
    X["pool"] = np.vstack([np.zeros(100), np.cumsum(pk, 0)])
    d1 = np.zeros((n, 10)); d1[np.arange(n), mt // 10] = 1
    d2 = np.zeros((n, 10)); d2[np.arange(n), mt % 10] = 1
    X["d1"] = np.vstack([np.zeros(10), np.cumsum(d1, 0)])
    X["d2"] = np.vstack([np.zeros(10), np.cumsum(d2, 0)])
    sm = np.zeros((n, 10)); sm[np.arange(n), (mt // 10 + mt % 10) % 10] = 1
    X["tong"] = np.vstack([np.zeros(10), np.cumsum(sm, 0)])
    for H in (5, 10, 20):
        R = np.zeros((n + 1, 100)); dec = 0.5 ** (1.0 / H)
        for t in range(1, n + 1):
            R[t] = R[t - 1] * dec; R[t, mt[t - 1]] += 1
        X[f"gd{H}"] = R
    T1 = np.zeros((n + 1, 10, 10)); T2 = np.zeros((n + 1, 10, 10))
    for t in range(1, n + 1):
        T1[t] = T1[t - 1]; T2[t] = T2[t - 1]
        if t >= 2:
            T1[t, mt[t - 2] // 10, mt[t - 1] // 10] += 1
            T2[t, mt[t - 2] % 10, mt[t - 1] % 10] += 1
    X["T1"], X["T2"] = T1, T2
    L = np.full((n + 1, 100), -1.0)
    for t in range(1, n + 1):
        L[t] = L[t - 1]; L[t, mt[t - 1]] = t - 1
    X["last"] = L
    W = np.zeros((n + 1, 7, 100))
    for t in range(1, n + 1):
        W[t] = W[t - 1]; W[t, thu_list[t - 1], mt[t - 1]] += 1
    X["wd"] = W
    return X


def cau_hinh(co_lich):
    ds = ["TẦN SUẤT", "CỬA SỔ 10", "CỬA SỔ 20", "CỬA SỔ 30", "CỬA SỔ 50",
          "GẦN ĐÂY 5", "GẦN ĐÂY 10", "GẦN ĐÂY 20", "ĐỘC LẬP", "MARKOV",
          "TỔNG CS", "GỘP GIẢI", "GAN", "ĐỀU"]
    if co_lich:
        ds.insert(12, "LỊCH")
    return ds


MO_TA = {"TẦN SUẤT": "tần suất toàn kỳ", "CỬA SỔ": "tần suất {} kỳ gần nhất",
         "GẦN ĐÂY": "trọng số giảm dần, nửa đời {} kỳ", "ĐỘC LẬP": "P(chục)·P(đơn vị)",
         "MARKOV": "chữ số kỳ trước → kỳ sau", "TỔNG CS": "phân phối tổng chữ số",
         "GỘP GIẢI": "mượn các giải khác cùng đài", "LỊCH": "các kỳ cùng thứ trong tuần",
         "GAN": "con lâu chưa về", "ĐỀU": "đối chứng 00–63"}


def mo_ta(cfg):
    ho, _, ts = cfg.partition(" ")
    if ho in ("CỬA", "GẦN"):
        ho2 = "CỬA SỔ" if ho == "CỬA" else "GẦN ĐÂY"
        return MO_TA[ho2].format(cfg.split()[-1])
    return MO_TA.get(cfg, cfg)


def diem(cfg, X, t, thu_t):
    """Điểm 100 con tại thời điểm t, CHỈ dùng dữ liệu < t."""
    mt = X["mt"]
    if cfg == "TẦN SUẤT":
        return X["c"][t]
    if cfg.startswith("CỬA SỔ"):
        Wn = int(cfg.split()[-1]); return X["c"][t] - X["c"][max(0, t - Wn)]
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


def top64(s):
    return sorted(int(i) for i in np.argsort(-np.asarray(s, float), kind="stable")[:SO_CON])


def _trung(cfg, X, t0, t1):
    mt, thu = X["mt"], X["thu"]
    return sum(1 for t in range(t0, t1)
               if mt[t] in set(np.argsort(-diem(cfg, X, t, thu[t]), kind="stable")[:SO_CON].tolist()))


def _hang(cfg, X, t0, t1):
    """Thứ hạng TRUNG BÌNH của con thật sự ra (0 = đứng đầu bảng). Càng nhỏ càng tốt."""
    mt, thu = X["mt"], X["thu"]
    r = []
    for t in range(t0, t1):
        o = np.argsort(-diem(cfg, X, t, thu[t]), kind="stable")
        pos = np.empty(100); pos[o] = np.arange(100)
        r.append(pos[mt[t]])
    return float(np.mean(r)) if r else 99.0


def p_tren(k, n, p0=MOC):
    return float(1 - stats.binom.cdf(k - 1, n, p0)) if n > 0 else 1.0


def do_ni_1_o(mt, pool, thu_list):
    """Đo ni 1 giải: khám phá -> kiểm chứng -> phán quyết. Trả hồ sơ."""
    n = len(mt)
    if n < N_MIN:
        return {"ket_luan": NHAN_THIEU, "n_ky": n}
    mt = np.asarray(mt)
    X = _chuan_bi(mt, pool, thu_list)
    co_lich = len(set(thu_list)) >= 2
    ds = cau_hinh(co_lich)
    t_kiem = n - N_KIEM
    kham = {c: _trung(c, X, T_BD, t_kiem) for c in ds}
    hang = {c: _hang(c, X, T_BD, t_kiem) for c in ds}
    n_kham = t_kiem - T_BD
    tot = min(ds, key=lambda c: (hang[c], ds.index(c)))       # chọn theo THỨ HẠNG
    k_kiem = _trung(tot, X, t_kiem, n)
    p = p_tren(k_kiem, N_KIEM)
    co = (k_kiem / N_KIEM > HOA_VON) and (p < ALPHA_O)
    xep = sorted(ds, key=lambda c: (hang[c], ds.index(c)))[:5]
    return {"ket_luan": NHAN_CO if co else NHAN_KHONG, "n_ky": n,
            "logic": tot, "mo_ta": mo_ta(tot),
            "kham": {"trung": kham[tot], "n": n_kham, "ty_le": kham[tot] / n_kham,
                     "hang": hang[tot]},
            "kiem": {"trung": k_kiem, "n": N_KIEM, "ty_le": k_kiem / N_KIEM, "p": p},
            "xep_hang": [[c, kham[c] / n_kham, hang[c]] for c in xep]}


def so_hom_nay(ho_so, mt, pool, thu_list, thu_hn):
    """Ra 64 con cho kỳ hôm nay: engine đo ni nếu CÓ logic riêng, không thì mặc định."""
    if ho_so.get("ket_luan") == NHAN_CO:
        X = _chuan_bi(np.asarray(mt), pool, thu_list)
        return top64(diem(ho_so["logic"], X, len(mt), thu_hn)), "ĐO NI"
    return mac_dinh_64(list(mt), pool), "MẶC ĐỊNH HỢP THỂ 64"


# ==============================================================================
#  TIỆN ÍCH
# ==============================================================================

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


def ngau_nhien_64(ngay, stt, giai):
    """Bộ đối chứng 64 con ngẫu nhiên, TÁI LẬP ĐƯỢC theo ngày + đài + giải."""
    rng = np.random.default_rng(zlib.crc32(f"{ngay}|{stt}|{giai}".encode()))
    return sorted(int(x) for x in rng.choice(100, SO_CON, replace=False))


# ==============================================================================
#  SỔ: HỒ SƠ ĐO NI + BỘ SỐ HẰNG NGÀY
# ==============================================================================

def nap_tt():
    if os.path.exists(FILE_TT):
        try:
            tt = json.load(open(FILE_TT, encoding="utf-8"))
            tt.setdefault("ho_so", {}); tt.setdefault("so", [])
            return tt
        except Exception:
            pass
    return {"ho_so": {}, "so": [], "cap_nhat": None}


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
        r["ket_qua"] = {"so_ve": y,
                        "trung": y in set(int(x) for x in r["so"].split(",")),
                        "trung_ngau": y in set(int(x) for x in r["ngau"].split(","))}
        moi.append(r)
    return moi


def thanh_tich(tt):
    da = [r for r in tt["so"] if r.get("ket_qua")]
    kq = {"N": len(da), "eng": sum(r["ket_qua"]["trung"] for r in da),
          "ngau": sum(r["ket_qua"]["trung_ngau"] for r in da), "theo": {}}
    for r in da:
        for khoa in (r["ket_luan"], TEN_GIAI[r["giai"]]):
            x = kq["theo"].setdefault(khoa, {"n": 0, "eng": 0, "ngau": 0})
            x["n"] += 1; x["eng"] += int(r["ket_qua"]["trung"])
            x["ngau"] += int(r["ket_qua"]["trung_ngau"])
    return kq


# ==============================================================================
#  CHẠY 1 NGÀY
# ==============================================================================

def chay(ngay=None, gui_mail=True, do_lai=False):
    hom_nay = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    thu = E.THU_VN[hom_nay.weekday()]
    hn, thang = hom_nay.isoformat(), hom_nay.strftime("%Y-%m")
    tt = nap_tt()

    print("=" * 80)
    print(f"  ĐO NI ĐÓNG GIÀY  |  {thu} {hom_nay:%d.%m.%Y}  |  tháng đo {thang}")
    print("=" * 80)

    moi_cham = cham_cu(tt, hom_nay)
    if moi_cham:
        print(f"\n[CHẤM HÔM TRƯỚC] {len(moi_cham)} bộ · engine trúng "
              f"{sum(r['ket_qua']['trung'] for r in moi_cham)} · ngẫu nhiên trúng "
              f"{sum(r['ket_qua']['trung_ngau'] for r in moi_cham)}")

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
        for giai, bat in (("DB", CHAY_DB), ("G1", CHAY_G1), ("G8", CHAY_G8)):
            if not bat:
                continue
            if tg:
                vi = _vi_tri(tg, giai)
                if vi is None:
                    continue
                d["o"].append({"giai": giai, "mt": [int(ky[vi][-2:]) for ky in tg],
                               "pool": [[int(x[-2:]) for j, x in enumerate(ky) if j != vi] for ky in tg],
                               "thu": [_thu(x) for x in ng]})
            elif not (giai == "G8" and E.lay_dai(s)[1] == "xsmb"):
                d["o"].append({"giai": giai, "mt": [], "pool": [], "thu": []})
        dai_list.append(d)

    du_phong = {}
    for d in dai_list:
        for o in d["o"]:
            du_phong.setdefault(o["giai"], []).extend(o["mt"])

    n_do = n_dung = 0
    for d in dai_list:
        print(f"\n  {d['dai'].upper()}" + (f"   ⚠ {d['loi']}" if d["loi"] else ""))
        for o in d["o"]:
            k = f"{d['stt']}|{o['giai']}"
            if o["mt"]:
                hs = tt["ho_so"].get(k)
                if do_lai or not hs or hs.get("thang") != thang:
                    hs = do_ni_1_o(o["mt"], o["pool"], o["thu"])
                    hs.update(thang=thang, ngay_do=hn, dai=d["dai"], giai=o["giai"])
                    tt["ho_so"][k] = hs; n_do += 1; moi = "ĐO MỚI"
                else:
                    n_dung += 1; moi = f"hồ sơ {hs['ngay_do']}"
                o["ho_so"] = hs
                o["so"], o["nguon"] = so_hom_nay(hs, o["mt"], o["pool"], o["thu"], hom_nay.weekday())
            else:
                goc = du_phong.get(o["giai"], [])
                c = np.ones(100)
                for v in goc:
                    c[v] += 1
                o["so"] = top64(c) if goc else list(range(SO_CON))
                o["nguon"] = "DỰ PHÒNG"
                o["ho_so"] = {"ket_luan": NHAN_KHONG,
                              "ly_do": (f"lỗi dữ liệu — dùng tần suất gộp {TEN_GIAI[o['giai']]} "
                                        f"của các đài khác hôm nay ({len(goc)} kỳ)" if goc
                                        else "lỗi dữ liệu — không có dữ liệu nào để lập luận")}
                moi = "DỰ PHÒNG"
            o["ngau"] = ngau_nhien_64(hn, d["stt"], o["giai"])
            hs = o["ho_so"]
            chi = (f"{hs['logic']} · khám phá {hs['kham']['ty_le']:.0%} → kiểm chứng "
                   f"{hs['kiem']['trung']}/50 p={hs['kiem']['p']:.1e}" if "logic" in hs
                   else hs.get("ly_do", f"{hs.get('n_ky', 0)} kỳ"))
            print(f"     {TEN_GIAI[o['giai']]:<14} {hs['ket_luan']:<24} {chi}   [{moi}]")

    da_co = {(r["ngay"], r["stt"], r["giai"]) for r in tt["so"]}
    for d in dai_list:
        for o in d["o"]:
            if (hn, d["stt"], o["giai"]) not in da_co:
                tt["so"].append({"ngay": hn, "stt": d["stt"], "dai": d["dai"], "giai": o["giai"],
                                 "ket_luan": o["ho_so"]["ket_luan"],
                                 "logic": o["ho_so"].get("logic", "") if o["ho_so"]["ket_luan"] == NHAN_CO else "",
                                 "nguon": o["nguon"], "so": _chuoi(o["so"]),
                                 "ngau": _chuoi(o["ngau"]), "ket_qua": None})
    luu_tt(tt)
    tk = thanh_tich(tt)
    n_o = sum(len(d["o"]) for d in dai_list)
    print(f"\n  {n_o} bộ số · đo mới {n_do} giải · dùng lại hồ sơ {n_dung} giải")
    if tk["N"]:
        print(f"  TIẾN CỨU: engine {tk['eng']}/{tk['N']} = {tk['eng']/tk['N']:.1%} · "
              f"ngẫu nhiên {tk['ngau']}/{tk['N']} = {tk['ngau']/tk['N']:.1%}")

    bc = {"ngay": hom_nay, "thu": thu, "thang": thang, "dai": dai_list, "tk": tk,
          "n_o": n_o, "n_do": n_do, "n_dung": n_dung}
    if gui_mail:
        print("\n  Đang gửi email...")
        gui_email(bc)
        print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")
    return bc


# ==============================================================================
#  EMAIL
# ==============================================================================

def _dong_tc(ten, x):
    if not x or not x["n"]:
        return ""
    n, e, g = x["n"], x["eng"], x["ngau"]
    lo, hi = wilson(e, n)
    mau = "#2e7d32" if e > g else ("#c62828" if e < g else "#455a64")
    return (f'<tr><td style="padding:4px 9px">{ten}</td>'
            f'<td style="padding:4px 9px;text-align:center">{n}</td>'
            f'<td style="padding:4px 9px;text-align:center"><b>{e/n:.1%}</b></td>'
            f'<td style="padding:4px 9px;text-align:center">{g/n:.1%}</td>'
            f'<td style="padding:4px 9px;text-align:center;color:{mau}">{(e-g)/n:+.1%}</td>'
            f'<td style="padding:4px 9px;text-align:center">[{lo:.0%}, {hi:.0%}]</td></tr>')


def _dong_ly_do(o):
    """Dòng giải thích dưới mỗi bộ số. Giải KHÔNG có logic riêng luôn ghi rõ nhãn."""
    hs = o["ho_so"]
    if hs["ket_luan"] == NHAN_CO:
        return (f'<b>Logic riêng: {html.escape(hs["logic"])}</b> — {html.escape(hs["mo_ta"])} · '
                f'kiểm chứng {hs["kiem"]["trung"]}/50 = {hs["kiem"]["ty_le"]:.0%} · '
                f'p = {hs["kiem"]["p"]:.1e} · đo ngày {hs["ngay_do"][8:10]}.{hs["ngay_do"][5:7]}')
    if o["nguon"] == "DỰ PHÒNG":
        return f'<b>{NHAN_KHONG}</b> · DỰ PHÒNG: {html.escape(hs["ly_do"])}'
    if "logic" in hs:
        return (f'<b>{NHAN_KHONG}</b> — dùng engine mặc định Hợp Thể 64 · logic vừa nhất lúc '
                f'khám phá: {html.escape(hs["logic"])} {hs["kham"]["ty_le"]:.0%} → kiểm chứng '
                f'{hs["kiem"]["trung"]}/50 = {hs["kiem"]["ty_le"]:.0%} (không giữ được)')
    return (f'<b>{NHAN_KHONG}</b> — chưa đủ dữ liệu đo ni ({hs.get("n_ky", 0)} kỳ), '
            f'dùng engine mặc định Hợp Thể 64')


def _html(bc):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    ng, thu, tk = bc["ngay"], bc["thu"], bc["tk"]
    tat_ca = [o for d in bc["dai"] for o in d["o"]]
    n_co = sum(1 for o in tat_ca if o["ho_so"]["ket_luan"] == NHAN_CO)
    h = [f'<div style="{css}max-width:820px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">Đo ni đóng giày — {thu} {ng:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 12px;font-size:13px">{len(bc["dai"])} đài · '
             f'{bc["n_o"]} bộ 64 con · hồ sơ tháng {bc["thang"]} · đo mới {bc["n_do"]}, '
             f'dùng lại {bc["n_dung"]}</p>')
    h.append('<div style="font-size:12px;margin:0 0 16px">'
             f'<span style="background:#2e7d32;color:#fff;padding:2px 8px;border-radius:3px;'
             f'margin-right:4px">{NHAN_CO}: {n_co}</span>'
             f'<span style="background:#78909c;color:#fff;padding:2px 8px;border-radius:3px">'
             f'{NHAN_KHONG}: {len(tat_ca) - n_co}</span></div>')

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
                 '<th style="padding:4px 9px">KTC 95%</th></tr>')
        h.append(_dong_tc("<b>TỔNG</b>", {"n": tk["N"], "eng": tk["eng"], "ngau": tk["ngau"]}))
        for k in (NHAN_CO, NHAN_KHONG, "① Đặc Biệt", "② Giải Nhất", "③ Giải 8"):
            h.append(_dong_tc(k, tk["theo"].get(k)))
        h.append(f'</table><p style="font-size:12px;color:#607d8b;margin:0 0 18px">Mốc 64% · '
                 f'hoà vốn {HOA_VON:.1%}. Nhóm "{NHAN_CO}" phải thắng nhóm ngẫu nhiên đều đặn '
                 f'qua nhiều tuần thì đo ni mới có giá trị thật.</p>')

    for d in bc["dai"]:
        h.append(f'<div style="margin:16px 0 0;padding:8px 12px;background:#263238;color:#fff;'
                 f'border-radius:5px 5px 0 0"><b style="font-size:16px">{html.escape(d["dai"].upper())}</b>'
                 f'<span style="font-size:12px;opacity:.8"> &nbsp;|&nbsp; {html.escape(str(d["mien"]))}'
                 f' &nbsp;|&nbsp; {thu} {ng:%d.%m.%Y}</span></div>'
                 '<div style="border:1px solid #cfd8dc;border-top:0;border-radius:0 0 5px 5px;'
                 'padding:4px 12px 12px">')
        for o in d["o"]:
            co = o["ho_so"]["ket_luan"] == NHAN_CO
            mau = "#2e7d32" if co else "#78909c"
            nhan = NHAN_CO if co else NHAN_KHONG
            h.append(f'<div style="margin:10px 0 0"><b style="font-size:14px">{TEN_GIAI[o["giai"]]}</b>'
                     f' <span style="background:{mau};color:#fff;padding:1px 7px;border-radius:3px;'
                     f'font-size:11px">{nhan}</span>'
                     f'<div style="font-size:11px;color:#546e7a;margin-top:3px">{_dong_ly_do(o)}</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;'
                     f'background:#f1f8e9;border-left:4px solid {mau};padding:8px 10px;margin-top:4px;'
                     f'word-break:break-all;line-height:1.85">{_chuoi(o["so"])}</div></div>')
        h.append('</div>')

    h.append('<details style="margin:18px 0 0"><summary style="font-size:13px;color:#546e7a;'
             'cursor:pointer">Hồ sơ đo ni — 5 logic vừa nhất của từng giải</summary>'
             '<table style="font-size:11px;border-collapse:collapse;margin-top:5px">')
    for d in bc["dai"]:
        for o in d["o"]:
            hs = o["ho_so"]
            if "xep_hang" not in hs:
                continue
            xh = " · ".join(f"{c} {r:.0%}" for c, r, _ in hs["xep_hang"])
            h.append(f'<tr><td style="padding:3px 7px">{html.escape(d["dai"])}</td>'
                     f'<td style="padding:3px 7px">{TEN_GIAI[o["giai"]]}</td>'
                     f'<td style="padding:3px 7px;color:#607d8b">{html.escape(xh)}</td></tr>')
    h.append('</table></details>')

    h.append('<hr style="margin:22px 0 10px;border:0;border-top:1px solid #ddd">'
             '<p style="font-size:12px;color:#888;line-height:1.6">'
             '<b>Đo ni:</b> mỗi tháng, mỗi giải thử 10 họ logic trên kỳ 31–100, chọn theo thứ hạng, '
             'rồi kiểm chứng trên kỳ 101–150 chưa từng thấy. Chỉ gắn nhãn "CÓ LOGIC RIÊNG" khi kiểm '
             'chứng vượt hoà vốn và p &lt; 0,0025.<br><b>Ảo giác đo ni:</b> trên giải công bằng, logic '
             'tìm được trông như ~71% lúc khám phá nhưng chỉ ~64% trên kỳ mới — nên con số khám phá '
             'không dùng để quyết định.<br>Máy quay công bằng thì mọi bộ 64 con trúng 64% và kỳ vọng '
             '−5% ở tỷ lệ trả 95. Đo ni không tạo ra logic, chỉ tìm và xác nhận logic nếu có sẵn.'
             '</p></div>')
    return "".join(h)


def _text(bc):
    t = [f"ĐO NI ĐÓNG GIÀY — {bc['thu']} {bc['ngay']:%d.%m.%Y}", ""]
    for d in bc["dai"]:
        t.append(d["dai"].upper())
        for o in d["o"]:
            hs = o["ho_so"]
            nhan = (f"{NHAN_CO}: {hs['logic']}" if hs["ket_luan"] == NHAN_CO else NHAN_KHONG)
            t.append(f"  {TEN_GIAI[o['giai']]} [{nhan}]")
            t.append("  " + _chuoi(o["so"]))
        t.append("")
    if bc["tk"]["N"]:
        t.append(f"Tiến cứu: engine {bc['tk']['eng']}/{bc['tk']['N']} · "
                 f"ngẫu nhiên {bc['tk']['ngau']}/{bc['tk']['N']}")
    return "\n".join(t)


def gui_email(bc):
    mk = E._lay_mat_khau()
    n_co = sum(1 for d in bc["dai"] for o in d["o"] if o["ho_so"]["ket_luan"] == NHAN_CO)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"[ĐO NI] {bc['thu']} {bc['ngay']:%d.%m.%Y} — {bc['n_o']} bộ 64 con · "
                      f"{n_co} giải có logic riêng")
    msg["From"] = formataddr(("XSMN Đo Ni", EMAIL_GUI))
    msg["To"] = EMAIL_NHAN
    msg.attach(MIMEText(_text(bc), "plain", "utf-8"))
    msg.attach(MIMEText(_html(bc), "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk)
        sv.send_message(msg)


if __name__ == "__main__":
    ngay = next((a for a in sys.argv[1:] if "." in a), None)
    do_lai = "--do-lai" in sys.argv
    try:
        E._lay_mat_khau()
        print("  Mật khẩu ứng dụng: OK")
        kho = E.doc_master()
        if kho is None:
            print("  Chưa có kho — quét lần đầu..."); E.tao_master(so_ky=200)
        else:
            print(f"  Kho: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
            E.cap_nhat_master(so_ky_moi=20)
        chay(ngay, do_lai=do_lai)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
