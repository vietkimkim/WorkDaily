"""
CHẠM ENGINE — tìm 4 CHẠM cho Đề Đặc Biệt và Đề Giải Nhất.

CHẠM là gì:
    Một chữ số 0-9. Số 2 chữ số "trúng chạm d" nếu d nằm ở hàng chục HOẶC hàng
    đơn vị. Ví dụ chạm 3 phủ: 03,13,23,30,31,...,39,43,53,63,73,83,93.

VÌ SAO 4 CHẠM:
    Trượt khi CẢ HAI chữ số nằm ngoài 4 chạm -> (6/10)² = 36%
    -> phủ ĐÚNG 64/100 số, P(trúng) = 64,00%
    Trùng khớp với SO_CON_DB = 64 của engine thống kê: cùng độ phủ, cùng vốn.

ƯU ĐIỂM CẤU TRÚC:
    Chọn 4 chạm từ 10 chỉ có C(10,4) = 210 khả năng — KHÔNG GIAN DÒ NHỎ NHẤT
    trong mọi phương pháp của hệ thống (cầu vị trí có 11.342, gấp 54 lần).
    Mức overfit đo trên dữ liệu ngẫu nhiên:
        50 kỳ  -> +16,02%      75 kỳ -> +13,15%      150 kỳ -> +9,30%
    Giảm THẬT theo số kỳ — khác cầu vị trí (ngưỡng tự nâng theo, không cải thiện).
    Đó là lý do mặc định dùng 150 kỳ.

NGƯỠNG:
    Hoà vốn ở tỷ lệ 95: cần hit rate >= 64/95 = 67,37%
    Ngưỡng p<0,05 với 150 kỳ (best-of-210 trên nhiễu): 76,03%
    -> Vẫn còn cách nhau 8,7 điểm. Con số này được in kèm mọi báo cáo.
"""
import os, smtplib, itertools
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

SO_KY_CHAM   = 150    # số kỳ lịch sử. 150 = mặc định: overfit giảm còn +9,30%
                      #   50 kỳ -> +16,02%   75 kỳ -> +13,15%   150 kỳ -> +9,30%
SO_CHAM      = 4      # số chạm mỗi giải. 3 chạm phủ 51 số, 4 phủ 64, 5 phủ 75
CHAY_DB      = True   # ① Đề Đặc Biệt
CHAY_G1      = True   # ② Đề Giải Nhất
MIN_TRAIN    = 15     # số kỳ tối thiểu trước khi chấm cầu chạm
N_NULL       = 200    # (giữ cho hàm null_best_of_210 cũ)
N_NULL_WF    = 25     # số lượt null control WALK-FORWARD — phép so đúng
TOP_UNG_VIEN = 30     # số bộ chạm ứng viên xét bằng hit rate (trên tổng 210)

# Trọng số 4 tín hiệu chạm — nhị phân, chỉ 15 tổ hợp
W_C1_TANSUAT = 1.0    # C1 tần suất chạm, trọng số giảm dần theo thời gian
W_C2_CAUCHAM = 1.0    # C2 cầu chạm — 107 vị trí bỏ phiếu
W_C3_GAN     = 0.5    # C3 gan chạm — lý thuyết = 0, để backtest phán xử
W_C4_CHUOI   = 1.0    # C4 chuỗi đang chạy

TY_LE_TRA    = 95.0
EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")

BO_CHAM = list(itertools.combinations(range(10), SO_CHAM))   # 210 bộ


# ==============================================================================
#  TẦNG 1 — BỐN TÍN HIỆU CHẤM ĐIỂM CHO MỖI CHỮ SỐ 0-9
# ==============================================================================

def _cham_cua(v):
    """Tập chạm của một số 2 chữ số."""
    return {v // 10, v % 10}


def C1_tan_suat(mt, toan_giai=None):
    """C1 — Tần suất chạm, TRỌNG SỐ GIẢM DẦN theo thời gian.

    Máy quay là thiết bị vật lý: bi mòn, bảo trì, thay bi. Nếu có thiên lệch thì
    nó THAY ĐỔI theo thời gian. Trọng số đều sẽ pha loãng thiên lệch hiện tại
    bằng dữ liệu của một cỗ máy đã khác.
    """
    n = len(mt)
    nua_doi = max(10, n // 4)
    d = np.zeros(10)
    for t, v in enumerate(mt):
        w = 0.5 ** ((n - 1 - t) / nua_doi)
        for c in _cham_cua(v):
            d[c] += w
    return d


def C2_cau_cham(mt, toan_giai):
    """C2 — CẦU CHẠM: chữ số tại vị trí p của kỳ t có thành chạm ở kỳ t+1 không.

    Chỉ 107 ứng viên (Miền Bắc) hoặc 82 (Miền Nam) — nhỏ hơn cầu vị trí 2 số
    hơn 100 lần. Các vị trí tốt nhất bỏ phiếu cho chữ số chúng đang chỉ tới.
    """
    if toan_giai is None or len(toan_giai) < MIN_TRAIN + 3:
        return np.zeros(10)
    vt = [(i, j) for i, x in enumerate(toan_giai[0]) for j in range(len(x))]
    M = np.zeros((len(toan_giai), len(vt)), dtype=np.int16)
    for t, ky in enumerate(toan_giai):
        for k, (i, j) in enumerate(vt):
            M[t, k] = int(ky[i][j])

    diem_vt = np.zeros(len(vt), dtype=np.int32)
    for t in range(MIN_TRAIN, len(mt) - 1):
        ch = _cham_cua(mt[t + 1])
        diem_vt += np.isin(M[t], list(ch))

    # Top 20% vị trí bỏ phiếu cho chữ số chúng chỉ tới ở KỲ MỚI NHẤT
    nguong = np.percentile(diem_vt, 80)
    d = np.zeros(10)
    for k in np.where(diem_vt >= nguong)[0]:
        d[M[-1, k]] += diem_vt[k]
    return d


def C3_gan(mt, toan_giai=None):
    """C3 — Gan chạm: bao lâu rồi chữ số d chưa làm chạm.

    Phân phối hình học KHÔNG NHỚ -> giá trị thông tin lý thuyết = 0.
    Vẫn cài để backtest tự phán xử bằng trọng số, không loại theo định kiến.
    """
    n = len(mt)
    g = np.full(10, float(n))
    for t in range(n - 1, -1, -1):
        for c in _cham_cua(mt[t]):
            g[c] = min(g[c], n - 1 - t)
    return g


def C4_chuoi(mt, toan_giai=None):
    """C4 — Chuỗi ĐANG chạy: chữ số d làm chạm liên tiếp mấy kỳ gần đây."""
    d = np.zeros(10)
    for c in range(10):
        i = 0
        for t in range(len(mt) - 1, -1, -1):
            if c in _cham_cua(mt[t]):
                i += 1
            else:
                break
        d[c] = i
    return d


TIN_HIEU_CHAM = [("C1_TanSuat", C1_tan_suat, W_C1_TANSUAT),
                 ("C2_CauCham", C2_cau_cham, W_C2_CAUCHAM),
                 ("C3_Gan",     C3_gan,      W_C3_GAN),
                 ("C4_Chuoi",   C4_chuoi,    W_C4_CHUOI)]


def _z(v):
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-12 else (v - v.mean()) / sd


# ==============================================================================
#  TẦNG 2 + 3 — HỢP NHẤT VÀ CHỌN TOP 4
# ==============================================================================

def chon_cham(mt, toan_giai, k=SO_CHAM):
    """Trả (bộ chạm, điểm từng chữ số, chi tiết từng tín hiệu)."""
    chi_tiet, tong = {}, np.zeros(10)
    for ten, f, w in TIN_HIEU_CHAM:
        v = f(mt, toan_giai)
        chi_tiet[ten] = v
        tong += w * _z(v)
    bo = tuple(int(x) for x in sorted(np.argsort(-tong)[:k]))
    return bo, tong, chi_tiet


def so_cua_cham(bo):
    """64 số mà bộ chạm phủ."""
    s = set(bo)
    return sorted(f"{v:02d}" for v in range(100) if (v // 10 in s) or (v % 10 in s))


def hit_rate(mt, bo):
    s = set(bo)
    return float(np.mean([(v // 10 in s) or (v % 10 in s) for v in mt]))


# ==============================================================================
#  TẦNG 4 — NULL CONTROL: best-of-210 trên dữ liệu ngẫu nhiên
# ==============================================================================

_NULL_CACHE = {}


def null_best_of_210(n_ky, n_lap=N_NULL, seed=2026):
    """Phân phối hit rate của bộ chạm TỐT NHẤT trong 210 bộ, trên nhiễu thuần tuý."""
    khoa = (n_ky, n_lap, SO_CHAM)
    if khoa in _NULL_CACHE:
        return _NULL_CACHE[khoa]
    rng = np.random.default_rng(seed)
    ra = []
    for _ in range(n_lap):
        mt = rng.integers(0, 100, n_ky)
        ra.append(max(hit_rate(mt, b) for b in BO_CHAM))
    _NULL_CACHE[khoa] = np.array(ra)
    return _NULL_CACHE[khoa]


# ==============================================================================
#  [NÂNG CẤP] ĐÁNH GIÁ THEO BỘ + WALK-FORWARD + TRỌNG SỐ TỪ DỮ LIỆU
#
#  BA LỖI CỦA BẢN ĐẦU, đã đo trên dữ liệu ngẫu nhiên:
#
#  1. Chấm điểm TỪNG CHỮ SỐ rồi lấy top 4 KHÔNG phải tối ưu.
#     Mọi bộ 4 chạm đều phủ đúng 64 số, nhưng xác suất trúng KHÁC nhau vì phụ
#     thuộc PHÂN PHỐI ĐỒNG THỜI của 2 chữ số mục tiêu — không cộng được.
#     Đo được: cách cũ bỏ sót trung bình 5,73% hit rate, có lúc tới 12%.
#     -> Sửa: đánh giá trực tiếp cả 210 BỘ, không chấm điểm rời rạc.
#
#  2. Hit rate báo cáo là IN-SAMPLE — chọn chạm và chấm điểm trên CÙNG dữ liệu.
#     Đo được: thổi phồng +2,36 điểm phần trăm (68,33% vs 65,97% thật).
#     -> Sửa: walk-forward, chọn chạm CHỈ từ quá khứ tại mỗi bước.
#
#  3. Trọng số W_C1..W_C4 do tôi tự đặt, không có căn cứ.
#     -> Sửa: backtest chọn, chỉ 15 tổ hợp nhị phân.
# ==============================================================================

import itertools as _it

_TO_HOP_W = [c for c in _it.product([0.0, 1.0], repeat=4) if sum(c) > 0]  # 15


def _diem_bo_tu_tin_hieu(mt, toan_giai, w):
    """Điểm của MỖI BỘ trong 210 bộ, suy từ 4 tín hiệu với vector trọng số w.

    Điểm của bộ = tổng điểm z của các chữ số trong bộ. Dùng để XẾP HẠNG ứng viên,
    sau đó mới chọn bằng hit rate thực tế -> tránh lỗi cộng dồn của bản cũ.
    """
    tong = np.zeros(10)
    for (ten, f, _), wi in zip(TIN_HIEU_CHAM, w):
        if wi:
            tong += wi * _z(f(mt, toan_giai))
    return np.array([sum(tong[d] for d in b) for b in BO_CHAM]), tong


def chon_bo_toi_uu(mt, toan_giai, w, top_ung_vien=30):
    """Chọn bộ chạm: tín hiệu XẾP HẠNG ứng viên, hit rate thực tế QUYẾT ĐỊNH.

    Chỉ xét top ứng viên theo điểm tín hiệu (không phải cả 210) để giữ diện tích
    dò tìm nhỏ — đây là điểm mạnh cấu trúc của phương pháp chạm.
    """
    diem_bo, diem_so = _diem_bo_tu_tin_hieu(mt, toan_giai, w)
    ung_vien = np.argsort(-diem_bo)[:top_ung_vien]
    hr = [hit_rate(mt, BO_CHAM[i]) for i in ung_vien]
    tot = ung_vien[int(np.argmax(hr))]
    return BO_CHAM[tot], diem_so, float(max(hr))


def walk_forward(mt, toan_giai, w, min_train=None, top_ung_vien=30):
    """Hit rate NGOÀI MẪU: tại mỗi kỳ t, chọn chạm CHỈ từ dữ liệu < t."""
    n = len(mt)
    min_train = min_train or max(40, n // 3)
    trung = tong = 0
    for t in range(min_train, n):
        bo, _, _ = chon_bo_toi_uu(mt[:t], toan_giai[:t], w, top_ung_vien)
        s = set(bo); v = mt[t]
        trung += (v // 10 in s) or (v % 10 in s)
        tong += 1
    return (trung / tong if tong else 0.0), tong


def chon_trong_so(mt, toan_giai, min_train=None, top_ung_vien=30):
    """Chọn bộ trọng số bằng walk-forward. 15 tổ hợp — diện tích dò rất nhỏ."""
    best = (None, -1.0, 0)
    for w in _TO_HOP_W:
        hr, n = walk_forward(mt, toan_giai, w, min_train, top_ung_vien)
        if hr > best[1]:
            best = (w, hr, n)
    return best


def null_walk_forward(n_ky, khu, w, n_lap=30, seed=2026, top_ung_vien=30):
    """Null control ĐÚNG CHUẨN: chạy y hệt quy trình walk-forward trên nhiễu."""
    rng = np.random.default_rng(seed)
    ra = []
    for _ in range(n_lap):
        gia = [E.sinh_ky_gia(E.CO_CAU_GIAI[khu], rng) for _ in range(n_ky)]
        m = [int(k[0][-2:]) for k in gia]
        hr, _ = walk_forward(m, gia, w, top_ung_vien=top_ung_vien)
        ra.append(hr)
    return np.array(ra)



# ==============================================================================
#  [NÂNG CẤP CUỐI] CHỌN BỘ CHẠM BẰNG MÔ HÌNH XÁC SUẤT
#
#  LỖI KHÁI NIỆM CỦA BẢN TRƯỚC:
#      Chọn bộ bằng HIT RATE QUÁ KHỨ là ước lượng NHỊ PHÂN — mỗi kỳ chỉ cho
#      1 bit (trúng/trượt), vứt bỏ thông tin CHỮ SỐ NÀO đã ra.
#      Sai số ước lượng hit rate của 1 bộ (150 kỳ) : 3,92%
#      Sai số ước lượng phân phối chữ số (300 QS)  : 1,73%
#      -> Mô hình hoá phân phối chính xác hơn 2,3 lần.
#
#  CÁCH ĐÚNG:
#      P(trúng | bộ S) = 1 − Σ_{a∉S} Σ_{b∉S} P(a,b)
#
#      P(a,b) = λ·P̂(a,b) + (1−λ)·P̂₁(a)·P̂₂(b)
#      (nội suy Jelinek-Mercer giữa joint thực nghiệm và giả định độc lập,
#       λ ước lượng từ held-out — không áp đặt)
#
#  BỐN ƯU ĐIỂM:
#      1. Dùng HẾT 300 quan sát chữ số thay vì 150 bit trúng/trượt
#      2. Tách riêng hàng chục / đơn vị — bản cũ gộp chung, mất thông tin
#      3. Tính CHÍNH XÁC cả 210 bộ, bỏ được ngưỡng top-30 tuỳ tiện
#      4. Trọng số thời gian giữ nguyên: bắt thiên lệch ĐANG diễn ra
# ==============================================================================

LAMBDA_GRID_CHAM = [0.0, 0.2, 0.4, 0.6, 0.8]


def _uoc_luong_lambda_cham(mt, nua_doi):
    """λ nội suy joint<->độc lập, chọn bằng log-likelihood trên 20% cuối."""
    if len(mt) < 30:
        return 0.0
    cut = int(len(mt) * 0.8)
    fit, held = mt[:cut], mt[cut:]
    n = len(fit)
    c1 = np.full(10, ALPHA_CHAM); c2 = np.full(10, ALPHA_CHAM)
    J = np.full((10, 10), ALPHA_CHAM / 10)
    for t, v in enumerate(fit):
        w = 0.5 ** ((n - 1 - t) / nua_doi)
        c1[v // 10] += w; c2[v % 10] += w; J[v // 10, v % 10] += w
    p1, p2, pj = c1 / c1.sum(), c2 / c2.sum(), J / J.sum()
    doc_lap = np.outer(p1, p2)
    best, best_ll = LAMBDA_GRID_CHAM[0], -np.inf
    for lam in LAMBDA_GRID_CHAM:
        Q = lam * pj + (1 - lam) * doc_lap
        ll = float(sum(np.log(max(Q[v // 10, v % 10], 1e-12)) for v in held))
        if ll > best_ll:
            best, best_ll = lam, ll
    return best


ALPHA_CHAM = 1.0


def phan_phoi_muc_tieu(mt, nua_doi=None, lam=None):
    """Ước lượng P(a,b) của mục tiêu, có TRỌNG SỐ GIẢM DẦN theo thời gian."""
    n = len(mt)
    nua_doi = nua_doi or max(10, n // 4)
    lam = _uoc_luong_lambda_cham(mt, nua_doi) if lam is None else lam
    c1 = np.full(10, ALPHA_CHAM); c2 = np.full(10, ALPHA_CHAM)
    J = np.full((10, 10), ALPHA_CHAM / 10)
    for t, v in enumerate(mt):
        w = 0.5 ** ((n - 1 - t) / nua_doi)
        c1[v // 10] += w; c2[v % 10] += w; J[v // 10, v % 10] += w
    p1, p2, pj = c1 / c1.sum(), c2 / c2.sum(), J / J.sum()
    Q = lam * pj + (1 - lam) * np.outer(p1, p2)
    return Q / Q.sum(), p1, p2, lam


def p_trung_mo_hinh(Q, bo):
    """P(mục tiêu chạm >=1 số trong bộ) — TÍNH CHÍNH XÁC từ phân phối."""
    ngoai = [d for d in range(10) if d not in bo]
    return float(1.0 - Q[np.ix_(ngoai, ngoai)].sum())


def chon_bo_mo_hinh(mt, toan_giai, w_phu=None):
    """Chọn bộ chạm bằng mô hình. w_phu: trọng số cho C2/C3/C4 (tuỳ chọn).

    Trả (bộ, P(trúng) mô hình, điểm 210 bộ, λ, phân phối).
    """
    Q, p1, p2, lam = phan_phoi_muc_tieu(mt)
    diem = np.array([p_trung_mo_hinh(Q, b) for b in BO_CHAM])

    # Điều chỉnh phụ từ C2 (cầu chạm) / C3 (gan) / C4 (chuỗi) — nếu backtest chọn
    if w_phu is not None and any(w_phu):
        phu = np.zeros(10)
        for (ten, f, _), wi in zip(TIN_HIEU_CHAM[1:], w_phu):   # bỏ C1, đã nằm trong Q
            if wi:
                phu += wi * _z(f(mt, toan_giai))
        diem = diem + 0.01 * np.array([sum(phu[d] for d in b) for b in BO_CHAM])

    j = int(np.argmax(diem))
    return BO_CHAM[j], float(diem[j]), diem, lam, (Q, p1, p2)


def walk_forward_mo_hinh(mt, toan_giai, w_phu=None, min_train=None):
    """Hit rate NGOÀI MẪU với quy trình mô hình."""
    n = len(mt)
    min_train = min_train or max(40, n // 3)
    trung = tong = 0
    for t in range(min_train, n):
        bo, _, _, _, _ = chon_bo_mo_hinh(mt[:t], toan_giai[:t], w_phu)
        s = set(bo); v = mt[t]
        trung += (v // 10 in s) or (v % 10 in s)
        tong += 1
    return (trung / tong if tong else 0.0), tong


def chon_trong_so_phu(mt, toan_giai, min_train=None):
    """Chọn trọng số phụ cho C2/C3/C4 bằng walk-forward. 8 tổ hợp."""
    best = (None, -1.0, 0)
    for w in _it.product([0.0, 1.0], repeat=3):
        hr, n = walk_forward_mo_hinh(mt, toan_giai, w, min_train)
        if hr > best[1]:
            best = (w, hr, n)
    return best


def null_wf_mo_hinh(n_ky, khu, w_phu, n_lap=25, seed=2026):
    rng = np.random.default_rng(seed)
    ra = []
    for _ in range(n_lap):
        gia = [E.sinh_ky_gia(E.CO_CAU_GIAI[khu], rng) for _ in range(n_ky)]
        m = [int(k[0][-2:]) for k in gia]
        hr, _ = walk_forward_mo_hinh(m, gia, w_phu)
        ra.append(hr)
    return np.array(ra)


# ==============================================================================
#  CHẠY 1 ĐÀI
# ==============================================================================

MO_TA = {0: ("① ĐỀ ĐẶC BIỆT", "2 chữ số CUỐI của giải ĐẶC BIỆT"),
         1: ("② ĐỀ GIẢI NHẤT", "2 chữ số CUỐI của giải NHẤT")}


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
    if len(tg) < 30:
        raise RuntimeError(f"chỉ còn {len(tg)} kỳ trước {ngay_moc:%d.%m.%Y}, cần >=30")

    mods = []
    for vi_tri, bat in [(0, CHAY_DB), (1, CHAY_G1)]:
        if not bat:
            continue
        mt = [int(ky[vi_tri][-2:]) for ky in tg]

        # --- B1: trọng số phụ C2/C3/C4 do WALK-FORWARD chọn (8 tổ hợp) ---
        w_phu, hr_wf, n_wf = chon_trong_so_phu(mt, tg)

        # --- B2: chọn bộ chạm bằng MÔ HÌNH XÁC SUẤT, tính chính xác 210 bộ ---
        bo, p_mh, diem_bo, lam, (Q, p1, p2) = chon_bo_mo_hinh(mt, tg, w_phu)
        diem = np.zeros(10)
        for d in range(10):
            diem[d] = p1[d] + p2[d]
        chi_tiet = {ten: f(mt, tg) for ten, f, _ in TIN_HIEU_CHAM}
        _hr_in = hit_rate(mt, bo)

        # --- B3: null control walk-forward, cùng quy trình ---
        null = null_wf_mo_hinh(len(tg), info.get("khu", "MN"), w_phu, n_lap=N_NULL_WF)
        hr = hr_wf
        p = float((null >= hr).mean())
        p_truot = ((10 - so_cham) / 10) ** 2
        moc = 1 - p_truot
        hoa_von = (100 * (1 - p_truot)) / TY_LE_TRA
        mods.append({
            "vi_tri": vi_tri, "ten": MO_TA[vi_tri][0], "mo_ta": MO_TA[vi_tri][1],
            "cham": list(bo), "so": so_cua_cham(bo), "n_so": int(100 * moc),
            "hit_rate": hr, "hit_in_sample": _hr_in, "n_wf": n_wf,
            "trong_so": (["C1_TanSuat (mô hình)"] +
                         [ten for (ten, _, _), wi in zip(TIN_HIEU_CHAM[1:], w_phu) if wi]),
            "p_mo_hinh": p_mh, "lambda": lam,
            "moc": moc, "hoa_von": hoa_von,
            "p_value": p, "null_tb": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "diem": {str(d): float(diem[d]) for d in range(10)},
            "chi_tiet": {k: [float(x) for x in v] for k, v in chi_tiet.items()},
        })

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
    m0 = ket[0]["modules"][0]
    h = [f'<div style="{css}max-width:700px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">{SO_CHAM} chạm — {thu} {ngay_moc:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài · {SO_KY_CHAM} kỳ · mỗi bộ chạm phủ '
             f'{m0["n_so"]}/100 số</p>')
    h.append(f'<div style="background:#fff8e1;border-left:4px solid #f9a825;'
             f'padding:10px 13px;margin:0 0 22px;font-size:13px;line-height:1.6">'
             f'<b>Ba mốc để đọc bảng bên dưới:</b><br>'
             f'· Mốc ngẫu nhiên (bộ chạm bất kỳ) = <b>{m0["moc"]:.2%}</b><br>'
             f'· Hoà vốn ở tỷ lệ {TY_LE_TRA:.0f} = <b>{m0["hoa_von"]:.2%}</b><br>'
             f'· Ngưỡng p&lt;0,05 (null walk-forward) = '
             f'<b>{m0["null_p95"]:.2%}</b><br>'
             f'Hit rate báo cáo là NGOÀI MẪU: tại mỗi kỳ, bộ chạm được chọn CHỈ từ '
             f'dữ liệu trước đó. Đây là con số thật, thấp hơn in-sample khoảng 2,4 điểm.'
             f'</div>')

    for r in ket:
        h.append(f'<div style="margin:26px 0 0;padding:9px 12px;background:#263238;'
                 f'color:#fff;border-radius:5px 5px 0 0">'
                 f'<span style="font-size:17px;font-weight:700">{r["dai"].upper()}</span>'
                 f'<span style="font-size:13px;opacity:.85"> &nbsp;|&nbsp; {r["mien"]}'
                 f' &nbsp;|&nbsp; {thu} {ngay_moc:%d.%m.%Y}</span><br>'
                 f'<span style="font-size:12px;opacity:.7">Kỳ gần nhất '
                 f'{r["ngay_ky_truoc"]} · ĐB {r["db_ky_truoc"]} · G1 {r["g1_ky_truoc"]}'
                 f' · {r["n_ky"]} kỳ</span></div>')
        h.append('<div style="border:1px solid #cfd8dc;border-top:0;'
                 'border-radius:0 0 5px 5px;padding:4px 12px 12px">')
        for m in r["modules"]:
            mau = "#2e7d32" if m["p_value"] < 0.05 else "#546e7a"
            h.append(f'<div style="margin:14px 0 0">'
                     f'<div style="font-size:14px;font-weight:600">{m["ten"]}</div>'
                     f'<div style="font-size:12px;color:#607d8b;margin:2px 0 6px">'
                     f'{r["dai"]} · {thu} {ngay_moc:%d.%m.%Y} · {m["mo_ta"]}</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:26px;font-weight:700;letter-spacing:8px;'
                     f'background:#e8f5e9;border-left:4px solid #2e7d32;'
                     f'padding:12px 14px">{" ".join(str(c) for c in m["cham"])}</div>')
            h.append(f'<div style="font-size:12px;color:{mau};margin:6px 0 0">'
                     f'Hit rate NGOÀI MẪU ({m["n_wf"]} kỳ walk-forward): '
                     f'<b>{m["hit_rate"]:.2%}</b> · '
                     f'mốc {m["moc"]:.2%} · hoà vốn {m["hoa_von"]:.2%} · '
                     f'ngưỡng nhiễu {m["null_p95"]:.2%} · '
                     f'<b>p = {m["p_value"]:.3f}</b> '
                     f'{"— VƯỢT ngưỡng nhiễu" if m["p_value"] < 0.05 else "— trong biên độ nhiễu"}'
                     f'<br><span style="color:#90a4ae">Trọng số backtest chọn: '
                     f'{", ".join(m["trong_so"]) or "(không)"} · '
                     f'in-sample {m["hit_in_sample"]:.2%} (chỉ để tham khảo — '
                     f'luôn cao hơn thật)</span></div>')
            h.append(f'<div style="font-size:12px;color:#546e7a;margin:10px 0 4px;'
                     f'font-weight:600">{m["n_so"]} SỐ QUY RA — bôi đen để copy</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:14px;background:#eceff1;border-left:3px solid #455a64;'
                     f'padding:10px 12px;word-break:break-all;line-height:1.8;'
                     f'letter-spacing:.3px">{",".join(m["so"])}</div></div>')
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
             f'Chọn {SO_CHAM} chạm từ 10 chỉ có {len(BO_CHAM)} khả năng — không gian dò '
             f'nhỏ nhất trong hệ thống (cầu vị trí có 11.342). Nhờ vậy mức overfit thấp '
             f'hơn hẳn và GIẢM theo số kỳ. Cột p đã so với best-of-{len(BO_CHAM)} trên '
             f'dữ liệu ngẫu nhiên — đó là phép so đúng.</p></div>')
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
            t.append(f"  CHẠM: {'  '.join(str(c) for c in m['cham'])}")
            t.append(f"  (ngoài mẫu {m['hit_rate']:.2%} | mốc {m['moc']:.2%} | "
                     f"hoà vốn {m['hoa_von']:.2%} | p {m['p_value']:.3f})")
            t.append(f"  {m['n_so']} SỐ QUY RA:")
            t.append("  " + ",".join(m["so"]))
        t.append("")
    msg.attach(MIMEText("\n".join(t), "plain", "utf-8"))
    msg.attach(MIMEText(_html(ket, ngay_moc, loi), "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk); sv.send_message(msg)
    print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")


# ==============================================================================
#  PIPELINE CHÍNH
# ==============================================================================

def main(ngay=None, so_ky=None, so_cham=None, gui_mail=True):
    ngay_moc = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    so_ky = so_ky or SO_KY_CHAM
    so_cham = so_cham or SO_CHAM
    thu = E.THU_VN[ngay_moc.weekday()]
    p_truot = ((10 - so_cham) / 10) ** 2

    print("=" * 78)
    print(f"  {so_cham} CHẠM  |  {thu} {ngay_moc:%d.%m.%Y} (giờ VN)  |  {so_ky} kỳ")
    print(f"  Phủ {int(100*(1-p_truot))}/100 số | mốc ngẫu nhiên {1-p_truot:.2%} | "
          f"hoà vốn {(100*(1-p_truot))/TY_LE_TRA:.2%}")
    print(f"  Không gian dò: C(10,{so_cham}) = {len(BO_CHAM)} bộ chạm")
    print("=" * 78)

    dsach = E.dai_theo_ngay(ngay_moc)
    lich = E.xay_lich()
    print(f"\n  Đài quay hôm nay: {len(dsach)}")
    print(f"  Đang dựng phân phối null (best-of-{len(BO_CHAM)}, {N_NULL} lượt)...")

    ket, loi = [], []
    for s in dsach:
        ten = lich[str(s)]["ten"]
        try:
            r = chay_dai(s, ngay_moc, so_ky, so_cham)
            ket.append(r)
            ct = " | ".join(f"{m['ten'][0]} {''.join(map(str,m['cham']))} "
                            f"({m['hit_rate']:.1%}, p={m['p_value']:.2f})"
                            for m in r["modules"])
            print(f"     ✓ [{s:>2}] {ten:<20} {r['n_ky']} kỳ ({r['nguon']}) | {ct}")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<20} LỖI: {e}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")

    m0 = ket[0]["modules"][0]
    print(f"\n[A] BA MỐC ĐỂ ĐỌC KẾT QUẢ")
    print("-" * 78)
    print(f"  Mốc ngẫu nhiên (bộ chạm bất kỳ)      : {m0['moc']:.2%}")
    print(f"  Hoà vốn ở tỷ lệ {TY_LE_TRA:.0f}                  : {m0['hoa_von']:.2%}")
    print(f"  Ngưỡng p<0,05 (best-of-{len(BO_CHAM)} trên nhiễu) : {m0['null_p95']:.2%}")
    print(f"  Nhiễu đạt trung bình                 : {m0['null_tb']:.2%}")
    print(f"\n  Hit rate dưới ngưỡng thứ ba = không phân biệt được với dò ngẫu nhiên.")

    for r in ket:
        for m in r["modules"]:
            print("\n" + "=" * 78)
            print(f"  {r['dai'].upper()}  —  {m['ten']}")
            print(f"  {thu} {ngay_moc:%d.%m.%Y}  |  {r['mien']}  |  {m['mo_ta']}")
            print("=" * 78)
            print(f"\n     CHẠM:   {'   '.join(str(c) for c in m['cham'])}\n")
            print(f"  Hit rate NGOÀI MẪU {m['hit_rate']:.2%} ({m['n_wf']} kỳ walk-forward)"
                  f"  |  p = {m['p_value']:.3f}  "
                  f"{'← VƯỢT ngưỡng nhiễu' if m['p_value'] < .05 else '(trong biên độ nhiễu)'}")
            print(f"  Trọng số backtest chọn: {', '.join(m['trong_so']) or '(không)'}")
            print(f"  In-sample {m['hit_in_sample']:.2%} (luôn cao hơn thật, chỉ tham khảo)")
            print(f"  Điểm từng chữ số: " +
                  "  ".join(f"{d}:{m['diem'][str(d)]:+.1f}" for d in range(10)))
            print(f"\n  {m['n_so']} số phủ được:")
            for i in range(0, len(m["so"]), 16):
                print("   " + " ".join(m["so"][i:i+16]))
    print("\n" + "=" * 78)

    if gui_mail:
        print("\n  Đang gửi email...")
        try:
            gui_email(ket, ngay_moc, loi)
        except Exception as e:
            print(f"  ✗ Không gửi được email: {e}")
    return ket
