"""
DE_ENGINE12 — 12 engine THỐNG KÊ tìm 64 con 2 số cho ĐB · G1 · G8.

KHÁC HẲN BỘ CŨ:
    Bộ cũ (E01-E12) gồm 6 quy tắc dân gian: gan, bệt, bóng, tổng, hiệu, số lạnh.
    Bộ này KHÔNG có quy tắc dân gian nào. Mọi engine đều là một ƯỚC LƯỢNG
    THỐNG KÊ của phân phối xác suất trên 100 ô, khác nhau ở GIẢ ĐỊNH:

      D01 Dirichlet   : hậu nghiệm Bayes đúng chuẩn, α chọn bằng held-out
      D02 JamesStein  : co ngót về trung bình — tối ưu khi mẫu thưa
      D03 JelinekJoint: nội suy phân phối đồng thời với giả định độc lập
      D04 RecencyML   : nửa đời ước lượng bằng log-likelihood, không áp đặt
      D05 MarkovShrink: Markov 100 trạng thái, co ngót về phân phối biên
      D06 CrossPrize  : ước lượng từ TOÀN BỘ 18 giải rồi áp cho giải mục tiêu
      D07 Hierarchy   : mô hình phân tầng tài/xỉu × chẵn/lẻ × kép
      D08 DigitIndep  : P(ab) = P1(a)·P2(b), giả định độc lập hai chữ số
      D09 KernelSmooth: làm mượt theo khoảng cách chữ số (ô lân cận vay mượn nhau)
      D10 MinVariance : chọn 64 ô tối thiểu hoá phương sai danh mục
      D11 GapHazard   : mô hình sống sót theo khoảng cách — ĐỐI CHỨNG ÂM
                        (hazard phẳng -> lý thuyết = 0, để đo mức nhiễu)
      D12 Uniform     : BỐC NGẪU NHIÊN 64 ô — ĐỐI CHỨNG CHUẨN

    D11 và D12 là ĐỐI CHỨNG CỐ Ý. Nếu 10 engine kia không vượt được D12,
    đó là câu trả lời — và nó đáng tin hơn mọi p-value.

TRẦN CỨNG KHÔNG ENGINE NÀO VƯỢT ĐƯỢC:
    64 ô / 100 ô -> P(trúng) = 64,00% nếu máy quay đều.
    Hoà vốn ở tỷ lệ 95 cần 67,37%, tức 36 ô BỊ LOẠI phải nhẹ hơn đều 9,4%.
    Engine chỉ đổi CHỌN Ô NÀO, không đổi SỐ Ô.
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

SO_KY        = 100    # 90 học + 10 đo phong độ
CUA_SO_GAN   = 10
SO_CON       = 64     # 64 ô -> P(trúng) 64,00% · hoà vốn 67,37% · EV -5,0%
CHAY_DB      = True
CHAY_G1      = True
CHAY_G8      = True   # chỉ MN/MT

SO_ENGINE    = 4      # gộp bao nhiêu engine hàng đầu để ra bộ số
W_GAN        = 0.6    # trọng số phong độ 10 kỳ gần
MIN_TRAIN    = 25
TY_LE_TRA    = 95.0

EMAIL_NHAN   = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI    = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")

PAIRS = np.array([(i // 10, i % 10) for i in range(100)])
P_D1, P_D2 = PAIRS[:, 0], PAIRS[:, 1]
MO_TA = {"DB": ("① ĐỀ ĐẶC BIỆT", "2 chữ số CUỐI giải ĐẶC BIỆT"),
         "G1": ("② ĐỀ GIẢI NHẤT", "2 chữ số CUỐI giải NHẤT"),
         "G8": ("③ ĐỀ ĐẦU (G8)", "giải TÁM — đã là số 2 chữ số")}


def _z(v):
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-12 else (v - v.mean()) / sd


def _dem(mt, w=None):
    """Đếm 100 ô, tuỳ chọn trọng số thời gian nửa đời w."""
    c = np.zeros(100); n = len(mt)
    for t, v in enumerate(mt):
        c[v] += 1.0 if w is None else 0.5 ** ((n - 1 - t) / w)
    return c


# ==============================================================================
#  12 ENGINE — mỗi engine trả mảng 100 điểm (càng cao càng nên chọn)
#  mt : list số mục tiêu 0-99 theo từng kỳ
#  pool: list theo kỳ, mỗi kỳ là list 2-số-cuối của TOÀN BỘ 18 giải
# ==============================================================================

def D01_Dirichlet(mt, pool, rng):
    """Hậu nghiệm Dirichlet-multinomial. α chọn bằng log-likelihood held-out.
       Đây là ước lượng Bayes ĐÚNG CHUẨN cho đếm đa thức thưa."""
    n = len(mt)
    if n < 20:
        return np.zeros(100)
    cut = int(n * 0.8)
    best_a, best_ll = 1.0, -np.inf
    c_fit = _dem(mt[:cut])
    for a in (0.1, 0.5, 1.0, 2.0, 5.0, 20.0):
        p = (c_fit + a) / (c_fit.sum() + 100 * a)
        ll = float(np.sum(np.log(p[mt[cut:]])))
        if ll > best_ll:
            best_a, best_ll = a, ll
    c = _dem(mt)
    return _z(np.log((c + best_a) / (c.sum() + 100 * best_a)))


def D02_JamesStein(mt, pool, rng):
    """Co ngót James-Stein về phân phối đều. Hệ số B TỰ CHẨN ĐOÁN:
       B≈0 nghĩa là mọi chênh lệch giữa các ô CHỈ LÀ nhiễu đếm."""
    c = _dem(mt); N = max(c.sum(), 1)
    ph = c / N; u = 0.01
    S = float(((ph - u) ** 2).sum())
    var_nhieu = float((ph * (1 - ph)).sum()) / N
    B = 0.0 if S <= 1e-12 else float(np.clip(1 - 97 * var_nhieu / max(S * 100, 1e-12), 0, 1))
    return _z(np.log(np.clip(u + B * (ph - u), 1e-9, None)))


def D03_JelinekJoint(mt, pool, rng):
    """P(a,b) = λ·P̂(a,b) + (1−λ)·P̂1(a)·P̂2(b). λ từ held-out.
       Nội suy giữa đồng thời thực nghiệm và giả định độc lập."""
    n = len(mt)
    if n < 20:
        return np.zeros(100)
    cut = int(n * 0.8)
    c1 = np.ones(10); c2 = np.ones(10); J = np.full((10, 10), 0.1)
    for v in mt[:cut]:
        c1[v // 10] += 1; c2[v % 10] += 1; J[v // 10, v % 10] += 1
    p1, p2, pj = c1 / c1.sum(), c2 / c2.sum(), J / J.sum()
    dl = np.outer(p1, p2)
    best_l, best_ll = 0.0, -np.inf
    for lam in (0.0, 0.2, 0.4, 0.6, 0.8):
        Q = lam * pj + (1 - lam) * dl
        ll = float(sum(np.log(max(Q[v // 10, v % 10], 1e-12)) for v in mt[cut:]))
        if ll > best_ll:
            best_l, best_ll = lam, ll
    c1 = np.ones(10); c2 = np.ones(10); J = np.full((10, 10), 0.1)
    for v in mt:
        c1[v // 10] += 1; c2[v % 10] += 1; J[v // 10, v % 10] += 1
    p1, p2, pj = c1 / c1.sum(), c2 / c2.sum(), J / J.sum()
    Q = best_l * pj + (1 - best_l) * np.outer(p1, p2)
    return _z(np.log(np.clip(Q[P_D1, P_D2], 1e-12, None)))


def D04_RecencyML(mt, pool, rng):
    """Nửa đời ước lượng bằng log-likelihood held-out, KHÔNG áp đặt.
       Nếu không có trôi theo thời gian, nửa đời tự chọn giá trị lớn nhất."""
    n = len(mt)
    if n < 20:
        return np.zeros(100)
    cut = int(n * 0.8)
    best_h, best_ll = 1e9, -np.inf
    for h in (5, 10, 20, 40, 1e9):
        c = _dem(mt[:cut], None if h > 1e8 else h)
        p = (c + 1.0) / (c.sum() + 100)
        ll = float(np.sum(np.log(p[mt[cut:]])))
        if ll > best_ll:
            best_h, best_ll = h, ll
    c = _dem(mt, None if best_h > 1e8 else best_h)
    return _z(np.log((c + 1.0) / (c.sum() + 100)))


def D05_MarkovShrink(mt, pool, rng):
    """Markov 100 trạng thái, co ngót mạnh về phân phối biên.
       10.000 ô chuyển trạng thái từ ~100 quan sát -> phải co ngót rất mạnh."""
    n = len(mt)
    if n < 20:
        return np.zeros(100)
    bien = (_dem(mt) + 1.0); bien /= bien.sum()
    T = np.zeros(100)
    dem_tu = 0
    for t in range(n - 1):
        if mt[t] == mt[-1]:
            T[mt[t + 1]] += 1; dem_tu += 1
    k = 5.0                               # sức co ngót
    p = (T + k * 100 * bien) / (dem_tu + k * 100)
    return _z(np.log(np.clip(p, 1e-12, None)))


def D06_CrossPrize(mt, pool, rng):
    """Ước lượng phân phối từ TOÀN BỘ 18 giải (gấp 18 lần dữ liệu) rồi áp cho
       giải mục tiêu. Giả định: mọi giải rút từ cùng một phân phối."""
    c = np.zeros(100)
    for ky in pool:
        for v in ky:
            c[v] += 1
    return _z(np.log((c + 1.0) / (c.sum() + 100)))


def D07_Hierarchy(mt, pool, rng):
    """Mô hình phân tầng: ước lượng ở mức ĐẶC TRÍNH (tài/xỉu, chẵn/lẻ, kép,
       chạm tổng) rồi tổ hợp. Ít tham số hơn -> ước lượng ổn định hơn."""
    n = max(len(mt), 1)
    tai = np.mean([v >= 50 for v in mt]); chan = np.mean([v % 2 == 0 for v in mt])
    kep = np.mean([v // 10 == v % 10 for v in mt])
    ts = np.ones(10)
    for v in mt:
        ts[(v // 10 + v % 10) % 10] += 1
    ts /= ts.sum()
    sc = np.zeros(100)
    sc += np.log(np.where(np.arange(100) >= 50, max(tai, .01), max(1 - tai, .01)))
    sc += np.log(np.where(np.arange(100) % 2 == 0, max(chan, .01), max(1 - chan, .01)))
    sc += np.log(np.where(P_D1 == P_D2, max(kep, .005) / .1, max(1 - kep, .005) / .9))
    sc += np.log(ts[(P_D1 + P_D2) % 10] * 10)
    return _z(sc)


def D08_DigitIndep(mt, pool, rng):
    """P(ab) = P1(a)·P2(b). Chỉ 20 tham số thay vì 100 -> mật độ mẫu cao nhất."""
    c1 = np.ones(10); c2 = np.ones(10)
    for v in mt:
        c1[v // 10] += 1; c2[v % 10] += 1
    p1, p2 = c1 / c1.sum(), c2 / c2.sum()
    return _z(np.log(p1[P_D1]) + np.log(p2[P_D2]))


def D09_KernelSmooth(mt, pool, rng):
    """Làm mượt theo khoảng cách chữ số: ô lân cận vay mượn thông tin nhau.
       Giảm phương sai ước lượng khi mẫu thưa."""
    c = _dem(mt)
    G = np.zeros((100, 100))
    for i in range(100):
        d = np.minimum(np.abs(P_D1 - P_D1[i]), 10 - np.abs(P_D1 - P_D1[i])) ** 2 \
            + np.minimum(np.abs(P_D2 - P_D2[i]), 10 - np.abs(P_D2 - P_D2[i])) ** 2
        G[i] = np.exp(-d / 2.0)
    G /= G.sum(1, keepdims=True)
    return _z(np.log(np.clip(G @ c + 0.5, 1e-9, None)))


def D10_MinVariance(mt, pool, rng):
    """Tối thiểu hoá phương sai danh mục: ưu tiên ô có ước lượng ỔN ĐỊNH
       (sai số chuẩn nhỏ), không chỉ ô có tần suất cao."""
    n = max(len(mt), 1)
    c = _dem(mt); p = (c + 1.0) / (c.sum() + 100)
    se = np.sqrt(p * (1 - p) / n)
    return _z(p / (se + 1e-9))


def D11_GapHazard(mt, pool, rng):
    """ĐỐI CHỨNG ÂM. Mô hình sống sót theo khoảng cách chưa về.
       Hazard của phân phối hình học là PHẲNG -> giá trị lý thuyết BẰNG 0.
       Giữ lại để đo mức nhiễu mà một engine vô dụng vẫn đạt được."""
    n = len(mt)
    g = np.full(100, float(n))
    for t in range(n - 1, -1, -1):
        g[mt[t]] = min(g[mt[t]], n - 1 - t)
    return _z(g)


def D12_Uniform(mt, pool, rng):
    """ĐỐI CHỨNG CHUẨN — bốc ngẫu nhiên. Nếu không engine nào vượt được nó,
       đó là câu trả lời, và nó đáng tin hơn mọi p-value."""
    return rng.random(100)


ENGINES = [("D01_Dirichlet", D01_Dirichlet), ("D02_JamesStein", D02_JamesStein),
           ("D03_JelinekJoint", D03_JelinekJoint), ("D04_RecencyML", D04_RecencyML),
           ("D05_MarkovShrink", D05_MarkovShrink), ("D06_CrossPrize", D06_CrossPrize),
           ("D07_Hierarchy", D07_Hierarchy), ("D08_DigitIndep", D08_DigitIndep),
           ("D09_KernelSmooth", D09_KernelSmooth), ("D10_MinVariance", D10_MinVariance),
           ("D11_GapHazard", D11_GapHazard), ("D12_Uniform", D12_Uniform)]
TEN_ENGINE = [n for n, _ in ENGINES]
LOAI = {"D01_Dirichlet": "Bayes", "D02_JamesStein": "co ngót",
        "D03_JelinekJoint": "nội suy", "D04_RecencyML": "thời gian",
        "D05_MarkovShrink": "Markov", "D06_CrossPrize": "gộp giải",
        "D07_Hierarchy": "phân tầng", "D08_DigitIndep": "độc lập",
        "D09_KernelSmooth": "làm mượt", "D10_MinVariance": "ổn định",
        "D11_GapHazard": "ĐỐI CHỨNG", "D12_Uniform": "ĐỐI CHỨNG"}


# ==============================================================================
#  CHẤM ĐIỂM & CHỌN BỘ SỐ
# ==============================================================================

def cham_diem(mt, pool, k=SO_CON, cua_so_gan=CUA_SO_GAN, min_train=MIN_TRAIN,
              seed=2026):
    """Walk-forward: tại mỗi kỳ t, engine chỉ nhìn dữ liệu < t."""
    n = len(mt)
    mtr = min(min_train, max(10, n // 3))
    rng = np.random.default_rng(seed)
    hit = np.zeros((len(ENGINES), n - mtr), dtype=bool)
    for i, t in enumerate(range(mtr, n)):
        for j, (_, f) in enumerate(ENGINES):
            try:
                sel = np.argsort(-f(mt[:t], pool[:t], rng), kind="stable")[:k]
            except Exception:
                sel = np.arange(k)
            hit[j, i] = mt[t] in set(int(x) for x in sel)
    n_test = hit.shape[1]
    g = min(cua_so_gan, n_test)
    rng2 = np.random.default_rng(seed + 1)
    ra = []
    for j, (ten, f) in enumerate(ENGINES):
        try:
            diem = f(mt, pool, rng2)
        except Exception:
            diem = np.zeros(100)
        ra.append({"ten": ten, "loai": LOAI[ten],
                   "hr_all": float(hit[j].mean()), "hr_gan": float(hit[j, -g:].mean()),
                   "diem_mang": diem,
                   "diem": float(hit[j].mean() + W_GAN * hit[j, -g:].mean())})
    ra.sort(key=lambda x: -x["diem"])
    return ra, n_test, g, hit


def chon_so(bang, k=SO_CON, n_eng=SO_ENGINE):
    """Gộp điểm z của n_eng engine hàng đầu, lấy top k ô."""
    tong = np.zeros(100)
    dung = bang[:n_eng]
    for e in dung:
        tong += _z(e["diem_mang"])
    sel = np.argsort(-tong, kind="stable")[:k]
    return sorted(int(x) for x in sel), dung


def null_chon(n_test, g, k=SO_CON, n_eng=SO_ENGINE, n_lap=1500, seed=7):
    """Nếu CẢ 12 engine đều vô dụng (64%), n_eng engine top đạt bao nhiêu?"""
    rng = np.random.default_rng(seed)
    p = k / 100.0
    h = rng.random((n_lap, len(ENGINES), n_test)) < p
    diem = h.mean(axis=2) + W_GAN * h[:, :, -g:].mean(axis=2)
    top = np.argsort(-diem, axis=1)[:, :n_eng]
    return np.array([h[i, top[i], -g:].mean() for i in range(n_lap)])


def _vi_tri_g8(tg):
    return len(tg[0]) - 1 if (len(tg[0]) == 18 and len(tg[0][-1]) == 2) else None


def chay_dai(stt, ngay_moc, so_ky=None, k=None):
    so_ky = so_ky or SO_KY
    k = k or SO_CON
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
    if len(tg) < 35:
        raise RuntimeError(f"chỉ còn {len(tg)} kỳ, cần >=35")

    pool = [[int(s[-2:]) for s in ky] for ky in tg]
    vt_g8 = _vi_tri_g8(tg)
    ds = [("DB", CHAY_DB, 0), ("G1", CHAY_G1, 1)]
    if CHAY_G8 and vt_g8 is not None:
        ds.append(("G8", True, vt_g8))

    mods = []
    for khoa, bat, vi in ds:
        if not bat:
            continue
        mt = [int(ky[vi][-2:]) for ky in tg]
        bang, n_test, g, hit = cham_diem(mt, pool, k)
        bo, dung = chon_so(bang, k)
        nb = null_chon(n_test, g, k)
        hr = float(np.mean([e["hr_gan"] for e in dung]))
        mods.append({
            "key": khoa, "ten": MO_TA[khoa][0], "mo_ta": MO_TA[khoa][1],
            "so": [f"{v:02d}" for v in bo], "bang": bang, "dung": dung,
            "n_test": n_test, "g": g, "hr_gan": hr, "null_tb": float(nb.mean()),
            "p_value": float((nb >= hr).mean()),
            "moc": k / 100.0, "hoa_von": k / TY_LE_TRA, "von": k,
        })
    return {"stt": stt, "dai": ten, "mien": mien, "nguon": nguon, "n_ky": len(tg),
            "ngay_ky_truoc": ng[-1].strftime("%d/%m/%Y"), "db_ky_truoc": tg[-1][0],
            "g1_ky_truoc": tg[-1][1],
            "g8_ky_truoc": tg[-1][vt_g8] if vt_g8 is not None else None,
            "modules": mods, "ket_qua_that": that}


# ==============================================================================
#  CHẾ ĐỘ BAO LÔ — cùng 12 engine, đổi MỤC TIÊU sang 18 lô
#
#  Khác chế độ đề: mục tiêu không phải 1 số mà là TẬP 18 lô.
#  Thước đo = SỐ CON TRÚNG (không phải "có trúng hay không" — với 20 con
#  P(trúng >=1) = 98,2%, gần như luôn trúng kể cả bốc bừa).
# ==============================================================================

SO_CON_LO    = 20     # số con bao lô
SO_ENGINE_LO = 4


def cham_diem_lo(pool, k=SO_CON_LO, cua_so_gan=CUA_SO_GAN,
                 min_train=MIN_TRAIN, seed=2026):
    """Engine học trên chuỗi TOÀN BỘ lô, chấm bằng SỐ CON TRÚNG mỗi kỳ."""
    n = len(pool)
    mtr = min(min_train, max(10, n // 3))
    rng = np.random.default_rng(seed)
    hit = np.zeros((len(ENGINES), n - mtr))
    for i, t in enumerate(range(mtr, n)):
        thuc = set(pool[t])
        mt_phang = [v for ky in pool[:t] for v in ky]
        for j, (_, f) in enumerate(ENGINES):
            try:
                sel = np.argsort(-f(mt_phang, pool[:t], rng), kind="stable")[:k]
            except Exception:
                sel = np.arange(k)
            hit[j, i] = sum(1 for v in sel if int(v) in thuc)
    n_test = hit.shape[1]
    g = min(cua_so_gan, n_test)
    rng2 = np.random.default_rng(seed + 1)
    mt_all = [v for ky in pool for v in ky]
    ra = []
    for j, (ten, f) in enumerate(ENGINES):
        try:
            diem = f(mt_all, pool, rng2)
        except Exception:
            diem = np.zeros(100)
        ra.append({"ten": ten, "loai": LOAI[ten],
                   "hr_all": float(hit[j].mean() / k), "hr_gan": float(hit[j, -g:].mean() / k),
                   "tb_all": float(hit[j].mean()), "tb_gan": float(hit[j, -g:].mean()),
                   "diem_mang": diem,
                   "diem": float(hit[j].mean() + W_GAN * hit[j, -g:].mean())})
    ra.sort(key=lambda x: -x["diem"])
    return ra, n_test, g, hit


def null_chon_lo(n_test, g, k=SO_CON_LO, n_eng=SO_ENGINE_LO, n_lap=1200, seed=7):
    rng = np.random.default_rng(seed)
    p = 1 - (1 - 0.01) ** 18
    h = rng.random((n_lap, len(ENGINES), n_test, k)) < p
    hr_all = h.mean(axis=(2, 3)); hr_gan = h[:, :, -g:, :].mean(axis=(2, 3))
    top = np.argsort(-(hr_all + W_GAN * hr_gan), axis=1)[:, :n_eng]
    return np.array([hr_gan[i, top[i]].mean() for i in range(n_lap)])


def chay_dai_lo(stt, ngay_moc, so_ky=None, k=None):
    so_ky = so_ky or SO_KY
    k = k or SO_CON_LO
    ten, ma, mien, nd = E.lay_dai(stt)
    if ma == "xsmb":
        raise RuntimeError("bỏ qua Miền Bắc (27 lô)")
    m = E.lay_tu_master(stt, so_ky + 20)
    if m:
        toan_giai, ngay_full, info = m; nguon = "KHO"
    else:
        _, ngay, ngay_full, toan_giai, info = E.lay_du_lieu(stt, min(200, so_ky + 30))
        nguon = "WEB"
    if not toan_giai:
        raise RuntimeError("không bóc được toàn bảng giải")
    tg, ng, that = E.cat_truoc_ngay(toan_giai, ngay_full, ngay_moc, so_ky)
    if len(tg) < 35 or len(tg[0]) != 18:
        raise RuntimeError(f"{len(tg)} kỳ / {len(tg[0])} giải — cần >=35 kỳ và 18 giải")

    pool = [[int(s[-2:]) for s in ky] for ky in tg]
    n_lo = len(pool[0])
    bang, n_test, g, hit = cham_diem_lo(pool, k)
    bo, dung = chon_so(bang, k, SO_ENGINE_LO)
    nb = null_chon_lo(n_test, g, k)
    hr = float(np.mean([e["hr_gan"] for e in dung]))
    von = k * n_lo
    return {"stt": stt, "dai": ten, "mien": mien, "nguon": nguon, "n_ky": len(tg),
            "n_lo": n_lo, "ngay_ky_truoc": ng[-1].strftime("%d/%m/%Y"),
            "db_ky_truoc": tg[-1][0], "ket_qua_that": that,
            "so": [f"{v:02d}" for v in bo], "bang": bang, "dung": dung,
            "n_test": n_test, "g": g, "hr_gan": hr, "null_tb": float(nb.mean()),
            "p_value": float((nb >= hr).mean()),
            "von": von, "trung_tb": n_lo * k / 100.0, "hoa_von": von / TY_LE_TRA}


# ==============================================================================
#  EMAIL + PIPELINE
# ==============================================================================

def _bang_html(bang, dung, n_test, g, don_vi="%"):
    d = {e["ten"] for e in dung}
    h = ['<details style="margin:8px 0 0"><summary style="font-size:12px;color:#546e7a;'
         'cursor:pointer">Bảng 12 engine</summary>'
         '<table style="font-size:11px;border-collapse:collapse;margin-top:5px">'
         '<tr style="background:#eceff1"><th style="padding:3px 7px">#</th>'
         '<th style="padding:3px 7px;text-align:left">Engine</th>'
         '<th style="padding:3px 7px">Loại</th>'
         f'<th style="padding:3px 7px">{n_test} kỳ</th>'
         f'<th style="padding:3px 7px">{g} kỳ gần</th></tr>']
    for i, e in enumerate(bang, 1):
        bg = "#e8f5e9" if e["ten"] in d else ("#ffebee" if "ĐỐI CHỨNG" in e["loai"] else "#fff")
        h.append(f'<tr style="background:{bg}"><td style="padding:3px 7px">{i}</td>'
                 f'<td style="padding:3px 7px">{e["ten"]}</td>'
                 f'<td style="padding:3px 7px;text-align:center;color:#90a4ae">{e["loai"]}</td>'
                 f'<td style="padding:3px 7px;text-align:center">{e["hr_all"]:.0%}</td>'
                 f'<td style="padding:3px 7px;text-align:center"><b>{e["hr_gan"]:.0%}</b></td></tr>')
    h.append('</table><div style="font-size:11px;color:#b71c1c;margin-top:4px">'
             'Dòng đỏ = ĐỐI CHỨNG (D11 hazard phẳng, D12 bốc ngẫu nhiên). '
             'Nếu đối chứng lọt vào nhóm dẫn đầu, thứ hạng là NHIỄU.</div></details>')
    return "".join(h)


def _html(ket, ngay_moc, loi, che_do):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    thu = E.THU_VN[ngay_moc.weekday()]
    la_lo = che_do == "LO"
    h = [f'<div style="{css}max-width:760px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">'
             f'{SO_CON_LO if la_lo else SO_CON} con — {thu} {ngay_moc:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài · 12 engine THỐNG KÊ (không có quy tắc dân gian) · '
             f'{SO_KY} kỳ</p>')
    h.append('<div style="background:#fff8e1;border-left:4px solid #f9a825;'
             'padding:10px 13px;margin:0 0 22px;font-size:13px;line-height:1.6">'
             '<b>Hai engine ĐỐI CHỨNG cố ý trong bảng:</b><br>'
             '· <b>D11_GapHazard</b> — hazard phân phối hình học PHẲNG, giá trị lý '
             'thuyết bằng 0<br>'
             '· <b>D12_Uniform</b> — bốc ngẫu nhiên hoàn toàn<br>'
             'Nếu hai cái này lọt vào nhóm dẫn đầu, nghĩa là thứ hạng engine hôm đó '
             'là NHIỄU, không phải kỹ năng. Đây là phép thử trung thực nhất.</div>')

    for r in ket:
        h.append(f'<div style="margin:26px 0 0;padding:9px 12px;background:#263238;'
                 f'color:#fff;border-radius:5px 5px 0 0">'
                 f'<span style="font-size:17px;font-weight:700">{r["dai"].upper()}</span>'
                 f'<span style="font-size:13px;opacity:.85"> &nbsp;|&nbsp; {r["mien"]}'
                 f' &nbsp;|&nbsp; {thu} {ngay_moc:%d.%m.%Y}</span><br>'
                 f'<span style="font-size:12px;opacity:.7">Kỳ gần nhất {r["ngay_ky_truoc"]}'
                 f' · ĐB {r["db_ky_truoc"]} · {r["n_ky"]} kỳ</span></div>')
        h.append('<div style="border:1px solid #cfd8dc;border-top:0;'
                 'border-radius:0 0 5px 5px;padding:10px 12px 14px">')
        muc = [r] if la_lo else r["modules"]
        for m in muc:
            if not la_lo:
                h.append(f'<div style="font-size:14px;font-weight:600;margin-top:10px">'
                         f'{m["ten"]}</div>'
                         f'<div style="font-size:12px;color:#607d8b;margin:2px 0 5px">'
                         f'{r["dai"]} · {thu} {ngay_moc:%d.%m.%Y} · {m["mo_ta"]}</div>')
            mau = "#2e7d32" if m["p_value"] < .05 else "#c62828"
            h.append(f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:15px;background:#e8f5e9;border-left:4px solid #2e7d32;'
                     f'padding:11px 13px;word-break:break-all;line-height:1.9">'
                     f'{",".join(m["so"])}</div>')
            h.append(f'<div style="font-size:12px;color:#546e7a;margin:6px 0 0">'
                     f'Gộp từ {len(m["dung"])} engine: '
                     + " · ".join(f'{e["ten"]} ({e["hr_gan"]:.0%})' for e in m["dung"])
                     + '</div>')
            h.append(f'<div style="font-size:12px;margin:3px 0 0;color:{mau}">'
                     f'Đạt <b>{m["hr_gan"]:.1%}</b> trong {m["g"]} kỳ gần · '
                     f'ảo giác chọn lọc {m["null_tb"]:.1%} · <b>p = {m["p_value"]:.3f}</b> — '
                     f'{"VƯỢT ảo giác" if m["p_value"] < .05 else "chưa hơn ảo giác"} · '
                     f'mốc {m["moc"]:.0%} · hoà vốn {m["hoa_von"]:.2%}'
                     if not la_lo else
                     f'Đạt <b>{m["hr_gan"]:.1%}</b>/con trong {m["g"]} kỳ gần · '
                     f'ảo giác {m["null_tb"]:.1%} · <b>p = {m["p_value"]:.3f}</b> · '
                     f'vốn {m["von"]:,} · trúng TB {m["trung_tb"]:.2f} · '
                     f'hoà vốn cần {m["hoa_von"]:.2f}')
            h.append('</div>')
            h.append(_bang_html(m["bang"], m["dung"], m["n_test"], m["g"]))
        h.append('</div>')

    if loi:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;'
                 'padding:10px 13px;margin:18px 0;font-size:13px"><b>Đài lỗi:</b><ul>')
        for s_, t_, e_ in loi:
            h.append(f'<li>[{s_}] {t_}: {e_}</li>')
        h.append('</ul></div>')
    h.append('<hr style="margin:26px 0 10px;border:0;border-top:1px solid #ddd">'
             '<p style="font-size:12px;color:#888;line-height:1.6">'
             '12 engine đều là ước lượng thống kê của phân phối trên 100 ô, khác nhau ở '
             'GIẢ ĐỊNH (Bayes · co ngót · nội suy · Markov · phân tầng · làm mượt...). '
             'Không có quy tắc dân gian nào.<br>'
             'EV = 95/100 − 1 = −5,0%, cố định bởi tỷ lệ trả. Engine chỉ đổi CHỌN Ô NÀO, '
             'không đổi SỐ Ô — nên không đổi kỳ vọng.</p></div>')
    return "".join(h)


def gui_email(ket, ngay_moc, loi, che_do):
    mk = E._lay_mat_khau()
    thu = E.THU_VN[ngay_moc.weekday()]
    n = SO_CON_LO if che_do == "LO" else SO_CON
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[{n} CON {'BAO LÔ' if che_do=='LO' else 'ĐỀ'}] {thu} {ngay_moc:%d.%m.%Y} — {len(ket)} đài"
    msg["From"] = formataddr((f"XSMN {n} Con", EMAIL_GUI)); msg["To"] = EMAIL_NHAN
    t = [f"{n} CON — {thu} {ngay_moc:%d.%m.%Y}", ""]
    for r in ket:
        t.append("=" * 60); t.append(f"{r['dai'].upper()} | {r['mien']}")
        for m in ([r] if che_do == "LO" else r["modules"]):
            t.append("")
            if che_do != "LO":
                t.append(m["ten"])
            t.append(f"(p {m['p_value']:.3f} | {m['hr_gan']:.1%} vs ảo giác {m['null_tb']:.1%})")
            t.append(",".join(m["so"]))
        t.append("")
    msg.attach(MIMEText("\n".join(t), "plain", "utf-8"))
    msg.attach(MIMEText(_html(ket, ngay_moc, loi, che_do), "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk); sv.send_message(msg)
    print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")


def main(ngay=None, che_do="DE", so_ky=None, gui_mail=True):
    """che_do: 'DE' = 64 con cho ĐB/G1/G8 | 'LO' = 20 con bao lô (MN/MT)"""
    ngay_moc = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    so_ky = so_ky or SO_KY
    thu = E.THU_VN[ngay_moc.weekday()]
    la_lo = che_do.upper() == "LO"
    k = SO_CON_LO if la_lo else SO_CON

    print("=" * 80)
    print(f"  {k} CON {'BAO LÔ' if la_lo else 'ĐỀ'} — 12 ENGINE THỐNG KÊ  |  "
          f"{thu} {ngay_moc:%d.%m.%Y}")
    print(f"  {so_ky} kỳ · gộp {SO_ENGINE_LO if la_lo else SO_ENGINE} engine hàng đầu")
    if la_lo:
        print(f"  CHỈ Miền Nam + Miền Trung (18 lô)")
    print("=" * 80)

    dsach = E.dai_theo_ngay(ngay_moc)
    if la_lo:
        dsach = [s for s in dsach if E.lay_dai(s)[1] != "xsmb"]
    lich = E.xay_lich()
    print(f"\n  Đài hôm nay: {len(dsach)}")

    ket, loi = [], []
    for s in dsach:
        ten = lich.get(str(s), {}).get("ten", E.lay_dai(s)[0])
        try:
            r = chay_dai_lo(s, ngay_moc, so_ky, k) if la_lo else chay_dai(s, ngay_moc, so_ky, k)
            ket.append(r)
            p = r["p_value"] if la_lo else r["modules"][0]["p_value"]
            print(f"     ✓ [{s:>2}] {ten:<20} {r['n_ky']} kỳ ({r['nguon']}) | p={p:.2f}")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<20} LỖI: {e}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")

    # --- Bảng gộp engine + kiểm tra đối chứng ---
    print(f"\n[A] ENGINE NÀO MẠNH — gộp mọi đài hôm nay")
    print("-" * 80)
    gop = {t: [] for t in TEN_ENGINE}
    hang_dc = []
    for r in ket:
        for m in ([r] if la_lo else r["modules"]):
            for i, e in enumerate(m["bang"], 1):
                gop[e["ten"]].append(e["hr_all"])
                if "ĐỐI CHỨNG" in e["loai"]:
                    hang_dc.append(i)
    print(f"  {'Engine':<18}{'Loại':>11}{'Hit toàn kỳ':>14}")
    print("  " + "-" * 50)
    for t in sorted(TEN_ENGINE, key=lambda x: -np.mean(gop[x])):
        dau = "  ← ĐỐI CHỨNG" if "ĐỐI CHỨNG" in LOAI[t] else ""
        print(f"  {t:<18}{LOAI[t]:>11}{np.mean(gop[t]):>13.1%}{dau}")
    tb_dc = np.mean(hang_dc)
    print(f"\n  Hạng trung bình của 2 ĐỐI CHỨNG: {tb_dc:.1f}/12  (ngẫu nhiên = 6,5)")
    print("  → " + ("ĐỐI CHỨNG lọt nhóm dẫn đầu — thứ hạng engine là NHIỄU."
                    if tb_dc <= 6.5 else
                    "Đối chứng xếp dưới trung bình — có dấu hiệu engine thật hơn nhiễu."))

    for r in ket:
        for m in ([r] if la_lo else r["modules"]):
            print("\n" + "=" * 80)
            print(f"  {r['dai'].upper()}" + ("" if la_lo else f"  —  {m['ten']}"))
            print("=" * 80)
            print(f"\n  {','.join(m['so'])}\n")
            print(f"  Gộp từ: " + " · ".join(f"{e['ten']}({e['hr_gan']:.0%})" for e in m["dung"]))
            print(f"  Đạt {m['hr_gan']:.1%} trong {m['g']} kỳ gần | ảo giác {m['null_tb']:.1%} "
                  f"| p = {m['p_value']:.3f}")
            print(f"\n  {'#':>3} {'Engine':<18}{'Loại':>11}{'toàn kỳ':>9}{'gần':>8}")
            print("  " + "-" * 52)
            d = {e["ten"] for e in m["dung"]}
            for i, e in enumerate(m["bang"], 1):
                dau = " ←" if e["ten"] in d else (" ⚠ĐC" if "ĐỐI CHỨNG" in e["loai"] else "")
                print(f"  {i:>3} {e['ten']:<18}{e['loai']:>11}{e['hr_all']:>8.0%}"
                      f"{e['hr_gan']:>8.0%}{dau}")
    print("\n" + "=" * 80)

    if gui_mail:
        print("\n  Đang gửi email...")
        try:
            gui_email(ket, ngay_moc, loi, che_do.upper())
        except Exception as e:
            print(f"  ✗ Không gửi được email: {e}")
    return ket
