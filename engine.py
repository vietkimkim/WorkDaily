# ==============================================================================
#  XSMN AUTO ENGINE v8-GH  —  BẢN CHẠY TRÊN GITHUB ACTIONS
#
#  Khác bản Colab ở 4 điểm:
#   • Mật khẩu email đọc từ BIẾN MÔI TRƯỜNG (GitHub Secrets), không ghi trong code
#   • Không hỏi mật khẩu tương tác — Actions không có bàn phím, sẽ báo lỗi rõ ràng
#   • Kho dữ liệu + null cache nằm ngay trong repo, được commit lại sau mỗi lần chạy
#   • Chạy tự động 06:00 giờ Việt Nam mỗi ngày, không cần mở trình duyệt
#  Chọn đài bằng SỐ THỨ TỰ → tự lấy dữ liệu → chạy 5 tầng → trả về N cặp số
#
#  Nguồn: xskt.com.vn/{ma_dai}/{n}-ngay
#  BA MODULE, cùng không gian 100 (00-99), cùng tỷ lệ trả 95:
#     ① ĐỀ ĐẶC BIỆT  : 2 số cuối giải ĐB       —  1 mục tiêu/kỳ
#     ② ĐỀ GIẢI NHẤT : 2 số cuối giải Nhất     —  1 mục tiêu/kỳ
#     ③ BAO LÔ 2 SỐ  : 2 số cuối của MỌI giải  — 18 mục tiêu/kỳ (MB: 27)
#
#  NÂNG CẤP so với v6:
#   • 9 tín hiệu (v6 có 6). Ba tín hiệu mới khai thác TOÀN BẢNG GIẢI:
#       S7 Toàn giải   - tần suất từ 1.800 quan sát thay vì ~500
#       S8 Bigram JM   - P(b|a) nội suy Jelinek-Mercer, lambda tự ước lượng
#       S9 James-Stein - đếm trực tiếp 100 ô + co ngót Bayes thực nghiệm
#   • SỬA LỖI null theo họ: v6 dùng max(hit rate) - SAI khi các module có mốc
#     ngẫu nhiên khác nhau (50% vs 10%). v7 dùng MIN-P: quy về p-value riêng
#     của từng module rồi lấy p nhỏ nhất. Đây là phép hiệu chỉnh đúng.
#   • Bảng LÃI/LỖ TỔNG HỢP khi bật >=2 module + cảnh báo P(ít nhất 1 module trúng)
#   • Chi-square lần đầu HỢP LỆ ở module ③ (tần số kỳ vọng 18 >= 5)

# ==============================================================================

import re, os, time, math, json, hashlib, unicodedata
from datetime import date
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats
from itertools import product

# ==============================================================================
#  ┌────────────────────────────────────────────────────────────────────────┐
#  │  [1] BẢNG ĐIỀU KHIỂN  —  CHỈ SỬA 3 DÒNG NÀY                            │
#  └────────────────────────────────────────────────────────────────────────┘
# ==============================================================================

# ①②③ Số con mỗi module. Đặt 0 để tắt module đó.
SO_KY       = 150            # số kỳ dữ liệu huấn luyện mỗi đài
SO_CON_DB   = 64             # ① ĐỀ ĐẶC BIỆT : 2 số cuối giải ĐB        (0 = tắt)
SO_CON_LO2  = 0              # ② BAO LÔ 2 SỐ : 2 số cuối của MỌI giải   (0 = tắt)
SO_CON_3SO  = 50             # ③ LÔ 3 SỐ     : 3 số cuối của mọi giải   (0 = tắt)
SO_CON_4SO  = 0              # ④ LÔ 4 SỐ     : 4 số cuối của mọi giải   (0 = tắt)
                             #    ⚠ EV -11,12% — TỆ NHẤT. Mật độ 0,30 mẫu/ô.

# Số ĐIỂM cược mỗi con — quyết định vốn thật
DIEM_DB     = 5              # ① 82 con × 5 điểm × 1 lô  = 410
DIEM_LO2    = 2              # ② 4 con × 2 điểm × số lô
DIEM_3SO    = 1              # ③ 50 con × 1 điểm × số lô
DIEM_4SO    = 1              # ④ điểm cược mỗi con

# Số LÔ tính tiền cho bao lô. Đặt None = dùng đúng số giải bóc được.
# Nhiều nơi KHÔNG tính G8 vào bao lô 2 số → khi đó MN/MT là 17, không phải 18.
SO_LO_2SO_MN = 17            # bao lô 2 số Miền Nam / Trung
SO_LO_2SO_MB = 27            # bao lô 2 số Miền Bắc
SO_LO_3SO_MN = 17            # bao lô 3 số Miền Nam / Trung
SO_LO_3SO_MB = 23            # bao lô 3 số Miền Bắc
SO_LO_4SO_MN = 16            # bao lô 4 số Miền Nam / Trung (giải >= 4 chữ số)
SO_LO_4SO_MB = 20            # bao lô 4 số Miền Bắc

TY_LE_TRA_2SO = 95.0         # thưởng con 2 số  (hoà vốn = 100)
TY_LE_TRA_3SO = 961.0        # thưởng con 3 số  (hoà vốn = 1000)
TY_LE_TRA_4SO = 8888.0       # thưởng con 4 số  (hoà vốn = 10000 → EV -11,12%)


HIEN_DONG_COPY = False   # True = in thêm dòng số phân cách bằng dấu phẩy dưới mỗi vùng

# ---- GỬI EMAIL (tuỳ chọn) ----
GUI_EMAIL      = True
EMAIL_NHAN     = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI      = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")
# MẬT KHẨU KHÔNG BAO GIỜ NẰM TRONG FILE NÀY. Repo công khai — bot quét GitHub
# sẽ tìm thấy trong vài phút. Nó đến từ GitHub Secrets qua biến môi trường.
EMAIL_MAT_KHAU = ""
TEN_SECRET     = "GMAIL_APP_PASSWORD"

HIEN_BANG_CHI_TIET = False   # True = in bảng điểm từng tín hiệu (in TRƯỚC bộ số)

# ---- Tham số nâng cao (thường không cần đụng) ----
ALPHA, AUX_WEIGHT  = 1.0, 0.35
DIVERSITY_CAP      = None          # None = tự tính theo số lượng cần trả về
BACKTEST_MIN_TRAIN = 10
# --- CHỌN BỘ TÍN HIỆU ---
# "GON"  : 5 tín hiệu có cơ chế thống kê rõ ràng -> 31 tổ hợp. KHUYẾN NGHỊ.
# "DAY"  : cả 9 tín hiệu -> 511 tổ hợp. Overfit gấp ~1,8 lần trên nhiễu.
# Đo trên dữ liệu ngẫu nhiên (không có quy luật), hit rate ảo so với mốc 20%:
#     9 tín hiệu -> +3,10%      5 tín hiệu -> +1,74%      (giảm 44% ảo giác)
BO_TIN_HIEU        = "GON"

# Tín hiệu bị loại khỏi bộ GON và lý do:
#   S3_Gan      - phân phối hình học KHÔNG NHỚ, giá trị thông tin lý thuyết = 0
#   S4_ChamTong - hàm của chính các chữ số, trùng lặp với S1/S7
#   S5_BongSo   - quy tắc dân gian, chỉ là hoán vị của S1, không có cơ chế
#   S6_CauTruc  - giả định "bù trừ" (mean reversion) — sai với các kỳ độc lập
TIN_HIEU_GON = ["S1_TanSuat", "S2_MarkovKy", "S7_ToanGiai",
                "S8_BigramJM", "S9_JamesStein",
                "S10_GanDay",      # trọng số giảm dần — bắt thiên lệch ĐANG diễn ra
                "S11_PhanTang",    # phân tầng nhóm giải — kiểm định giả định đồng nhất
                "S12_Cau"]         # cầu vị trí — chọn cầu CHỈ từ quá khứ

WEIGHT_GRID        = [0.0, 1.0]
WEIGHT_GRID_3      = [0.0, 1.0]            # 7 tín hiệu 3 số -> 2^7 = 127 tổ hợp
WEIGHT_GRID_4      = [0.0, 1.0]            # 7 tín hiệu 4 số -> 2^7 = 127 tổ hợp
LAMBDA_GRID        = [0.3, 0.5, 0.7, 0.9]  # lưới nội suy Jelinek-Mercer
N_NULL             = None          # None = tự tune theo SO_KY
MAX_BACKTEST_PTS   = None
SEED               = 20260821

rng_global = np.random.default_rng(SEED)
MIRROR = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
PAIRS  = np.array([(i//10, i%10) for i in range(100)])
P_D1, P_D2 = PAIRS[:,0], PAIRS[:,1]
THU_VN = ["Thứ 2","Thứ 3","Thứ 4","Thứ 5","Thứ 6","Thứ 7","Chủ nhật"]

# Không gian 3 chữ số: 000-999
TRIPLES = np.array([(i//100, (i//10)%10, i%10) for i in range(1000)])
T_D1, T_D2, T_D3 = TRIPLES[:,0], TRIPLES[:,1], TRIPLES[:,2]

# Cơ cấu giải: tên -> (số lượng, số chữ số). Dùng làm CHỮ KÝ kiểm tra parser.
CO_CAU_GIAI = {
    "MN": {"G8":(1,2), "G7":(1,3), "G6":(3,4), "G5":(1,4), "G4":(7,5),
           "G3":(2,5), "G2":(1,5), "G1":(1,5), "DB":(1,6)},     # tổng 18 số, 17 lô 3 số
    "MB": {"G7":(4,2), "G6":(3,3), "G5":(6,4), "G4":(4,4), "G3":(6,5),
           "G2":(2,5), "G1":(1,5), "DB":(1,5)},                 # tổng 27 số, 23 lô 3 số
}


# ==============================================================================
#  [2] DANH SÁCH 36 ĐÀI  —  mã lấy trực tiếp từ menu điều hướng xskt.com.vn
#      Số thứ tự cố định, không đổi giữa các lần chạy.
# ==============================================================================

DAI_LIST = [
    # STT 1–21: MIỀN NAM
    ("TP. Hồ Chí Minh",  "xshcm-xstp",  "Miền Nam"),   # 1
    ("An Giang",         "xsag",        "Miền Nam"),   # 2
    ("Bạc Liêu",         "xsbl",        "Miền Nam"),   # 3
    ("Bến Tre",          "xsbt",        "Miền Nam"),   # 4
    ("Bình Dương",       "xsbd",        "Miền Nam"),   # 5
    ("Bình Phước",       "xsbp",        "Miền Nam"),   # 6
    ("Bình Thuận",       "xsbth",       "Miền Nam"),   # 7
    ("Cà Mau",           "xscm",        "Miền Nam"),   # 8
    ("Cần Thơ",          "xsct",        "Miền Nam"),   # 9
    ("Đà Lạt (Lâm Đồng)","xsld-xsdl",   "Miền Nam"),   # 10
    ("Đồng Nai",         "xsdn",        "Miền Nam"),   # 11
    ("Đồng Tháp",        "xsdt",        "Miền Nam"),   # 12
    ("Hậu Giang",        "xshg",        "Miền Nam"),   # 13
    ("Kiên Giang",       "xskg",        "Miền Nam"),   # 14
    ("Long An",          "xsla",        "Miền Nam"),   # 15
    ("Sóc Trăng",        "xsst",        "Miền Nam"),   # 16
    ("Tây Ninh",         "xstn",        "Miền Nam"),   # 17
    ("Tiền Giang",       "xstg",        "Miền Nam"),   # 18
    ("Trà Vinh",         "xstv",        "Miền Nam"),   # 19
    ("Vĩnh Long",        "xsvl",        "Miền Nam"),   # 20
    ("Vũng Tàu",         "xsvt",        "Miền Nam"),   # 21
    # STT 22–35: MIỀN TRUNG
    ("Bình Định",        "xsbdi",       "Miền Trung"), # 22
    ("Đà Nẵng",          "xsdng-xsdna", "Miền Trung"), # 23
    ("Đắk Lắk",          "xsdlk",       "Miền Trung"), # 24
    ("Đắk Nông",         "xsdno",       "Miền Trung"), # 25
    ("Gia Lai",          "xsgl",        "Miền Trung"), # 26
    ("Khánh Hòa",        "xskh",        "Miền Trung"), # 27
    ("Kon Tum",          "xskt",        "Miền Trung"), # 28
    ("Ninh Thuận",       "xsnt",        "Miền Trung"), # 29
    ("Phú Yên",          "xspy",        "Miền Trung"), # 30
    ("Quảng Bình",       "xsqb",        "Miền Trung"), # 31
    ("Quảng Nam",        "xsqnm-xsqna", "Miền Trung"), # 32
    ("Quảng Ngãi",       "xsqng",       "Miền Trung"), # 33
    ("Quảng Trị",        "xsqt",        "Miền Trung"), # 34
    ("Thành phố Huế",    "xstth",       "Miền Trung"), # 35
    # STT 36: MIỀN BẮC
    ("Miền Bắc (XSMB)",  "xsmb",        "Miền Bắc"),   # 36
]

PAGE_SIZES = [10, 30, 90, 100, 200]


def in_danh_sach_dai():
    """In bảng 36 đài kèm số thứ tự. Chạy hàm này để tra cứu."""
    print("=" * 74)
    print("  DANH SÁCH 36 ĐÀI  —  v7.1 tự chọn đài theo NGÀY, bảng này để tra cứu")
    print("=" * 74)
    mien_hien = None
    for i, (ten, ma, mien) in enumerate(DAI_LIST, 1):
        if mien != mien_hien:
            print(f"\n  ── {mien.upper()} " + "─" * (60 - len(mien)))
            mien_hien = mien
        print(f"    {i:>2}. {ten:<22} [{ma}]")
    print("\n" + "=" * 74)
    print(f"  Tổng: {len(DAI_LIST)} đài  |  Miền Bắc (36) quay hàng ngày")
    print("=" * 74)


def lay_dai(stt):
    if not isinstance(stt, int) or not (1 <= stt <= len(DAI_LIST)):
        in_danh_sach_dai()
        raise ValueError(f"Số thứ tự đài phải là 1–{len(DAI_LIST)}, nhận được: {stt!r}")
    ten, ma, mien = DAI_LIST[stt - 1]
    so_cs = 5 if ma == "xsmb" else 6          # XSMB: giải ĐB 5 chữ số
    return ten, ma, mien, so_cs


# ==============================================================================
#  [3] SCRAPER  —  hai chiến lược bóc tách độc lập + đối chiếu chéo
# ==============================================================================

HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
           "Accept-Language": "vi-VN,vi;q=0.9"}


def bo_dau(s):
    s = unicodedata.normalize("NFD", str(s).lower().strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.replace("đ", "d")).strip()


def _tai_1_trang(ma_dai, size, retries=3):
    url = f"https://xskt.com.vn/{ma_dai}/{size}-ngay"
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text, url
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Không tải được {url}. Lỗi cuối: {last}")


def _boc_bang(html, nd):
    """Chiến lược A — cấu trúc: ô đầu hàng là 'ĐB', lấy số cùng hàng."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        if bo_dau(cells[0].get_text(" ", strip=True)) in ("db", "dac biet", "giai dac biet", "gdb"):
            for c in cells[1:]:
                m = re.search(rf"(?<!\d)(\d{{{nd}}})(?!\d)", c.get_text(" ", strip=True))
                if m:
                    out.append(m.group(1)); break
    return out


def _boc_text(html, nd):
    """Chiến lược B — văn bản: gỡ hết thẻ, bắt 'ĐB → n chữ số' theo thứ tự."""
    txt = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return re.findall(rf"ĐB[^\d]{{0,25}}(?<!\d)(\d{{{nd}}})(?!\d)", txt)


# Nhiều mẫu ngày, thử hết rồi lấy mẫu cho NHIỀU kết quả nhất.
_MAU_NGAY = [
    r"ngày\s+(\d{1,2})[/\-.](\d{1,2})[/\-.](?:20)?(\d{2})",   # ngày 14/08/2026
    r"ngày\s+(\d{1,2})[/\-.](\d{1,2})(?![/\-.\d])",           # ngày 14/08
    r"XS[A-ZĐ]{2,8}\s+(\d{1,2})[/\-.](\d{1,2})[/\-.](?:20)?(\d{2})",
    r"XS[A-ZĐ]{2,8}\s+(\d{1,2})[/\-.](\d{1,2})(?![/\-.\d])",
    r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.]20(\d{2})(?!\d)",  # 14/08/2026 bất kỳ
]


def _boc_ngay(html, n):
    """Trả list chuỗi ngày theo thứ tự MỚI -> CŨ.

    QUAN TRỌNG: bản cũ dùng `if len(ds) >= n` — tìm được 189 ngày mà cần 190 thì
    VỨT SẠCH cả 189 rồi trả K1..K190. Một đài chỉ cần lệch 1 ngày là mất toàn bộ
    thông tin thời gian, và cả đài bị loại. Bản này trả về TẤT CẢ những gì tìm
    được, chỉ chèn nhãn tạm cho phần đuôi còn thiếu.
    """
    txt = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    tot = []
    for pat in _MAU_NGAY:
        ds, seen = [], set()
        for m in re.finditer(pat, txt):
            g = m.groups()
            key = f"{int(g[0]):02d}/{int(g[1]):02d}" + (f"/{g[2]}" if len(g) > 2 and g[2] else "")
            if key not in seen:
                seen.add(key); ds.append(key)
        if len(ds) > len(tot):
            tot = ds
        if len(tot) >= n:
            break
    if len(tot) >= n:
        return tot[:n]
    # Thiếu bao nhiêu thì chèn nhãn tạm ở ĐUÔI (kỳ cũ nhất), giữ nguyên phần đọc được
    if tot:
        print(f"     ⚠ chỉ đọc được {len(tot)}/{n} ngày mở thưởng — "
              f"{n-len(tot)} kỳ cũ nhất sẽ bị bỏ qua")
    return tot + [f"K{i+1}" for i in range(n - len(tot))]


def _gan_nam(dd_mm_moi_den_cu):
    """Trang nguồn chỉ ghi dd/mm. Suy ra năm bằng 2 ràng buộc:
       (1) kỳ mới nhất không thể nằm ở tương lai;
       (2) đi ngược thời gian thì mỗi kỳ phải cũ hơn kỳ liền trước."""
    homnay = date.today(); out = []; truoc = None
    for s in dd_mm_moi_den_cu:
        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2}))?", str(s))
        if not m:
            out.append(None); continue
        d, mo = int(m.group(1)), int(m.group(2))
        if m.group(3):                      # đã có năm -> dùng luôn, khỏi suy
            try:
                cand = date(2000 + int(m.group(3)), mo, d)
                out.append(cand); truoc = cand; continue
            except ValueError:
                out.append(None); continue
        y = truoc.year if truoc else homnay.year
        cand = None
        for _ in range(3):
            try: cand = date(y, mo, d)
            except ValueError: cand = None; break
            if cand > homnay or (truoc and cand >= truoc): y -= 1; cand = None
            else: break
        out.append(cand)
        if cand: truoc = cand
    return out


# Thứ tự chuẩn hoá: ĐB LUÔN đứng đầu (tín hiệu Markov liên kỳ dùng ĐB làm trạng thái)
THU_TU_GIAI = ["DB", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]

NHAN_GIAI = {"db":"DB","dac biet":"DB","giai dac biet":"DB","gdb":"DB",
             "g1":"G1","giai 1":"G1","giai nhat":"G1",
             "g2":"G2","giai 2":"G2","giai nhi":"G2",
             "g3":"G3","giai 3":"G3","giai ba":"G3",
             "g4":"G4","giai 4":"G4","giai tu":"G4",
             "g5":"G5","giai 5":"G5","giai nam":"G5",
             "g6":"G6","giai 6":"G6","giai sau":"G6",
             "g7":"G7","giai 7":"G7","giai bay":"G7",
             "g8":"G8","giai 8":"G8","giai tam":"G8"}


def _so_trong_o(txt):
    """Lấy các số 2-6 chữ số trong 1 ô. Bỏ ô chứa dấu '/' (ngày tháng)."""
    if "/" in txt or "%" in txt:
        return []
    return re.findall(r"(?<!\d)(\d{2,6})(?!\d)", txt)


def _boc_toan_giai_A(html, co_cau):
    """Chiến lược A — theo NHÃN GIẢI: đọc từng hàng, map nhãn -> danh sách số,
       rồi đối chiếu với cơ cấu giải (số lượng + số chữ số của từng giải)."""
    soup = BeautifulSoup(html, "html.parser")
    ket = []
    for tb in soup.find_all("table"):
        got = {}
        for tr in tb.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            nhan = NHAN_GIAI.get(bo_dau(cells[0].get_text(" ", strip=True)))
            if not nhan or nhan not in co_cau:
                continue
            sl, nd = co_cau[nhan]
            nums = [x for c in cells[1:] for x in _so_trong_o(c.get_text(" ", strip=True))
                    if len(x) == nd]
            if len(nums) == sl:
                got[nhan] = nums
        if len(got) == len(co_cau):                      # đủ mọi giải -> hợp lệ
            ket.append([x for g in THU_TU_GIAI if g in got for x in got[g]])
    return ket


def _boc_toan_giai_B(html, co_cau):
    """Chiến lược B — theo CHỮ KÝ ĐỘ DÀI: gom mọi số trong 1 bảng, chỉ nhận bảng
       có phân bố độ dài khớp chính xác cơ cấu giải. Không phụ thuộc nhãn."""
    chu_ky = {}
    for sl, nd in co_cau.values():
        chu_ky[nd] = chu_ky.get(nd, 0) + sl
    tong = sum(chu_ky.values())
    soup = BeautifulSoup(html, "html.parser")
    ket = []
    for tb in soup.find_all("table"):
        nums = [x for c in tb.find_all(["td", "th"])
                  for x in _so_trong_o(c.get_text(" ", strip=True))]
        if len(nums) != tong:
            continue
        dem = {}
        for x in nums:
            dem[len(x)] = dem.get(len(x), 0) + 1
        if dem == chu_ky:
            ket.append(nums)
    return ket


def lo3_cua_ky(so_list):
    """3 chữ số cuối của mọi giải >= 3 chữ số. ĐB đứng đầu (trạng thái Markov T3)."""
    return [s[-3:] for s in so_list if len(s) >= 3]


def tach_giai(toan_giai, vi_tri):
    """Tách 1 luồng số theo vị trí trong THU_TU_GIAI: 0 = ĐB, 1 = G1."""
    return [k[vi_tri] for k in toan_giai]


def _phat_hien_lich_quay(html):
    """Suy ra thứ mở thưởng từ chính trang nguồn (thay vì hard-code)."""
    txt = bo_dau(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
    chuan = {"hai":"Thứ 2","2":"Thứ 2","ba":"Thứ 3","3":"Thứ 3","tu":"Thứ 4","4":"Thứ 4",
             "nam":"Thứ 5","5":"Thứ 5","sau":"Thứ 6","6":"Thứ 6","bay":"Thứ 7","7":"Thứ 7"}
    dem = {}
    for m in re.findall(r"thu (hai|ba|tu|nam|sau|bay|[2-7])\b", txt):
        k = chuan.get(m)
        if k: dem[k] = dem.get(k, 0) + 1
    for _ in re.findall(r"chu nhat", txt):
        dem["Chủ nhật"] = dem.get("Chủ nhật", 0) + 1
    return max(dem, key=dem.get) if dem else "không xác định"


def lay_du_lieu(stt, n_can):
    """Trả (db_list, ngay, ngay_full, toan_giai, info) — tất cả CŨ NHẤT -> MỚI NHẤT.
       toan_giai = None nếu không bóc được đủ bảng giải (module 3 số sẽ tự tắt)."""
    ten, ma, mien, nd = lay_dai(stt)
    co_cau = CO_CAU_GIAI["MB" if ma == "xsmb" else "MN"]
    db = ngay = tg = None
    url = ""; info = {}

    for size in [s for s in PAGE_SIZES if s >= n_can] or [PAGE_SIZES[-1]]:
        html, url = _tai_1_trang(ma, size)

        # --- Ưu tiên: bóc TOÀN BỘ bảng giải, đối chiếu 2 chiến lược ---
        FA = _boc_toan_giai_A(html, co_cau)          # theo nhãn giải
        FB = _boc_toan_giai_B(html, co_cau)          # theo chữ ký độ dài
        khop3 = (len(FA) == len(FB)
                 and all(sorted(x) == sorted(y) for x, y in zip(FA, FB)))

        # --- Dự phòng: chỉ bóc dòng ĐB (như v4) ---
        A, B = _boc_bang(html, nd), _boc_text(html, nd)
        kq = A if len(A) >= len(B) else B

        info = {"n_A": len(A), "n_B": len(B),
                "khop": len(A) == len(B) and all(x == y for x, y in zip(A, B)),
                "n_FA": len(FA), "n_FB": len(FB), "khop3": khop3,
                "page_size": size, "lich": _phat_hien_lich_quay(html)}

        if len(FA) >= n_can:                          # có đủ toàn giải -> dùng
            tg = FA[:n_can]
            db = [k[0] for k in tg]
            ngay = _boc_ngay(html, len(FA))[:n_can]
            break
        if len(kq) >= n_can:                          # chỉ có ĐB
            tg = FA if FA else None
            db = kq[:n_can]
            ngay = _boc_ngay(html, len(kq))[:n_can]
            break
        db, ngay, tg = kq, _boc_ngay(html, len(kq)), (FA or None)

    if not db:
        raise RuntimeError(f"Không bóc được kết quả nào từ {url}. Dùng chế độ nhập tay.")
    if len(db) < n_can:
        print(f"  ⚠ Chỉ có {len(db)} kỳ (yêu cầu {n_can}) — dùng {len(db)} kỳ.")
        ngay = ngay[:len(db)]
    if tg is not None and len(tg) != len(db):
        tg = tg[:len(db)] if len(tg) > len(db) else None

    ngay_full = _gan_nam(ngay)
    info.update({"ten": ten, "ma": ma, "mien": mien, "n_digits": nd, "url": url,
                 "khu": "MB" if ma == "xsmb" else "MN"})
    return (db[::-1], ngay[::-1], ngay_full[::-1],
            (tg[::-1] if tg else None), info)



# ==============================================================================
#  [4] TIỆN ÍCH
# ==============================================================================

def zscore(v):
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-12 else (v - v.mean()) / sd

def wilson_ci(k, n, conf=.95):
    if n == 0: return (0., 1.)
    z = stats.norm.ppf(1 - (1-conf)/2); p, den = k/n, 1 + z**2/n
    ctr = (p + z**2/(2*n)) / den
    hw  = z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / den
    return (max(0., ctr-hw), min(1., ctr+hw))

def tinh_cap(k):
    """Trần đa dạng hoá. K nhỏ thì nới ra, tránh triệt tiêu tín hiệu tập trung."""
    return int(min(10, max(3, math.ceil(k/10) + 1)))

def fmt_ngay(d, fallback):
    return d.strftime("%d/%m/%Y") if d else str(fallback)

def kiem_tra_toan_ven(ngay_full, lich_phat_hien):
    cb = []; ds = [d for d in ngay_full if d]
    if len(ds) < 2: return cb, None, None
    thu_moi = THU_VN[ds[-1].weekday()]
    if lich_phat_hien not in ("không xác định", "-") and thu_moi != lich_phat_hien:
        cb.append(f"Kỳ mới nhất rơi vào {thu_moi} nhưng trang ghi lịch quay là "
                  f"{lich_phat_hien} → NGHI NGỜ lấy nhầm đài.")
    gaps = [(ds[i+1]-ds[i]).days for i in range(len(ds)-1)]
    chuan = max(set(gaps), key=gaps.count) if gaps else None
    la = sum(1 for g in gaps if g != chuan)
    if la:
        cb.append(f"{la}/{len(gaps)} khoảng cách giữa các kỳ khác {chuan} ngày "
                  f"(thường do nghỉ Tết, hoặc thiếu kỳ).")
    return cb, thu_moi, chuan


# ==============================================================================
#  [5] TẦNG 0 — MỞ RỘNG MẪU
#      chuoi: list theo kỳ, mỗi kỳ là list các SỐ NGUỒN của module.
#         ① [số ĐB]   ② [số G1]   ③ [toàn bộ 18 giải]
#      Cặp cuối = mục tiêu (trọng số 1,0); cặp liền kề = phụ (AUX_WEIGHT).
# ==============================================================================

def muc_tieu(chuoi):
    """Danh sách mục tiêu mỗi kỳ, dạng số nguyên 0-99."""
    return [[int(s[-2:]) for s in ky] for ky in chuoi]

def _mo_rong(chuoi):
    obs = []
    for ky in chuoi:
        for s in ky:
            d = [int(c) for c in s]
            obs.append((d[-2], d[-1], 1.0))
            for k in range(len(d) - 2):
                obs.append((d[k], d[k+1], AUX_WEIGHT))
    return obs

def _phan_phoi(obs):
    c1 = np.full(10, ALPHA); c2 = np.full(10, ALPHA)
    for a, b, w in obs: c1[a] += w; c2[b] += w
    return c1/c1.sum(), c2/c2.sum()


# ==============================================================================
#  [6] CHÍN TÍN HIỆU
#      S1-S6 : tính từ chuỗi nguồn RIÊNG của module   (như v4/v6)
#      S7-S9 : tính từ POOL = 2 số cuối của TOÀN BỘ 18 giải  (mới ở v7)
# ==============================================================================

def S1_TanSuat(chuoi, pool, ctx):
    p1, p2 = _phan_phoi(_mo_rong(chuoi))
    return zscore(np.log(p1[P_D1]) + np.log(p2[P_D2]))

def S2_MarkovKy(chuoi, pool, ctx):
    """Markov liên kỳ. Trạng thái = mục tiêu ĐẦU TIÊN của kỳ trước (neo vào ĐB)."""
    mt = muc_tieu(chuoi)
    T1 = np.full((10,10), ALPHA); T2 = np.full((10,10), ALPHA)
    for t in range(len(mt)-1):
        a0, b0 = mt[t][0]//10, mt[t][0]%10
        for v in mt[t+1]:
            T1[a0, v//10] += 1.; T2[b0, v%10] += 1.
    T1 /= T1.sum(1, keepdims=True); T2 /= T2.sum(1, keepdims=True)
    la, lb = mt[-1][0]//10, mt[-1][0]%10
    return zscore(np.log(T1[la][P_D1]) + np.log(T2[lb][P_D2]))

def S3_Gan(chuoi, pool, ctx):
    """Gan mức chữ số. Hazard phẳng ⇒ giá trị thông tin lý thuyết = 0;
       vẫn cài để backtest tự phán xử bằng trọng số."""
    mt = muc_tieu(chuoi); n = len(mt)
    g1 = np.full(10, float(n)); g2 = np.full(10, float(n))
    for t in range(n-1, -1, -1):
        for v in mt[t]:
            g1[v//10] = min(g1[v//10], n-1-t); g2[v%10] = min(g2[v%10], n-1-t)
    return zscore(zscore(g1)[P_D1] + zscore(g2)[P_D2])

def S4_ChamTong(chuoi, pool, ctx):
    cs = np.full(10, ALPHA)
    for a, b, w in _mo_rong(chuoi): cs[(a+b)%10] += w
    ps = cs/cs.sum()
    return zscore(np.log(ps[(P_D1+P_D2)%10]))

def S5_BongSo(chuoi, pool, ctx):
    p1, p2 = _phan_phoi(_mo_rong(chuoi))
    m1 = np.array([p1[MIRROR[d]] for d in range(10)])
    m2 = np.array([p2[MIRROR[d]] for d in range(10)])
    return zscore(np.log(m1[P_D1]) + np.log(m2[P_D2]))

def S6_CauTruc(chuoi, pool, ctx):
    mt = np.array([v for ky in muc_tieu(chuoi) for v in ky])
    a, b = mt//10, mt%10
    obs = {"chan": (b%2==0).mean(), "tai": (mt>=50).mean(), "kep": (a==b).mean(),
           "satkep": ((np.abs(a-b)==1)|(np.abs(a-b)==9)).mean()}
    exp = {"chan": .50, "tai": .50, "kep": .10, "satkep": .20}
    ind = {"chan": (P_D2%2==0).astype(float), "tai": ((P_D1*10+P_D2)>=50).astype(float),
           "kep": (P_D1==P_D2).astype(float),
           "satkep": ((np.abs(P_D1-P_D2)==1)|(np.abs(P_D1-P_D2)==9)).astype(float)}
    sc = np.zeros(100)
    for k in exp: sc += (exp[k]-obs[k]) * (2*ind[k]-1)
    return zscore(sc)

# ---------- Ba tín hiệu mới, dùng POOL 1.800 quan sát ----------

def S7_ToanGiai(chuoi, pool, ctx):
    """Tần suất chữ số ước lượng từ 2 số cuối của TOÀN BỘ 18 giải.
       Gấp ~3,6 lần dữ liệu so với chỉ dùng số nguồn của module."""
    return zscore(np.log(ctx["pu1"][P_D1]) + np.log(ctx["pu2"][P_D2]))

def S8_BigramJM(chuoi, pool, ctx):
    """P(ab) = P1(a) · [λ·P(b|a) + (1-λ)·P2(b)].
       v4/v6 GIẢ ĐỊNH hai chữ số độc lập — đây là lỗ hổng được vá ở v7.
       18 mẫu/ô bigram (nhờ pool 1.800) mới đủ để ước lượng."""
    Q = ctx["lam"]*ctx["Pba"] + (1-ctx["lam"])*ctx["pu2"][None, :]
    return zscore(np.log(ctx["pu1"][P_D1]) + np.log(Q[P_D1, P_D2]))

def S9_JamesStein(chuoi, pool, ctx):
    """Đếm trực tiếp 100 ô + co ngót Bayes thực nghiệm.
       Hệ số B TỰ NÓ là chẩn đoán: B≈0 nghĩa là mọi chênh lệch chỉ là nhiễu đếm."""
    return zscore(np.log(ctx["ps"]))



def S10_GanDay(chuoi, pool, ctx):
    """S10 — Tần suất CÓ TRỌNG SỐ GIẢM DẦN THEO THỜI GIAN.

    Cơ chế thật: mọi tín hiệu hiện có coi kỳ cách đây 150 kỳ ngang kỳ hôm qua.
    Nhưng máy quay là thiết bị VẬT LÝ — bi mòn, trục lệch, bảo trì, thay bi.
    Nếu có thiên lệch cơ học, nó THAY ĐỔI theo thời gian. Tín hiệu trọng số đều
    sẽ pha loãng thiên lệch hiện tại bằng dữ liệu của một cỗ máy đã khác.

    Trọng số giảm theo hàm mũ: kỳ cách t bước có trọng số HALF_LIFE^(t/nửa đời).
    Đây là bổ sung có cơ sở duy nhất mà bộ hiện tại thiếu.
    """
    mt = muc_tieu(chuoi)
    n = len(mt)
    nua_doi = max(10, n // 4)          # nửa đời = 1/4 lịch sử
    c1 = np.full(10, ALPHA); c2 = np.full(10, ALPHA)
    for t, ky in enumerate(mt):
        w = 0.5 ** ((n - 1 - t) / nua_doi)
        for v in ky:
            c1[v // 10] += w; c2[v % 10] += w
    p1, p2 = c1 / c1.sum(), c2 / c2.sum()
    return zscore(np.log(p1[P_D1]) + np.log(p2[P_D2]))


def S11_PhanTang(chuoi, pool, ctx):
    """S11 — Phân tầng theo NHÓM GIẢI.

    Cơ chế thật: các giải khác nhau có thể quay bằng lồng cầu khác nhau, hoặc
    quay ở thời điểm khác nhau trong buổi. Gộp cả 18/27 giải vào một rổ giả định
    chúng đồng nhất — giả định đó chưa bao giờ được kiểm chứng.

    Tín hiệu này ước lượng riêng cho NỬA ĐẦU bảng giải (giải lớn) và NỬA SAU
    (giải nhỏ), rồi lấy chênh lệch. Nếu hai nhóm đồng nhất, chênh lệch ~ 0 và
    tín hiệu tự triệt tiêu. Nếu khác nhau, nó bắt được.
    """
    n_g = len(chuoi[0]) if chuoi and chuoi[0] else 1
    if n_g < 4:                        # module 1 mục tiêu/kỳ -> không phân tầng được
        return np.zeros(100)
    nua = n_g // 2
    ca = np.full(10, ALPHA); cb = np.full(10, ALPHA)
    da = np.full(10, ALPHA); db = np.full(10, ALPHA)
    for ky in chuoi:
        for i, sN in enumerate(ky):
            a, b = int(sN[-2]), int(sN[-1])
            if i < nua: ca[a] += 1; cb[b] += 1
            else:       da[a] += 1; db[b] += 1
    pa, pb = ca / ca.sum(), cb / cb.sum()
    qa, qb = da / da.sum(), db / db.sum()
    return zscore((np.log(pa[P_D1]) + np.log(pb[P_D2]))
                  - (np.log(qa[P_D1]) + np.log(qb[P_D2])))



# ==============================================================================
#  [6B] TÍN HIỆU CẦU VỊ TRÍ
#
#  Cầu = ghép 2 (hoặc 3) vị trí chữ số CỐ ĐỊNH của kỳ trước thành một số, rồi
#  đánh số đó ở kỳ sau.
#
#  CHỐNG OVERFIT — điểm sống còn:
#    Tại mỗi bước, cầu được chấm điểm CHỈ TRÊN DỮ LIỆU TRƯỚC ĐÓ, rồi lấy top-N
#    cầu tốt nhất để dự báo. Không một mẩu thông tin nào của kỳ đang dự báo
#    tham gia vào việc chọn cầu.
#
#    Dù vậy vẫn phải nhớ: dò 11.342 cầu trên ~80 kỳ thì tìm được cầu "chạy 7 kỳ"
#    là điều CHẮC CHẮN xảy ra kể cả với dữ liệu ngẫu nhiên. Null control sẽ nói
#    tín hiệu này có giá trị thật hay không — đừng tin nó chỉ vì nghe hợp lý.
# ==============================================================================

TOP_CAU      = 60     # số cầu tốt nhất được dùng để bỏ phiếu
N_CAU_3SO    = 30000  # số bộ ba vị trí lấy mẫu (toàn bộ 1,2 triệu là quá chậm)
_CAU_CACHE   = {}


def _bang_chu_so(toan_giai):
    """Ma trận (n_ky, n_vi_tri) mọi chữ số. Vector hoá để quét cầu cực nhanh."""
    vt = [(i, j) for i, x in enumerate(toan_giai[0]) for j in range(len(x))]
    M = np.zeros((len(toan_giai), len(vt)), dtype=np.int16)
    for t, ky in enumerate(toan_giai):
        for k, (i, j) in enumerate(vt):
            M[t, k] = int(ky[i][j])
    return M, vt


def _diem_cau(toan_giai, muc_tieu_moi_ky, kg, min_train=15, top=None):
    """Điểm bỏ phiếu của TOP cầu, cho không gian kg (100 hoặc 1000).

    muc_tieu_moi_ky: list theo kỳ, mỗi phần tử là set các số đã ra (0..kg-1).
    Trả vector kg phần tử: số phiếu mà các cầu tốt nhất dồn cho từng số.
    """
    top = top or TOP_CAU
    n = len(toan_giai)
    if n < min_train + 3:
        return np.zeros(kg)
    M, vt = _bang_chu_so(toan_giai)
    V = M.shape[1]
    rng = np.random.default_rng(SEED)

    if kg == 100:
        A = np.repeat(np.arange(V), V); B = np.tile(np.arange(V), V)
        giu = A != B
        A, B = A[giu], B[giu]
        so_cua = lambda t: M[t, A] * 10 + M[t, B]
    else:
        A = rng.integers(0, V, N_CAU_3SO)
        B = rng.integers(0, V, N_CAU_3SO)
        C = rng.integers(0, V, N_CAU_3SO)
        so_cua = lambda t: M[t, A] * 100 + M[t, B] * 10 + M[t, C]

    # --- Chấm điểm mọi cầu, CHỈ dùng các kỳ đã biết ---
    diem = np.zeros(len(A), dtype=np.int32)
    for t in range(min_train, n - 1):
        thuoc = np.zeros(kg, dtype=bool)
        for v in muc_tieu_moi_ky[t + 1]:
            thuoc[v] = True
        diem += thuoc[so_cua(t)]

    # --- Top cầu bỏ phiếu cho kỳ TIẾP THEO, dựa trên kỳ cuối cùng đã biết ---
    tot = np.argpartition(-diem, min(top, len(diem) - 1))[:top]
    du_bao = so_cua(n - 1)[tot]
    trong_so = diem[tot].astype(float)
    phieu = np.zeros(kg)
    np.add.at(phieu, du_bao, trong_so)
    return phieu


def S12_Cau(chuoi, pool, ctx):
    """S12 — Cầu vị trí 2 số. Top cầu (chọn từ quá khứ) bỏ phiếu cho 00-99."""
    p = ctx.get("cau2")
    if p is None or p.sum() == 0:
        return np.zeros(100)
    return zscore(np.log(p + 1.0))


SIGNALS = [("S1_TanSuat", S1_TanSuat), ("S2_MarkovKy", S2_MarkovKy), ("S3_Gan", S3_Gan),
           ("S4_ChamTong", S4_ChamTong), ("S5_BongSo", S5_BongSo), ("S6_CauTruc", S6_CauTruc),
           ("S7_ToanGiai", S7_ToanGiai), ("S8_BigramJM", S8_BigramJM),
           ("S9_JamesStein", S9_JamesStein),
           ("S10_GanDay", S10_GanDay), ("S11_PhanTang", S11_PhanTang),
           ("S12_Cau", S12_Cau)]
SIG_NAMES = [n for n, _ in SIGNALS]


def _uoc_luong_lambda(pool):
    """λ ước lượng từ held-out 80/20. Không cấu trúc bigram ⇒ λ tự rơi về mức thấp."""
    if len(pool) < 10: return LAMBDA_GRID[0]
    cut = int(len(pool)*0.8); fit, held = pool[:cut], pool[cut:]
    u1 = np.full(10, ALPHA); u2 = np.full(10, ALPHA); B = np.full((10,10), ALPHA)
    for ky in fit:
        for v in ky:
            a, b = v//10, v%10
            u1[a] += 1; u2[b] += 1; B[a, b] += 1
    p1 = u1/u1.sum(); p2 = u2/u2.sum(); Pba = B/B.sum(1, keepdims=True)
    best, best_ll = LAMBDA_GRID[0], -np.inf
    for lam in LAMBDA_GRID:
        Q = lam*Pba + (1-lam)*p2[None, :]
        ll = sum(np.log(p1[v//10]) + np.log(Q[v//10, v%10]) for ky in held for v in ky)
        if ll > best_ll: best, best_ll = lam, ll
    return best


def dung_ctx(pool):
    """Tính sẵn mọi đại lượng dùng chung cho S7-S9. pool: list kỳ -> list mục tiêu 0-99."""
    u1 = np.full(10, ALPHA); u2 = np.full(10, ALPHA)
    Bg = np.full((10,10), ALPHA); c = np.zeros(100); N = 0
    for ky in pool:
        for v in ky:
            a, b = v//10, v%10
            u1[a] += 1; u2[b] += 1; Bg[a, b] += 1; c[v] += 1; N += 1
    pu1, pu2 = u1/u1.sum(), u2/u2.sum()
    Pba = Bg/Bg.sum(1, keepdims=True)
    # --- James-Stein ---
    ph = c/max(N, 1); u = 0.01
    S = float(((ph-u)**2).sum())
    var_nhieu = float((ph*(1-ph)).sum())/max(N, 1)**1 if N else 0.0
    Bjs = 0.0 if S <= 1e-12 else float(np.clip(1 - (100-3)*var_nhieu/max(S*100, 1e-12), 0.0, 1.0))
    ps = np.clip(u + Bjs*(ph-u), 1e-6, None)
    return {"pu1": pu1, "pu2": pu2, "Pba": Pba, "lam": _uoc_luong_lambda(pool),
            "B_js": Bjs, "ps": ps, "N": N}


def ma_tran_tin_hieu(chuoi, pool, toan_giai=None, muc_tieu_cau=None):
    """toan_giai: bảng giải đầy đủ để tính tín hiệu cầu. None -> S12 tự triệt tiêu."""
    ctx = dung_ctx(pool)
    ctx["cau2"] = None
    if toan_giai is not None and len(toan_giai) > 20:
        mt = muc_tieu_cau or [set(ky) for ky in pool]
        ctx["cau2"] = _diem_cau(toan_giai, mt, 100)
    return np.vstack([f(chuoi, pool, ctx) for _, f in SIGNALS]), ctx


# ==============================================================================
#  [7] CHỌN TOP-K CÓ RÀNG BUỘC ĐA DẠNG HOÁ
# ==============================================================================

def select_top(scores, k, cap):
    order = np.argsort(-scores, kind="stable")
    for c in range(cap, 11):
        chosen, u1, u2 = [], [0]*10, [0]*10
        for idx in order:
            d1, d2 = P_D1[idx], P_D2[idx]
            if u1[d1] < c and u2[d2] < c:
                chosen.append(idx); u1[d1] += 1; u2[d2] += 1
                if len(chosen) == k: return np.array(chosen), c
    return order[:k], 10


# ==============================================================================
#  [8] BACKTEST WALK-FORWARD
# ==============================================================================

def cua_so(chuoi, pool, min_train, max_pts, toan_giai=None, mt_cau=None):
    """toan_giai: bảng giải đầy đủ (cho tín hiệu cầu). Luôn cắt tới t — không rò rỉ."""
    starts = list(range(min_train, len(chuoi)))
    if max_pts and len(starts) > max_pts: starts = starts[-max_pts:]
    out = []
    for t in starts:
        tg_t = toan_giai[:t] if toan_giai is not None else None
        mt_t = mt_cau[:t] if mt_cau is not None else None
        Z, _ = ma_tran_tin_hieu(chuoi[:t], pool[:t], tg_t, mt_t)
        out.append((Z, muc_tieu(chuoi)[t]))
    return out

def cham_diem(wins, w, k, cap):
    hits = 0; tong = 0; per = []
    for Z, mt in wins:
        sel = set(select_top(w @ Z, k, cap)[0].tolist())
        h = sum(1 for v in mt if v in sel)
        hits += h; tong += len(mt); per.append(h)
    return hits/tong, hits, tong, per

def _chi_so_tin_hieu():
    """Vị trí các tín hiệu đang bật. Bộ GON dò 31 tổ hợp thay vì 511."""
    if BO_TIN_HIEU.upper() == "GON":
        return [i for i, n in enumerate(SIG_NAMES) if n in TIN_HIEU_GON]
    return list(range(len(SIGNALS)))


def toi_uu(wins, k, cap):
    """Quét tổ hợp trọng số. CHỈ dò trên các tín hiệu đang bật — mỗi tín hiệu
       thêm vào làm gấp đôi diện tích dò tìm quy luật giả."""
    idx = _chi_so_tin_hieu()
    best = (None, -1., 0, None)
    for combo in product(WEIGHT_GRID, repeat=len(idx)):
        if sum(combo) == 0: continue
        w = np.zeros(len(SIGNALS))
        for j, c in zip(idx, combo): w[j] = c
        hr, h, tot, per = cham_diem(wins, w, k, cap)
        if hr > best[1]: best = (w, hr, h, per)
    return best


# ==============================================================================
#  [9] NULL CONTROL THEO HỌ — PHƯƠNG PHÁP MIN-P
#      v6 dùng max(hit rate): SAI khi các module có mốc ngẫu nhiên khác nhau
#      (module ①② mốc 50%, module ③ mốc 10% — max luôn chọn ①②).
#      v7 quy mỗi module về p-value theo phân phối null CỦA CHÍNH NÓ, rồi lấy
#      p nhỏ nhất mỗi lượt. Đây là phép hiệu chỉnh đúng.
# ==============================================================================

def sinh_ky_gia(co_cau, rng):
    """Sinh 1 kỳ giả đúng cơ cấu giải (18 số MN/MT, 27 số MB), ĐB đứng đầu."""
    ra = []
    for g in THU_TU_GIAI:
        if g not in co_cau: continue
        sl, nd = co_cau[g]
        ra += ["".join(map(str, rng.integers(0, 10, nd))) for _ in range(sl)]
    return ra

def null_theo_ho(n_runs, n_ky, co_cau, cau_hinh, min_train, max_pts, rng=None):
    """cau_hinh: list (khoa, vi_tri, K, cap). vi_tri=None nghĩa là lấy toàn bộ giải."""
    rng = rng or rng_global
    out = []
    for i in range(n_runs):
        gia = [sinh_ky_gia(co_cau, rng) for _ in range(n_ky)]
        pool = [[int(s[-2:]) for s in ky] for ky in gia]
        hang = []
        for _, vt, K, cap in cau_hinh:
            ch = gia if vt is None else [[ky[vt]] for ky in gia]
            hang.append(toi_uu(cua_so(ch, pool, min_train, max_pts), K, cap)[1])
        out.append(hang)
        if (i+1) % max(1, n_runs//10) == 0: print(f"     ... {i+1}/{n_runs}", end="\r")
    print(" "*44, end="\r")
    return np.array(out)

N_NULL_TOI_THIEU = 20   # dưới mức này, p-value không đáng tin


def p_min_ho(null, hr_that):
    """Trả (p_thô[], p_bonferroni[], p_theo_ho[]) theo thủ tục min-p."""
    n, m = null.shape
    if n < N_NULL_TOI_THIEU:
        print(f"  ⚠ CHỈ CÓ {n} LƯỢT NULL — p-value KHÔNG ĐÁNG TIN.")
        print(f"    Độ phân giải chỉ {1/n:.3f}; giá trị 0.000 là giả tạo, không phải phát hiện.")
        print(f"    Cần ít nhất {N_NULL_TOI_THIEU} lượt. Xoá null_cache.json rồi chạy lại.")
    p_tho = np.array([(null[:, j] >= hr_that[j]).mean() for j in range(m)])
    # p-value của từng lượt nhiễu, so với phân phối null của chính module đó
    P = np.empty((n, m))
    for j in range(m):
        col = null[:, j]
        P[:, j] = [(col >= v).mean() for v in col]
    p_min = P.min(axis=1)
    p_ho = np.array([(p_min <= p_tho[j]).mean() for j in range(m)])
    return p_tho, np.minimum(1.0, p_tho*m), p_ho


#  [14] ENGINE 3 SỐ  —  không gian 1.000, mục tiêu = 3 chữ số cuối của MỌI giải
#       Mỗi kỳ MN/MT cho 17 lô, MB cho 23 lô. 100 kỳ -> 1.700 quan sát THẬT
#       (không phải dữ liệu phụ trợ hạ trọng số như module 2 số).
# ==============================================================================

def _dem_3so(train3):
    """Đếm unigram theo vị trí + bigram (d1->d2, d2->d3) từ danh sách lô 3 số."""
    uni = np.full((3, 10), ALPHA)
    b12 = np.full((10, 10), ALPHA)
    b23 = np.full((10, 10), ALPHA)
    for ky in train3:
        for t in ky:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            uni[0][a] += 1; uni[1][b] += 1; uni[2][c] += 1
            b12[a][b] += 1; b23[b][c] += 1
    return uni, b12, b23


def _lambda_3so(train3):
    """Ước lượng λ (Jelinek-Mercer) BẰNG DỮ LIỆU: fit trên 80% đầu, chọn λ tối đa
       log-likelihood trên 20% cuối. Không áp đặt hằng số."""
    if len(train3) < 10:
        return 0.5
    cut = int(len(train3) * 0.8)
    fit, held = train3[:cut], train3[cut:]
    uni, b12, b23 = _dem_3so(fit)
    pu = uni / uni.sum(1, keepdims=True)
    P12 = b12 / b12.sum(1, keepdims=True)
    P23 = b23 / b23.sum(1, keepdims=True)
    best, best_ll = LAMBDA_GRID[0], -np.inf
    for lam in LAMBDA_GRID:
        Q12 = lam * P12 + (1 - lam) * pu[1][None, :]
        Q23 = lam * P23 + (1 - lam) * pu[2][None, :]
        ll = 0.0
        for ky in held:
            for t in ky:
                a, b, c = int(t[0]), int(t[1]), int(t[2])
                ll += np.log(pu[0][a]) + np.log(Q12[a, b]) + np.log(Q23[b, c])
        if ll > best_ll:
            best, best_ll = lam, ll
    return best


def T1_unigram(train3, ctx):
    """T1 — Tần suất chữ số theo VỊ TRÍ. 3 x 10 = 30 ô, mật độ mẫu cao nhất."""
    uni = ctx["uni"]
    p = uni / uni.sum(1, keepdims=True)
    return zscore(np.log(p[0][T_D1]) + np.log(p[1][T_D2]) + np.log(p[2][T_D3]))


def T2_ngram(train3, ctx):
    """T2 — N-gram nội suy Jelinek-Mercer. ĐÂY LÀ NÂNG CẤP CỐT LÕI:
       P(abc) ≈ P1(a) · [λP(b|a)+(1-λ)P(b)] · [λP(c|b)+(1-λ)P(c)]
       Ước lượng 10 + 100 + 100 ô thay vì 1.000 ô."""
    uni, b12, b23, lam = ctx["uni"], ctx["b12"], ctx["b23"], ctx["lam"]
    pu = uni / uni.sum(1, keepdims=True)
    P12 = b12 / b12.sum(1, keepdims=True)
    P23 = b23 / b23.sum(1, keepdims=True)
    Q12 = lam * P12 + (1 - lam) * pu[1][None, :]
    Q23 = lam * P23 + (1 - lam) * pu[2][None, :]
    return zscore(np.log(pu[0][T_D1]) + np.log(Q12[T_D1, T_D2]) + np.log(Q23[T_D2, T_D3]))


def T3_markov_lienky(train3, ctx):
    """T3 — Markov LIÊN KỲ: trạng thái = lô 3 số của ĐB kỳ trước (phần tử [0]),
       chuyển sang toàn bộ lô của kỳ sau. 3 ma trận 10x10."""
    T = np.full((3, 10, 10), ALPHA)
    for t in range(len(train3) - 1):
        prev = train3[t][0]
        for nxt in train3[t + 1]:
            for i in range(3):
                T[i][int(prev[i])][int(nxt[i])] += 1
    T /= T.sum(2, keepdims=True)
    p = train3[-1][0]
    a0, b0, c0 = int(p[0]), int(p[1]), int(p[2])
    return zscore(np.log(T[0][a0][T_D1]) + np.log(T[1][b0][T_D2]) + np.log(T[2][c0][T_D3]))


def T4_gan(train3, ctx):
    """T4 — Gan mức chữ số theo vị trí. Phân phối hình học KHÔNG NHỚ nên giá trị
       thông tin lý thuyết = 0; để backtest tự phán xử bằng trọng số."""
    n = len(train3)
    g = np.full((3, 10), float(n))
    for t in range(n - 1, -1, -1):
        for tt in train3[t]:
            for i in range(3):
                g[i][int(tt[i])] = min(g[i][int(tt[i])], n - 1 - t)
    return zscore(zscore(g[0])[T_D1] + zscore(g[1])[T_D2] + zscore(g[2])[T_D3])


def T5_chamtong(train3, ctx):
    """T5 — Chạm tổng (a+b+c) mod 10. Chỉ 10 ô -> mật độ mẫu rất tốt."""
    cs = np.full(10, ALPHA)
    for ky in train3:
        for t in ky:
            cs[(int(t[0]) + int(t[1]) + int(t[2])) % 10] += 1
    ps = cs / cs.sum()
    return zscore(np.log(ps[(T_D1 + T_D2 + T_D3) % 10]))


def T6_bongso(train3, ctx):
    """T6 — Bóng số 0↔5,1↔6,2↔7,3↔8,4↔9. Mã hoá kiểm định được, không mặc định đúng."""
    uni = ctx["uni"]
    p = uni / uni.sum(1, keepdims=True)
    m = np.array([[p[i][MIRROR[d]] for d in range(10)] for i in range(3)])
    return zscore(np.log(m[0][T_D1]) + np.log(m[1][T_D2]) + np.log(m[2][T_D3]))


def T8_Cau(train3, ctx):
    """T8 — Cầu 3 vị trí. Lấy mẫu 30.000 bộ ba (toàn bộ 1,2 triệu là quá chậm),
       chấm điểm CHỈ trên quá khứ, top cầu bỏ phiếu cho 000-999."""
    p = ctx.get("cau3")
    if p is None or p.sum() == 0:
        return np.zeros(1000)
    return zscore(np.log(p + 1.0))


def T7_cautruc(train3, ctx):
    """T7 — Cân bằng cấu trúc: kép bằng (aaa), kép (đúng 2 giống), tài/xỉu, chẵn hết.
       Theo hướng BÙ TRỪ; nếu sai hướng, backtest gán trọng số 0."""
    tot = 0
    obs = {"kepbang": 0.0, "kep": 0.0, "tai": 0.0, "chanhet": 0.0}
    for ky in train3:
        for t in ky:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            tot += 1
            if a == b == c: obs["kepbang"] += 1
            elif a == b or b == c or a == c: obs["kep"] += 1
            if a * 100 + b * 10 + c >= 500: obs["tai"] += 1
            if a % 2 == 0 and b % 2 == 0 and c % 2 == 0: obs["chanhet"] += 1
    for k in obs: obs[k] /= max(tot, 1)
    exp = {"kepbang": 0.010, "kep": 0.270, "tai": 0.500, "chanhet": 0.125}
    ind = {
        "kepbang": ((T_D1 == T_D2) & (T_D2 == T_D3)).astype(float),
        "kep": (((T_D1 == T_D2) | (T_D2 == T_D3) | (T_D1 == T_D3)).astype(int)
                - ((T_D1 == T_D2) & (T_D2 == T_D3)).astype(int)).astype(float),
        "tai": ((T_D1 * 100 + T_D2 * 10 + T_D3) >= 500).astype(float),
        "chanhet": ((T_D1 % 2 == 0) & (T_D2 % 2 == 0) & (T_D3 % 2 == 0)).astype(float),
    }
    sc = np.zeros(1000)
    for k in exp:
        sc += (exp[k] - obs[k]) * (2 * ind[k] - 1)
    return zscore(sc)


SIGNALS_3 = [("T1_ViTri", T1_unigram), ("T2_Ngram", T2_ngram),
             ("T3_MarkovKy", T3_markov_lienky), ("T4_Gan", T4_gan),
             ("T5_ChamTong", T5_chamtong), ("T6_BongSo", T6_bongso),
             ("T7_CauTruc", T7_cautruc), ("T8_Cau", T8_Cau)]
SIG3_NAMES = [n for n, _ in SIGNALS_3]


def build_signal_matrix_3(train3, toan_giai=None):
    """toan_giai: bảng giải đầy đủ cho tín hiệu cầu T8. None -> T8 tự triệt tiêu."""
    uni, b12, b23 = _dem_3so(train3)
    ctx = {"uni": uni, "b12": b12, "b23": b23, "lam": _lambda_3so(train3), "cau3": None}
    if toan_giai is not None and len(toan_giai) > 20:
        mt = [{int(x) for x in ky} for ky in train3]
        ctx["cau3"] = _diem_cau(toan_giai, mt, 1000)
    return np.vstack([f(train3, ctx) for _, f in SIGNALS_3]), ctx["lam"]


def select_top3(scores, k, cap):
    """Ràng buộc đa dạng hoá trên CẢ 3 vị trí chữ số."""
    order = np.argsort(-scores, kind="stable")
    for c in range(cap, k + 1):
        chosen, u = [], [[0] * 10 for _ in range(3)]
        for idx in order:
            d1, d2, d3 = T_D1[idx], T_D2[idx], T_D3[idx]
            if u[0][d1] < c and u[1][d2] < c and u[2][d3] < c:
                chosen.append(idx); u[0][d1] += 1; u[1][d2] += 1; u[2][d3] += 1
                if len(chosen) == k:
                    return np.array(chosen), c
    return order[:k], k


def tinh_cap3(k):
    return int(min(k, max(3, math.ceil(k / 10) + 2)))


def precompute_windows_3(draws3, min_train=30, max_pts=None, toan_giai=None):
    starts = list(range(min_train, len(draws3)))
    if max_pts and len(starts) > max_pts:
        starts = starts[-max_pts:]
    out = []
    for t in starts:
        Z, lam = build_signal_matrix_3(draws3[:t],
                                       toan_giai[:t] if toan_giai is not None else None)
        muc_tieu = np.array([int(x) for x in draws3[t]])
        out.append((Z, muc_tieu))
    return out


def cham_diem_3(wins3, w, k, cap):
    """Trả về (hit_rate mỗi lô, tổng lượt trúng, tổng số lô, list lượt trúng mỗi kỳ)."""
    hits, tong_lo, per_ky = 0, 0, []
    for Z, mt in wins3:
        sel = set(select_top3(w @ Z, k, cap)[0].tolist())
        h = sum(1 for x in mt if x in sel)
        hits += h; tong_lo += len(mt); per_ky.append(h)
    return hits / tong_lo, hits, tong_lo, per_ky


def toi_uu_trong_so_3(wins3, k, cap, grid=None):
    grid = grid or WEIGHT_GRID_3
    best = (None, -1.0, 0, None)
    for combo in product(grid, repeat=len(SIGNALS_3)):
        w = np.array(combo, float)
        if w.sum() == 0:
            continue
        hr, h, tot, per = cham_diem_3(wins3, w, k, cap)
        if hr > best[1]:
            best = (w, hr, h, per)
    return best


def null_control_3(n_runs, n_ky, n_lo, k, cap, max_pts, min_train, rng=None):
    """Chạy Y HỆT quy trình (kể cả quét trọng số) trên dữ liệu ngẫu nhiên hoàn toàn."""
    rng = rng or rng_global
    out = []
    for i in range(n_runs):
        fake = [["".join(map(str, rng.integers(0, 10, 3))) for _ in range(n_lo)]
                for _ in range(n_ky)]
        wins = precompute_windows_3(fake, min_train=min_train, max_pts=max_pts)
        out.append(toi_uu_trong_so_3(wins, k, cap)[1])
        if (i + 1) % max(1, n_runs // 10) == 0:
            print(f"     ... {i+1}/{n_runs}", end="\r")
    print(" " * 40, end="\r")
    return np.array(out)


def auto_tune_3(n):
    """(số lần null, trần điểm test, train tối thiểu) cho module 3 số."""
    if n <= 60:   return 40, None, 20
    if n <= 120:  return 40, 70, 30
    return 25, 90, 40




# ==============================================================================
#  [21] ENGINE 4 SỐ — không gian 10.000, mục tiêu = 4 chữ số cuối của mọi giải
#       có từ 4 chữ số trở lên.  MN/MT: 16 lô/kỳ   |   MB: 20 lô/kỳ
#
#  ⚠ CẢNH BÁO MẬT ĐỘ: 150 kỳ × 20 lô = 3.000 quan sát trên 10.000 ô = 0,30 mẫu/ô.
#    Đây là mật độ THẤP NHẤT trong mọi module. Khoảng 74% số 4 chữ số chưa từng
#    xuất hiện lần nào. Mô hình n-gram giảm nhẹ vấn đề nhưng KHÔNG giải quyết được.
#  ⚠ EV: 8888/10000 − 1 = −11,12% — tệ nhất trong mọi loại cược của hệ thống.
# ==============================================================================

QUADS = np.array([(i//1000, (i//100) % 10, (i//10) % 10, i % 10) for i in range(10000)])
Q_D1, Q_D2, Q_D3, Q_D4 = QUADS[:, 0], QUADS[:, 1], QUADS[:, 2], QUADS[:, 3]


def lo4_cua_ky(so_list):
    """4 chữ số cuối của mọi giải >= 4 chữ số. ĐB đứng đầu (trạng thái Markov Q3)."""
    return [s[-4:] for s in so_list if len(s) >= 4]


def _dem_4so(train4):
    """Unigram theo vị trí (4x10) + 3 bigram liền kề (10x10 mỗi cái)."""
    uni = np.full((4, 10), ALPHA)
    bg = np.full((3, 10, 10), ALPHA)
    for ky in train4:
        for t in ky:
            d = [int(c) for c in t]
            for i in range(4):
                uni[i][d[i]] += 1
            for i in range(3):
                bg[i][d[i]][d[i+1]] += 1
    return uni, bg


def _lambda_4so(train4):
    """λ Jelinek-Mercer ước lượng từ held-out 80/20 — không áp đặt hằng số.
       Không có cấu trúc bigram thì λ tự rơi xuống mức thấp nhất."""
    if len(train4) < 10:
        return LAMBDA_GRID[0]
    cut = int(len(train4) * 0.8)
    fit, held = train4[:cut], train4[cut:]
    uni, bg = _dem_4so(fit)
    pu = uni / uni.sum(1, keepdims=True)
    P = bg / bg.sum(2, keepdims=True)
    best, best_ll = LAMBDA_GRID[0], -np.inf
    for lam in LAMBDA_GRID:
        Q = [lam * P[i] + (1 - lam) * pu[i+1][None, :] for i in range(3)]
        ll = 0.0
        for ky in held:
            for t in ky:
                d = [int(c) for c in t]
                ll += np.log(pu[0][d[0]])
                for i in range(3):
                    ll += np.log(Q[i][d[i], d[i+1]])
        if ll > best_ll:
            best, best_ll = lam, ll
    return best


def Q1_vitri(train4, ctx):
    """Q1 — Tần suất chữ số theo VỊ TRÍ. 4x10 = 40 ô, mật độ mẫu cao nhất."""
    p = ctx["uni"] / ctx["uni"].sum(1, keepdims=True)
    return zscore(np.log(p[0][Q_D1]) + np.log(p[1][Q_D2])
                  + np.log(p[2][Q_D3]) + np.log(p[3][Q_D4]))


def Q2_ngram(train4, ctx):
    """Q2 — N-gram nội suy Jelinek-Mercer, CỐT LÕI của module này:
       P(abcd) ≈ P1(a)·Q(b|a)·Q(c|b)·Q(d|c)
       Ước lượng 10 + 3×100 = 310 tham số thay vì 10.000."""
    uni, bg, lam = ctx["uni"], ctx["bg"], ctx["lam"]
    pu = uni / uni.sum(1, keepdims=True)
    P = bg / bg.sum(2, keepdims=True)
    Q = [lam * P[i] + (1 - lam) * pu[i+1][None, :] for i in range(3)]
    return zscore(np.log(pu[0][Q_D1]) + np.log(Q[0][Q_D1, Q_D2])
                  + np.log(Q[1][Q_D2, Q_D3]) + np.log(Q[2][Q_D3, Q_D4]))


def Q3_markovky(train4, ctx):
    """Q3 — Markov LIÊN KỲ: trạng thái = lô 4 số của ĐB kỳ trước, chuyển sang
       toàn bộ lô của kỳ sau. 4 ma trận 10x10."""
    T = np.full((4, 10, 10), ALPHA)
    for t in range(len(train4) - 1):
        prev = train4[t][0]
        for nxt in train4[t + 1]:
            for i in range(4):
                T[i][int(prev[i])][int(nxt[i])] += 1
    T /= T.sum(2, keepdims=True)
    p = [int(c) for c in train4[-1][0]]
    return zscore(np.log(T[0][p[0]][Q_D1]) + np.log(T[1][p[1]][Q_D2])
                  + np.log(T[2][p[2]][Q_D3]) + np.log(T[3][p[3]][Q_D4]))


def Q4_gan(train4, ctx):
    """Q4 — Gan mức chữ số theo vị trí. Hazard phẳng ⇒ giá trị thông tin lý
       thuyết = 0; vẫn cài để backtest tự phán xử bằng trọng số."""
    n = len(train4)
    g = np.full((4, 10), float(n))
    for t in range(n - 1, -1, -1):
        for tt in train4[t]:
            for i in range(4):
                g[i][int(tt[i])] = min(g[i][int(tt[i])], n - 1 - t)
    return zscore(zscore(g[0])[Q_D1] + zscore(g[1])[Q_D2]
                  + zscore(g[2])[Q_D3] + zscore(g[3])[Q_D4])


def Q5_chamtong(train4, ctx):
    """Q5 — Chạm tổng (a+b+c+d) mod 10. Chỉ 10 ô -> mật độ mẫu rất tốt."""
    cs = np.full(10, ALPHA)
    for ky in train4:
        for t in ky:
            cs[sum(int(c) for c in t) % 10] += 1
    ps = cs / cs.sum()
    return zscore(np.log(ps[(Q_D1 + Q_D2 + Q_D3 + Q_D4) % 10]))


def Q6_bongso(train4, ctx):
    """Q6 — Bóng số 0↔5,1↔6,2↔7,3↔8,4↔9. Mã hoá kiểm định được."""
    p = ctx["uni"] / ctx["uni"].sum(1, keepdims=True)
    m = np.array([[p[i][MIRROR[d]] for d in range(10)] for i in range(4)])
    return zscore(np.log(m[0][Q_D1]) + np.log(m[1][Q_D2])
                  + np.log(m[2][Q_D3]) + np.log(m[3][Q_D4]))


def Q7_cautruc(train4, ctx):
    """Q7 — Cân bằng cấu trúc: có chữ số lặp, tài/xỉu, chẵn hết, tổng chẵn.
       Theo hướng BÙ TRỪ; sai hướng thì backtest gán trọng số 0."""
    tot = 0
    obs = {"lap": 0.0, "tai": 0.0, "chanhet": 0.0, "tongchan": 0.0}
    for ky in train4:
        for t in ky:
            d = [int(c) for c in t]
            tot += 1
            if len(set(d)) < 4: obs["lap"] += 1
            if int(t) >= 5000: obs["tai"] += 1
            if all(x % 2 == 0 for x in d): obs["chanhet"] += 1
            if sum(d) % 2 == 0: obs["tongchan"] += 1
    for k in obs:
        obs[k] /= max(tot, 1)
    exp = {"lap": 0.4960, "tai": 0.5000, "chanhet": 0.0625, "tongchan": 0.5000}
    val = Q_D1 * 1000 + Q_D2 * 100 + Q_D3 * 10 + Q_D4
    khac = ((Q_D1 != Q_D2) & (Q_D1 != Q_D3) & (Q_D1 != Q_D4)
            & (Q_D2 != Q_D3) & (Q_D2 != Q_D4) & (Q_D3 != Q_D4))
    ind = {"lap": (~khac).astype(float),
           "tai": (val >= 5000).astype(float),
           "chanhet": ((Q_D1 % 2 == 0) & (Q_D2 % 2 == 0)
                       & (Q_D3 % 2 == 0) & (Q_D4 % 2 == 0)).astype(float),
           "tongchan": (((Q_D1 + Q_D2 + Q_D3 + Q_D4) % 2) == 0).astype(float)}
    sc = np.zeros(10000)
    for k in exp:
        sc += (exp[k] - obs[k]) * (2 * ind[k] - 1)
    return zscore(sc)


SIGNALS_4 = [("Q1_ViTri", Q1_vitri), ("Q2_Ngram", Q2_ngram),
             ("Q3_MarkovKy", Q3_markovky), ("Q4_Gan", Q4_gan),
             ("Q5_ChamTong", Q5_chamtong), ("Q6_BongSo", Q6_bongso),
             ("Q7_CauTruc", Q7_cautruc)]
SIG4_NAMES = [n for n, _ in SIGNALS_4]


def build_signal_matrix_4(train4):
    uni, bg = _dem_4so(train4)
    ctx = {"uni": uni, "bg": bg, "lam": _lambda_4so(train4)}
    return np.vstack([f(train4, ctx) for _, f in SIGNALS_4]), ctx["lam"]


def tinh_cap4(k):
    """Trần đa dạng hoá cho 4 vị trí chữ số."""
    return int(min(k, max(3, math.ceil(k / 10) + 2)))


def select_top4(scores, k, cap):
    """Ràng buộc đa dạng hoá trên CẢ 4 vị trí chữ số."""
    order = np.argsort(-scores, kind="stable")
    for c in range(cap, k + 1):
        chosen, u = [], [[0] * 10 for _ in range(4)]
        for idx in order:
            d = (Q_D1[idx], Q_D2[idx], Q_D3[idx], Q_D4[idx])
            if all(u[i][d[i]] < c for i in range(4)):
                chosen.append(idx)
                for i in range(4):
                    u[i][d[i]] += 1
                if len(chosen) == k:
                    return np.array(chosen), c
    return order[:k], k


def precompute_windows_4(draws4, min_train=30, max_pts=None):
    starts = list(range(min_train, len(draws4)))
    if max_pts and len(starts) > max_pts:
        starts = starts[-max_pts:]
    out = []
    for t in starts:
        Z, _ = build_signal_matrix_4(draws4[:t])
        out.append((Z, np.array([int(x) for x in draws4[t]])))
    return out


def cham_diem_4(wins4, w, k, cap):
    hits, tong, per = 0, 0, []
    for Z, mt in wins4:
        sel = set(select_top4(w @ Z, k, cap)[0].tolist())
        h = sum(1 for x in mt if x in sel)
        hits += h; tong += len(mt); per.append(h)
    return hits / tong, hits, tong, per


def toi_uu_trong_so_4(wins4, k, cap, grid=None):
    grid = grid or WEIGHT_GRID_4
    best = (None, -1.0, 0, None)
    for combo in product(grid, repeat=len(SIGNALS_4)):
        w = np.array(combo, float)
        if w.sum() == 0:
            continue
        hr, h, tot, per = cham_diem_4(wins4, w, k, cap)
        if hr > best[1]:
            best = (w, hr, h, per)
    return best




# ==============================================================================
#  [22] CHẤM ĐIỂM TỪNG TÍN HIỆU
#
#  Vì sao cần: khi quét tổ hợp trọng số, một tín hiệu VÔ DỤNG vẫn có thể lọt vào
#  bộ thắng nhờ ăn may. Hàm này chạy MỖI TÍN HIỆU MỘT MÌNH — không quét trọng số,
#  nên KHÔNG CÓ overfit từ dò tìm. Con số thu được là giá trị thật của tín hiệu.
#
#  So với phân phối null của chính nó (RNG công bằng mô phỏng), có hiệu chỉnh
#  Bonferroni cho số tín hiệu được thử.
# ==============================================================================

def danh_gia_tin_hieu(toan_giai, key="LO2", K=20, min_train=None, n_null=25):
    """Chấm điểm từng tín hiệu 2 số riêng lẻ. Trả DataFrame đã xếp hạng."""
    n = len(toan_giai)
    khu = "MB" if len(toan_giai[0]) == 27 else "MN"
    pool = [[int(x[-2:]) for x in ky] for ky in toan_giai]
    chuoi = toan_giai if key == "LO2" else [[ky[0]] for ky in toan_giai]
    min_train = min_train or max(15, n // 3)
    cap = tinh_cap(K)
    wins = cua_so(chuoi, pool, min_train, None)
    m = len(SIGNALS)

    # --- hit rate của từng tín hiệu trên DỮ LIỆU THẬT ---
    that = []
    for j in range(m):
        w = np.zeros(m); w[j] = 1.0
        hr, h, tot, _ = cham_diem(wins, w, K, cap)
        that.append((hr, h, tot))

    # --- phân phối null: cùng phép đo trên RNG công bằng ---
    print(f"  Đang dựng phân phối null ({n_null} lượt × {m} tín hiệu)...")
    rng = np.random.default_rng(SEED)
    null = np.zeros((n_null, m))
    for i in range(n_null):
        gia = [sinh_ky_gia(CO_CAU_GIAI[khu], rng) for _ in range(n)]
        pg = [[int(x[-2:]) for x in ky] for ky in gia]
        cg = gia if key == "LO2" else [[ky[0]] for ky in gia]
        wg = cua_so(cg, pg, min_train, None)
        for j in range(m):
            w = np.zeros(m); w[j] = 1.0
            null[i, j] = cham_diem(wg, w, K, cap)[0]
        if (i + 1) % max(1, n_null // 5) == 0:
            print(f"     ... {i+1}/{n_null}", end="\r")
    print(" " * 40, end="\r")

    p0 = K / 100.0
    hang = []
    for j in range(m):
        hr = that[j][0]
        p_tho = (null[:, j] >= hr).mean()
        hang.append({"tin_hieu": SIG_NAMES[j], "hit_rate": hr,
                     "nhieu_TB": null[:, j].mean(),
                     "vuot_nhieu": hr - null[:, j].mean(),
                     "p_tho": p_tho, "p_Bonf": min(1.0, p_tho * m),
                     "dang_bat": SIG_NAMES[j] in TIN_HIEU_GON})
    df = pd.DataFrame(hang).sort_values("vuot_nhieu", ascending=False)

    print(f"\n{'='*78}")
    print(f"  CHẤM ĐIỂM TỪNG TÍN HIỆU — module {MO_TA[key][0]}, K={K}")
    print(f"  {len(wins)} kỳ test | mốc ngẫu nhiên {p0:.2%} | KHÔNG quét trọng số")
    print("="*78)
    print(f"  {'Tín hiệu':<16}{'Hit rate':>10}{'Nhiễu TB':>11}{'Vượt nhiễu':>13}"
          f"{'p Bonf':>9}{'Bật?':>7}")
    print("  " + "-"*74)
    for _, r in df.iterrows():
        dau = "✓" if r["dang_bat"] else "·"
        sao = " ***" if r["p_Bonf"] < 0.05 else ""
        print(f"  {r['tin_hieu']:<16}{r['hit_rate']:>9.2%}{r['nhieu_TB']:>11.2%}"
              f"{r['vuot_nhieu']:>+13.2%}{r['p_Bonf']:>9.3f}{dau:>7}{sao}")
    print("  " + "-"*74)
    tot = df[df["p_Bonf"] < 0.05]["tin_hieu"].tolist()
    if tot:
        print(f"  Tín hiệu VƯỢT ngưỡng nhiễu: {', '.join(tot)}")
        print(f"  → Đáng cân nhắc đưa vào TIN_HIEU_GON và chạy lại backtest.")
    else:
        print(f"  KHÔNG tín hiệu nào vượt ngưỡng nhiễu sau hiệu chỉnh Bonferroni.")
        print(f"  → Không có cơ sở để thêm hay bớt tín hiệu dựa trên dữ liệu này.")
    print("="*78)
    return df


# ==============================================================================
#  [12] CHỌN NGÀY  —  tự tìm đài quay ngày đó, chống rò rỉ dữ liệu tương lai
# ==============================================================================

from datetime import date as _date, datetime as _dt, datetime, timedelta, timezone

_LICH_FILE = os.environ.get("LICH_PATH", "data/lich_quay.json")
_NULL_FILE = os.environ.get("NULL_PATH", "data/null_cache.json")
_LICH_MEM, _NULL_MEM = None, None


def doc_ngay(s):
    """Nhận dd.mm.yyyy (cũng chấp nhận dd/mm/yyyy, dd-mm-yyyy)."""
    s = str(s).strip().replace("/", ".").replace("-", ".")
    try:
        return _dt.strptime(s, "%d.%m.%Y").date()
    except ValueError:
        raise ValueError(f"CHON_NGAY sai định dạng: {s!r}. Cần dd.mm.yyyy, ví dụ 23.08.2026")


def _thu_cua_dai(stt, n_mau=30):
    """Suy TẬP các thứ mở thưởng của 1 đài TỪ DỮ LIỆU THẬT.

    QUAN TRỌNG: nhiều đài quay HAI NGÀY MỘT TUẦN —
        TP.HCM   : Thứ 2 + Thứ 7
        Huế      : Chủ nhật + Thứ 2
        Đà Nẵng  : Thứ 4 + Thứ 7
        Khánh Hoà: Thứ 4 + Chủ nhật
    Bản cũ dùng max() nên chỉ giữ được MỘT thứ, làm các đài này biến mất
    khỏi lịch của ngày còn lại. Đây chính là lý do email chỉ có 4 đài.

    Một thứ được tính là ngày quay nếu xuất hiện >= 15% số kỳ mẫu (tối thiểu 2 lần).
    """
    _, ma, _, _ = lay_dai(stt)
    html, _ = _tai_1_trang(ma, max(30, n_mau))
    ds = [d for d in _gan_nam(_boc_ngay(html, n_mau)) if d]
    if not ds:
        raise RuntimeError("không đọc được ngày mở thưởng")
    dem = {}
    for d in ds:
        dem[d.weekday()] = dem.get(d.weekday(), 0) + 1
    nguong = max(2, int(0.15 * len(ds)))
    cac_thu = sorted(w for w, c in dem.items() if c >= nguong)
    if not cac_thu:                       # dữ liệu quá thưa -> lấy thứ hay gặp nhất
        cac_thu = [max(dem, key=dem.get)]
    return cac_thu, len(ds)


def xay_lich(bat_buoc_lam_moi=False, im_lang=False):
    """Lịch quay 36 đài. Đọc từ file/bộ nhớ nếu đã có, nếu không thì quét 1 lần."""
    global _LICH_MEM
    if _LICH_MEM and not bat_buoc_lam_moi:
        return _LICH_MEM
    if os.path.exists(_LICH_FILE) and not bat_buoc_lam_moi:
        cu = json.load(open(_LICH_FILE)).get("lich", {})
        # Lịch định dạng CŨ chỉ có "thu" (một thứ duy nhất) — sai với các đài quay
        # hai ngày/tuần. Nếu gặp, DỰNG LẠI thay vì âm thầm dùng dữ liệu hỏng.
        thieu = [k for k, v in cu.items() if "cac_thu" not in v]
        if cu and not thieu:
            _LICH_MEM = cu
            return _LICH_MEM
        if cu:
            print(f"  Lịch quay đang ở ĐỊNH DẠNG CŨ ({len(thieu)}/{len(cu)} đài thiếu"
                  f" trường cac_thu) — dựng lại để bắt đúng đài quay 2 ngày/tuần.")
    if not im_lang:
        print("  Đang dựng lịch quay 36 đài (chỉ làm 1 lần, ~1 phút)...")
    lich = {}
    con_thieu = list(range(1, len(DAI_LIST) + 1))
    for vong in range(3):                       # thử tối đa 3 vòng
        if not con_thieu:
            break
        if vong and not im_lang:
            print(f"  Thử lại {len(con_thieu)} đài lỗi (vòng {vong+1}/3)...")
        lan_nay = []
        for stt in con_thieu:
            ten, ma, mien, _ = lay_dai(stt)
            try:
                cac_thu, nm = _thu_cua_dai(stt)
                lich[str(stt)] = {"ten": ten, "ma": ma, "mien": mien,
                                  "thu": int(cac_thu[0]),          # tương thích ngược
                                  "cac_thu": [int(x) for x in cac_thu],
                                  "thu_vn": ", ".join(THU_VN[x] for x in cac_thu)}
                if not im_lang:
                    print(f"     {stt:>2}. {ten:<22} "
                          f"{', '.join(THU_VN[x] for x in cac_thu)}")
            except Exception as e:
                lan_nay.append(stt)
                if not im_lang: print(f"     {stt:>2}. {ten:<22} LỖI: {e}")
            time.sleep(0.5 if vong == 0 else 2.0)
        con_thieu = lan_nay

    if con_thieu:
        # KHÔNG im lặng bỏ qua — đài thiếu sẽ biến mất khỏi MỌI lần chạy sau
        print("\n" + "!" * 74)
        print(f"  ⚠ THIẾU {len(con_thieu)}/36 ĐÀI TRONG LỊCH QUAY")
        for stt in con_thieu:
            print(f"      [{stt:>2}] {DAI_LIST[stt-1][0]}  (mã {DAI_LIST[stt-1][1]})")
        print("  Các đài này sẽ KHÔNG BAO GIỜ được dự báo cho tới khi lịch được dựng lại.")
        print("  Cách sửa: xoá data/lich_quay.json rồi chạy lại workflow.")
        print("!" * 74 + "\n")
    os.makedirs(os.path.dirname(_LICH_FILE) or ".", exist_ok=True)
    json.dump({"tao_luc": time.strftime("%Y-%m-%d %H:%M"), "lich": lich},
              open(_LICH_FILE, "w"), ensure_ascii=False, indent=1)
    # Tóm tắt để đối chiếu ngay trong log
    if not im_lang:
        dem = {}
        for v in lich.values():
            if v["mien"] == "Miền Bắc": continue
            for t in v["cac_thu"]:
                dem[t] = dem.get(t, 0) + 1
        print("\n  Số đài MN/MT quay mỗi thứ (chưa kể Miền Bắc):")
        for t in range(7):
            mong = SO_DAI_MOI_THU.get(t)
            co = dem.get(t, 0)
            dau = "✓" if mong and co >= mong else "⚠"
            print(f"     {dau} {THU_VN[t]:<10} {co} đài" +
                  (f"  (mong đợi {mong})" if mong and co < mong else ""))
        nhieu = [(v["ten"], v["thu_vn"]) for v in lich.values() if len(v["cac_thu"]) > 1]
        if nhieu:
            print(f"\n  Đài quay NHIỀU NGÀY/tuần ({len(nhieu)} đài):")
            for ten, tv in nhieu:
                print(f"     {ten:<24} {tv}")
    _LICH_MEM = lich
    return lich


# Số đài MN/MT quay mỗi thứ (KHÔNG kể Miền Bắc), dùng làm CẬN DƯỚI để phát hiện
# lịch bị thiếu. Đã tính cả các đài quay hai ngày/tuần.
SO_DAI_MOI_THU = {0: 5, 1: 5, 2: 6, 3: 6, 4: 5, 5: 7, 6: 6}   # T2..CN


def dai_theo_ngay(ngay, gom_mien_bac=True):
    """Trả list stt các đài quay vào NGÀY đó. Miền Bắc (36) quay hàng ngày."""
    lich = xay_lich()
    thu = ngay.weekday()
    ra = [int(k) for k, v in lich.items()
          if thu in v.get("cac_thu") and v["mien"] != "Miền Bắc"]
    uu_tien = {"Miền Nam": 0, "Miền Trung": 1, "Miền Bắc": 2}
    ra.sort(key=lambda s: (uu_tien[lich[str(s)]["mien"]], lich[str(s)]["ten"]))
    # --- Đối chiếu với số đài mong đợi, cảnh báo nếu thiếu ---
    mong_doi = SO_DAI_MOI_THU.get(thu)
    if mong_doi and len(ra) < mong_doi:
        co = {int(k) for k in lich}
        vang = [s for s in range(1, 36) if s not in co]
        print("\n" + "!" * 74)
        print(f"  ⚠ CHỈ TÌM ĐƯỢC {len(ra)} ĐÀI cho {THU_VN[thu]}, đáng lẽ có {mong_doi}")
        if vang:
            print(f"  Lịch quay đang THIẾU {len(vang)} đài (không quét được lúc dựng lịch):")
            for s2 in vang:
                print(f"      [{s2:>2}] {DAI_LIST[s2-1][0]}")
        print("  → Xoá data/lich_quay.json rồi chạy lại workflow để dựng lại lịch.")
        print("!" * 74 + "\n")
    if gom_mien_bac and 36 not in ra:
        ra.append(36)
    return ra


def cat_truoc_ngay(toan_giai, ngay_full, ngay_moc, so_ky):
    """CHỐNG RÒ RỈ DỮ LIỆU TƯƠNG LAI.

    Nếu kỳ quay của ngay_moc ĐÃ diễn ra, trang nguồn sẽ trả về nó trong danh
    sách. Huấn luyện trên nó rồi 'dự báo' chính nó là gian lận với chính mình:
    mô hình đã nhìn thấy đáp án. Hàm này cắt bỏ mọi kỳ TỪ ngay_moc trở đi và
    tách riêng kỳ của ngay_moc (nếu có) để đối chiếu.

    Trả (toan_giai_train, ngay_train, ket_qua_that_hoac_None).
    """
    train_tg, train_ng, that = [], [], None
    for tg, d in zip(toan_giai, ngay_full):
        if d is None:
            continue
        if d < ngay_moc:
            train_tg.append(tg); train_ng.append(d)
        elif d == ngay_moc:
            that = tg
    return train_tg[-so_ky:], train_ng[-so_ky:], that


def nap_null_cache():
    global _NULL_MEM
    if _NULL_MEM is not None:
        return _NULL_MEM
    _NULL_MEM = json.load(open(_NULL_FILE))["cache"] if os.path.exists(_NULL_FILE) else {}
    return _NULL_MEM


# ==============================================================================
#  [12B] GỬI EMAIL
#        Dùng SMTP Gmail. Mật khẩu ỨNG DỤNG (16 ký tự), không phải mật khẩu
#        đăng nhập. Colab là môi trường chia sẻ — đừng dán mật khẩu vào code,
#        hãy để trống EMAIL_MAT_KHAU và nhập khi được hỏi (ô nhập sẽ ẩn ký tự).
# ==============================================================================

import smtplib, getpass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr


_MK_PHIEN = None      # nhớ mật khẩu trong phiên Colab, không ghi ra đĩa


def _lay_mat_khau():
    """Ba nguồn, ưu tiên từ an toàn nhất xuống:
       1. Colab Secrets  — lưu trong tài khoản Colab, KHÔNG nằm trong notebook.
                           Nhập một lần, mọi phiên sau tự dùng. An toàn nhất.
       2. EMAIL_MAT_KHAU — ghi thẳng trong code. Tiện nhưng lộ nếu chia sẻ notebook.
       3. Hỏi khi chạy   — getpass, ẩn ký tự, không lưu ra đĩa.
    """
    global _MK_PHIEN
    # (1) GitHub Secrets qua biến môi trường — cách dùng trên Actions
    mk = os.environ.get(TEN_SECRET) or os.environ.get("MAIL_APP_PASSWORD")
    if mk:
        return mk.replace(" ", "")
    # (2) Colab Secrets — nếu ai đó chạy file này trên Colab
    try:
        from google.colab import userdata
        mk = userdata.get(TEN_SECRET)
        if mk:
            return mk.replace(" ", "")
    except Exception:
        pass
    # (3) Ghi trong code (không dùng trên repo công khai)
    if EMAIL_MAT_KHAU:
        return EMAIL_MAT_KHAU.replace(" ", "")
    raise RuntimeError(
        "Thiếu mật khẩu ứng dụng Gmail.\n"
        "  Trên GitHub: Settings > Secrets and variables > Actions > New repository\n"
        "  secret, tên GMAIL_APP_PASSWORD, giá trị là chuỗi 16 chữ cái thường.")


def auto_tune7(n):
    """(số lượt null, trần điểm test, số kỳ train tối thiểu) theo cỡ dữ liệu."""
    if n <= 60:   return 60, None, 15
    if n <= 120:  return 60, 70, 30
    return 40, 90, 40


# ==============================================================================
#  [15] NULL CONTROL MIN-P CHO 3 MODULE
#       Ba module có mốc ngẫu nhiên khác nhau (50% / 10% / 5%). Lấy max(hit rate)
#       là SAI vì module mốc cao luôn thắng bất kể chất lượng. Min-p quy mỗi
#       module về p-value theo phân phối null của chính nó rồi lấy p nhỏ nhất.
# ==============================================================================

def null_ho_v8(n_runs, n_ky, co_cau, cau_hinh, min_train, max_pts, rng=None):
    """cau_hinh: list (key, loai, vt, K, cap). loai ∈ {'2so','3so'}."""
    rng = rng or rng_global
    out = []
    for i in range(n_runs):
        gia = [sinh_ky_gia(co_cau, rng) for _ in range(n_ky)]
        pool = [[int(s[-2:]) for s in ky] for ky in gia]
        g3 = [lo3_cua_ky(ky) for ky in gia]
        g4 = [lo4_cua_ky(ky) for ky in gia]
        hang = []
        for key, loai, vt, K, cap in cau_hinh:
            if loai == "4so":
                hang.append(toi_uu_trong_so_4(
                    precompute_windows_4(g4, min_train=min_train, max_pts=max_pts), K, cap)[1])
            elif loai == "3so":
                hang.append(toi_uu_trong_so_3(
                    precompute_windows_3(g3, min_train=min_train, max_pts=max_pts,
                                         toan_giai=gia), K, cap)[1])
            else:
                ch = gia if vt is None else [[ky[vt]] for ky in gia]
                mtc = [{int(k[0][-2:])} for k in gia] if vt == 0 else None
                hang.append(toi_uu(cua_so(ch, pool, min_train, max_pts,
                                          toan_giai=gia, mt_cau=mtc), K, cap)[1])
        out.append(hang)
        if (i + 1) % max(1, n_runs // 10) == 0:
            print(f"     ... {i+1}/{n_runs}", end="\r")
    print(" " * 46, end="\r")
    return np.array(out)


def khoa_v8(khu, n, Ks, min_train, max_pts):
    raw = (f"v8|{khu}|{n}|{Ks['DB']}|{Ks['LO2']}|{Ks['3SO']}|{Ks.get('4SO',0)}"
           f"|{min_train}|{max_pts}")
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def cau_hinh_module(Ks):
    ch = []
    if Ks.get("DB"):  ch.append(("DB",  "2so", 0,    Ks["DB"],  tinh_cap(Ks["DB"])))
    if Ks.get("LO2"): ch.append(("LO2", "2so", None, Ks["LO2"], tinh_cap(Ks["LO2"])))
    if Ks.get("3SO"): ch.append(("3SO", "3so", None, Ks["3SO"], tinh_cap3(Ks["3SO"])))
    if Ks.get("4SO"): ch.append(("4SO", "4so", None, Ks["4SO"], tinh_cap4(Ks["4SO"])))
    return ch


def bao_dam_cache_v8(khus, so_ky, Ks):
    cache = nap_null_cache()
    _, mp, mt = auto_tune7(so_ky)
    thieu = [k for k in khus if khoa_v8(k, so_ky, Ks, mt, mp) not in cache]
    if thieu:
        print(f"\n  Đang tính null cache cho {thieu}.")
        print(f"  CHỈ LÀM MỘT LẦN cho mỗi cấu hình (~6 phút/khu). Các lần sau ~1 phút.")
        for khu in thieu:
            t = time.time()
            arr = null_ho_v8(*[auto_tune7(so_ky)[0], so_ky, CO_CAU_GIAI[khu],
                               cau_hinh_module(Ks), mt, mp])
            cache[khoa_v8(khu, so_ky, Ks, mt, mp)] = arr.tolist()
            print(f"     {khu}: xong ({len(arr)} lượt, {time.time()-t:.0f}s)")
        os.makedirs(os.path.dirname(_NULL_FILE) or ".", exist_ok=True)
        json.dump({"meta": {"so_ky": so_ky, "ks": Ks}, "cache": cache}, open(_NULL_FILE, "w"))
        print(f"  → Đã lưu {_NULL_FILE}. Tải file này về, phiên sau upload lại để chạy nhanh.")
    return cache


# ==============================================================================
#  [16] CHẠY 1 ĐÀI — CẢ 3 MODULE
# ==============================================================================

MO_TA = {"DB":  ("① ĐỀ ĐẶC BIỆT",  "2 chữ số CUỐI của giải ĐẶC BIỆT"),
         "LO2": ("② BAO LÔ 2 SỐ",  "2 chữ số CUỐI của BẤT KỲ giải nào"),
         "3SO": ("③ LÔ 3 SỐ",      "3 chữ số CUỐI của mọi giải từ 3 chữ số trở lên"),
         "4SO": ("④ LÔ 4 SỐ",      "4 chữ số CUỐI của mọi giải từ 4 chữ số trở lên")}


def chay_dai_v8(stt, ngay_moc, so_ky, Ks, cache, dung_master=True):
    ten, ma, mien, nd = lay_dai(stt)
    tu_kho = False
    if dung_master:
        # Lấy dư 40 kỳ để sau khi cắt theo ngày mốc vẫn đủ so_ky
        m = lay_tu_master(stt, so_ky + 40)
        if m is None:
            m = lay_tu_master(stt, so_ky)
        if m:
            toan_giai, ngay_full, info = m
            ngay = [d.strftime("%d/%m") for d in ngay_full]
            tu_kho = True
    if not tu_kho:
        du = min(200, so_ky + 40)
        _, ngay, ngay_full, toan_giai, info = lay_du_lieu(stt, du)
    if not toan_giai:
        raise RuntimeError(
            f"không bóc được toàn bảng giải. Bóc theo nhãn={info.get('n_FA')} kỳ, "
            f"theo chữ ký độ dài={info.get('n_FB')} kỳ, dòng ĐB={info.get('n_A')} kỳ. "
            f"Nếu cả 3 đều 0 thì trang {info.get('url')} đổi cấu trúc HTML. "
            f"Nếu chỉ n_FA/n_FB = 0 thì cơ cấu giải của đài này khác chuẩn "
            f"{sum(q for q,_ in CO_CAU_GIAI[info.get('khu','MN')].values())} số.")
    n_co = len(toan_giai)
    tg, ng, that = cat_truoc_ngay(toan_giai, ngay_full, ngay_moc, so_ky)

    # Nếu quá ít kỳ đọc được ngày, thử lại với trang NHỎ HƠN — trang ít kỳ thường
    # có bố cục đơn giản hơn, tỷ lệ đọc được ngày cao hơn. Thà chạy 60 kỳ còn hơn
    # loại cả đài.
    if len(tg) < 40:
        for du_nho in (100, 90, 30):
            if du_nho >= du:
                continue
            try:
                _, ng2, nf2, tg2, info2 = lay_du_lieu(stt, du_nho)
                if not tg2:
                    continue
                t3, n3, th3 = cat_truoc_ngay(tg2, nf2, ngay_moc, so_ky)
                if len(t3) > len(tg):
                    print(f"       [{stt}] hạ xuống {du_nho} kỳ → dùng được "
                          f"{len(t3)} kỳ (trước đó {len(tg)})")
                    tg, ng, that, info = t3, n3, th3, info2
                if len(tg) >= 40:
                    break
            except Exception:
                continue

    if len(tg) < 40:
        ngays = [d for d in ngay_full if d]
        raise RuntimeError(
            f"chỉ còn {len(tg)} kỳ trước {ngay_moc:%d.%m.%Y} (cần >=40). "
            f"Tải về {n_co} kỳ, khoảng ngày "
            f"{min(ngays) if ngays else '?'} → {max(ngays) if ngays else '?'}. "
            f"Có {sum(1 for d in ngay_full if d is None)} kỳ không đọc được ngày.")

    khu = "MB" if ma == "xsmb" else "MN"
    n = len(tg); n_lo2 = len(tg[0]); n_lo3 = len(lo3_cua_ky(tg[0]))
    pool = [[int(s[-2:]) for s in ky] for ky in tg]
    g3 = [lo3_cua_ky(ky) for ky in tg]
    g4 = [lo4_cua_ky(ky) for ky in tg]
    n_lo4 = len(g4[0])
    _, mp, mt = auto_tune7(so_ky)

    mods = []
    for key, loai, vt, K, cap in cau_hinh_module(Ks):
        if loai == "4so":
            wins = precompute_windows_4(g4, min_train=mt, max_pts=mp)
            w, hr, h, per = toi_uu_trong_so_4(wins, K, cap)
            Zf, lam = build_signal_matrix_4(g4)
            sel, _ = select_top4(w @ Zf, K, cap)
            so = sorted(f"{i:04d}" for i in sel)
            lo_tien = SO_LO_4SO_MB if khu == "MB" else SO_LO_4SO_MN
            mt_per, p0, von = n_lo4, K/10000., lo_tien*K*DIEM_4SO
            tin = [nm for nm, wv in zip(SIG4_NAMES, w) if wv > 0]
            tra = TY_LE_TRA_4SO
            d_con = DIEM_4SO
        elif loai == "3so":
            wins = precompute_windows_3(g3, min_train=mt, max_pts=mp, toan_giai=tg)
            w, hr, h, per = toi_uu_trong_so_3(wins, K, cap)
            Zf, lam = build_signal_matrix_3(g3, tg)
            sel, _ = select_top3(w @ Zf, K, cap)
            so = sorted(f"{i:03d}" for i in sel)
            lo_tien = SO_LO_3SO_MB if khu == "MB" else SO_LO_3SO_MN
            mt_per, p0, von = n_lo3, K/1000., lo_tien*K*DIEM_3SO
            tin = [nm for nm, wv in zip(SIG3_NAMES, w) if wv > 0]
            tra = TY_LE_TRA_3SO
            d_con = DIEM_3SO
        else:
            ch = tg if vt is None else [[ky[vt]] for ky in tg]
            mtc = [{v} for v in [int(k[0][-2:]) for k in tg]] if vt == 0 else None
            wins = cua_so(ch, pool, mt, mp, toan_giai=tg, mt_cau=mtc)
            w, hr, h, per = toi_uu(wins, K, cap)
            Zf, _ = ma_tran_tin_hieu(ch, pool, tg, mtc)
            sel, _ = select_top(w @ Zf, K, cap)
            so = sorted(f"{i:02d}" for i in sel)
            mt_per = n_lo2 if vt is None else 1
            p0 = K/100.
            if key == "LO2":
                lo_tien = SO_LO_2SO_MB if khu == "MB" else SO_LO_2SO_MN
                von = lo_tien * K * DIEM_LO2
            else:
                von = K * DIEM_DB
            tin = [nm for nm, wv in zip(SIG_NAMES, w) if wv > 0]
            tra = TY_LE_TRA_2SO
            d_con = DIEM_LO2 if key == "LO2" else DIEM_DB
        mods.append({"key": key, "ten": MO_TA[key][0], "mo_ta": MO_TA[key][1],
                     "K": K, "hr": hr, "hits": h, "nt": len(wins), "mt": mt_per,
                     "p0": p0, "von": von, "tra": tra, "diem": d_con,
                     "so": so, "tin_hieu": tin,
                     "tong": len(wins)*mt_per})

    kh = khoa_v8(khu, so_ky, Ks, mt, mp)
    if kh in cache:
        null = np.array(cache[kh])
    else:
        null = null_ho_v8(auto_tune7(so_ky)[0], n, CO_CAU_GIAI[khu],
                          cau_hinh_module(Ks), mt, mp)
    p_tho, p_bon, p_ho = p_min_ho(null, [m["hr"] for m in mods])
    for j, m in enumerate(mods):
        m["p_tho"], m["p_bon"], m["p_ho"] = float(p_tho[j]), float(p_bon[j]), float(p_ho[j])
        m["nhieu_tb"] = float(null[:, j].mean())

    cb, thu_moi, _ = kiem_tra_toan_ven(ng, info["lich"])
    return {"stt": stt, "dai": ten, "ma": ma, "mien": mien, "khu": khu,
            "tu_kho": tu_kho, "_toan_giai": tg,
            "n_ky": n, "n_lo2": n_lo2, "n_lo3": n_lo3,
            "ngay_ky_truoc": ng[-1].strftime("%d/%m/%Y"),
            "db_ky_truoc": tg[-1][0], "modules": mods, "canh_bao": cb,
            "ket_qua_that": that,
            "tong_von": sum(m["von"] for m in mods),
            "tong_ev": sum(m["p0"]*m["mt"]*m["tra"]*m["diem"] - m["von"] for m in mods)}


# ==============================================================================
#  [17] EMAIL V8
# ==============================================================================

def html_v8(ket, ngay_moc, tong_von, tong_ev, loi, dong_dc):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    thu = THU_VN[ngay_moc.weekday()]
    h = [f'<div style="{css}max-width:700px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">Dự báo {thu} {ngay_moc:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài &middot; {sum(len(r["modules"]) for r in ket)} bộ số</p>')
    h.append(f'<div style="background:#fff8e1;border-left:4px solid #f9a825;'
             f'padding:10px 13px;margin:0 0 22px;font-size:13px;line-height:1.6">'
             f'<b>TỔNG VỐN {tong_von:,}/ngày &middot; kỳ vọng {tong_ev:+,.0f} '
             f'({tong_ev/max(tong_von,1):+.1%})</b><br>'
             f'Với {sum(len(r["modules"]) for r in ket)} bộ số, gần như ngày nào cũng có '
             f'bộ trúng. Hãy theo dõi con số tổng tích luỹ, đừng theo dõi từng bộ.</div>')

    for r in ket:
        h.append(f'<div style="margin:26px 0 0;padding:9px 12px;background:#263238;'
                 f'color:#fff;border-radius:5px 5px 0 0">'
                 f'<span style="font-size:17px;font-weight:700">{r["dai"].upper()}</span>'
                 f'<span style="font-size:13px;opacity:.85"> &nbsp;|&nbsp; {r["mien"]} '
                 f'&nbsp;|&nbsp; {thu} {ngay_moc:%d.%m.%Y}</span><br>'
                 f'<span style="font-size:12px;opacity:.7">Kỳ gần nhất '
                 f'{r["ngay_ky_truoc"]} &middot; ĐB {r["db_ky_truoc"]} &middot; '
                 f'{r["n_ky"]} kỳ huấn luyện</span></div>')
        h.append('<div style="border:1px solid #cfd8dc;border-top:0;'
                 'border-radius:0 0 5px 5px;padding:4px 12px 12px">')
        for m in r["modules"]:
            h.append(f'<div style="margin:12px 0 0">'
                     f'<div style="font-size:14px;font-weight:600;margin-bottom:2px">'
                     f'{m["ten"]} &mdash; {m["K"]} con</div>'
                     f'<div style="font-size:12px;color:#607d8b;margin-bottom:5px">'
                     f'{r["dai"]} &middot; {thu} {ngay_moc:%d.%m.%Y} &middot; '
                     f'{m["mo_ta"]} &middot; {m["diem"]} điểm/con &middot; vốn {m["von"]:,} &middot; '
                     f'thưởng {m["tra"]:.0f} &middot; p HỌ {m["p_ho"]:.3f}</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:14px;background:#eceff1;border-left:3px solid #455a64;'
                     f'padding:9px 11px;word-spacing:2px;line-height:1.75;'
                     f'word-break:break-all">{",".join(m["so"])}</div></div>')
        for c in r["canh_bao"]:
            h.append(f'<p style="color:#c62828;font-size:13px;margin:8px 0 0">⚠ {c}</p>')
        h.append('</div>')

    if dong_dc:
        h.append('<h3 style="margin:26px 0 4px">Đối chiếu kỳ đã quay</h3>'
                 '<p style="font-size:12px;color:#666;margin:0 0 8px">Kỳ này đã bị loại '
                 'khỏi dữ liệu huấn luyện — kiểm tra ngoài mẫu thật sự.</p>'
                 '<ul style="font-size:13px;line-height:1.7">')
        for d in dong_dc: h.append(f'<li>{d}</li>')
        h.append('</ul>')
    if loi:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;'
                 'padding:10px 13px;margin:18px 0;font-size:13px;line-height:1.6">'
                 '<b>Đài chạy lỗi — KHÔNG có bộ số:</b><ul style="margin:6px 0">')
        for s_, t_, e_ in loi:
            h.append(f'<li><b>[{s_}] {t_}</b><br>'
                     f'<span style="color:#666;font-family:monospace;font-size:12px">'
                     f'{e_}</span></li>')
        h.append('</ul></div>')
    h.append(f'<hr style="margin:26px 0 10px;border:0;border-top:1px solid #ddd">'
             f'<p style="font-size:12px;color:#888;line-height:1.6">'
             f'Kỳ vọng: đề/bao lô 2 số {TY_LE_TRA_2SO:.0f}/100 &minus; 1 = '
             f'{TY_LE_TRA_2SO/100-1:+.1%} &middot; lô 3 số {TY_LE_TRA_3SO:.0f}/1000 &minus; 1 '
             f'= {TY_LE_TRA_3SO/1000-1:+.1%}. Không phụ thuộc số con hay thuật toán.<br>'
             f'Cột p HỌ đã hiệu chỉnh so sánh bội bằng min-p; từ 0,05 trở lên nghĩa là '
             f'không phân biệt được với chọn ngẫu nhiên.</p></div>')
    return "".join(h)


def gui_email_v8(ket, ngay_moc, tong_von, tong_ev, loi, dong_dc):
    mk = _lay_mat_khau()
    if not mk:
        print("  ⚠ Chưa có mật khẩu ứng dụng — bỏ qua gửi email."); return False
    thu = THU_VN[ngay_moc.weekday()]
    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"[XSMN] {thu} {ngay_moc:%d.%m.%Y} — {len(ket)} đài, "
                      f"{sum(len(r['modules']) for r in ket)} bộ số")
    msg["From"] = formataddr(("XSMN v8", EMAIL_GUI)); msg["To"] = EMAIL_NHAN
    t = [f"DỰ BÁO {thu} {ngay_moc:%d.%m.%Y}",
         f"TỔNG VỐN {tong_von:,}/ngày — kỳ vọng {tong_ev:+,.0f} "
         f"({tong_ev/max(tong_von,1):+.1%})", ""]
    for r in ket:
        t.append("=" * 60)
        t.append(f"{r['dai'].upper()} | {r['mien']} | {thu} {ngay_moc:%d.%m.%Y}")
        t.append(f"Kỳ gần nhất {r['ngay_ky_truoc']} — ĐB {r['db_ky_truoc']}")
        for m in r["modules"]:
            t.append("")
            t.append(f"{m['ten']} — {m['K']} con — {r['dai']} — {thu} {ngay_moc:%d.%m.%Y}")
            t.append(f"({m['mo_ta']}; vốn {m['von']:,}; p HỌ {m['p_ho']:.3f})")
            t.append(",".join(m["so"]))
        t.append("")
    msg.attach(MIMEText("\n".join(t), "plain", "utf-8"))
    msg.attach(MIMEText(html_v8(ket, ngay_moc, tong_von, tong_ev, loi, dong_dc),
                        "html", "utf-8"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
            sv.login(EMAIL_GUI, mk); sv.send_message(msg)
        print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}"); return True
    except smtplib.SMTPAuthenticationError:
        globals()["_MK_PHIEN"] = None
        print("  ✗ Sai mật khẩu ứng dụng.")
        print("    Mật khẩu ỨNG DỤNG là 16 CHỮ CÁI THƯỜNG, không có số/ký tự đặc biệt.")
        print("    Tạo tại: Tài khoản Google > Bảo mật > Xác minh 2 bước > Mật khẩu ứng dụng")
    except Exception as e:
        print(f"  ✗ Không gửi được email: {e}")
    return False


# ==============================================================================
#  [18] PIPELINE CHÍNH — TỰ LẤY NGÀY HÔM NAY, CHẠY TẤT CẢ ĐÀI
# ==============================================================================

def main(ngay=None, so_ky=None, so_con_db=None, so_con_lo2=None, so_con_3so=None,
         so_con_4so=None):
    # --- TASK 1: ngày hôm nay theo GIỜ VIỆT NAM (máy chủ Colab chạy giờ UTC) ---
    GIO_VN = timezone(timedelta(hours=7))
    ngay_moc = doc_ngay(ngay) if ngay else datetime.now(GIO_VN).date()
    so_ky = so_ky or SO_KY
    Ks = {"DB":  so_con_db  if so_con_db  is not None else SO_CON_DB,
          "LO2": so_con_lo2 if so_con_lo2 is not None else SO_CON_LO2,
          "3SO": so_con_3so if so_con_3so is not None else SO_CON_3SO,
          "4SO": so_con_4so if so_con_4so is not None else SO_CON_4SO}
    if not any(Ks.values()):
        raise ValueError("Phải bật ít nhất một module")

    thu = THU_VN[ngay_moc.weekday()]
    print("=" * 78)
    print(f"  XSMN v8  |  {thu} {ngay_moc:%d.%m.%Y} (giờ VN)  |  {so_ky} kỳ/đài")
    print(f"  Module: " + "  ".join(f"{MO_TA[k][0]}={v}" for k, v in Ks.items() if v))
    _n = len(_chi_so_tin_hieu())
    print(f"  Tín hiệu 2 số: bộ {BO_TIN_HIEU} — {_n} tín hiệu, {2**_n-1} tổ hợp dò")
    print("=" * 78)

    # --- TASK 2 + 3: tìm đài quay hôm nay, luôn kèm đài 36 Miền Bắc ---
    dsach = dai_theo_ngay(ngay_moc)
    lich = xay_lich()
    print(f"\n  Đài quay {thu} {ngay_moc:%d.%m.%Y}: {len(dsach)} đài")
    for s in dsach:
        print(f"     [{s:>2}] {lich[str(s)]['ten']:<24}{lich[str(s)]['mien']}")

    kho = doc_master()
    if kho:
        print(f"\n  Kho dữ liệu: {len(kho['dai'])} đài, cập nhật {kho['tao_luc'][:16]}")
        kiem_tra_do_moi(ngay_moc)
    else:
        print(f"\n  Chưa có kho dữ liệu — sẽ tải trực tiếp từ web (chậm hơn).")
        print(f"  Chạy tao_master() một lần để rút ngắn các lần sau.")

    khus = sorted({("MB" if lay_dai(s)[1] == "xsmb" else "MN") for s in dsach})
    cache = bao_dam_cache_v8(khus, so_ky, Ks)

    # --- TASK 4-7: chạy 3 module cho từng đài ---
    print(f"\n  Đang phân tích...")
    ket, loi = [], []
    for s in dsach:
        ten = lich[str(s)]["ten"]
        try:
            t0 = time.time()
            r = chay_dai_v8(s, ngay_moc, so_ky, Ks, cache)
            ket.append(r)
            kt = f" | ĐÃ QUAY: ĐB {r['ket_qua_that'][0]}" if r["ket_qua_that"] else ""
            print(f"     ✓ [{s:>2}] {ten:<24}{time.time()-t0:5.1f}s  "
                  f"({r['n_ky']} kỳ ≤ {r['ngay_ky_truoc']}"
                  f"{', từ KHO' if r.get('tu_kho') else ', từ WEB'}){kt}")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<24}LỖI:")
            for dong in str(e).split(". "):
                if dong.strip(): print(f"          {dong.strip()}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")

    # --- TASK 8 + 10: bảng tổng hợp ---
    print(f"\n[A] TỔNG HỢP")
    print("  LƯU Ý: cột p HỌ dưới đây tính từ hit rate của backtest có RÒ RỈ chọn")
    print("  trọng số (dùng cả 70 kỳ test để chọn tham số). Để có con số SẠCH, chạy:")
    print("     chay_backtest(chon_dai=<số>, so_ky=100)")
    print("-" * 78)
    print(f"  {'Đài':<20}{'Module':<18}{'Con':>4}{'Điểm':>6}{'Vốn':>8}{'p HỌ':>8}  Kỳ trước")
    print("  " + "-" * 74)
    tong_von = tong_ev = 0
    for r in ket:
        for i, m in enumerate(r["modules"]):
            print(f"  {r['dai'] if i==0 else '':<20}{m['ten']:<18}{m['K']:>4}"
                  f"{m['diem']:>6}{m['von']:>8,}{m['p_ho']:>8.3f}  "
                  f"{r['ngay_ky_truoc'] if i==0 else ''}")
        for c in r["canh_bao"]: print(f"     ⚠ {c}")
        tong_von += r["tong_von"]; tong_ev += r["tong_ev"]
    print("  " + "-" * 74)
    print(f"  {'TỔNG CỘNG':<48}{tong_von:>8,}  kỳ vọng {tong_ev:+,.0f}/ngày "
          f"= {tong_ev/tong_von:+.1%}")
    if loi:
        print(f"  Đài lỗi: " + ", ".join(f"[{s}] {t}" for s, t, _ in loi))

    # --- Đối chiếu nếu kỳ đó đã quay ---
    da_quay = [r for r in ket if r["ket_qua_that"]]
    dong_dc = []
    if da_quay:
        print(f"\n[B] ĐỐI CHIẾU — kỳ {ngay_moc:%d.%m.%Y} đã quay")
        print("-" * 78)
        t_von = t_thu = 0
        for r in da_quay:
            g = r["ket_qua_that"]
            for m in r["modules"]:
                bo = set(m["so"])
                if m["key"] == "DB":    thuc = [g[0][-2:]]
                elif m["key"] == "LO2": thuc = [x[-2:] for x in g]
                elif m["key"] == "4SO": thuc = lo4_cua_ky(g)
                else:                   thuc = lo3_cua_ky(g)
                h = sum(1 for x in thuc if x in bo)
                thu_ = h * m["tra"]; t_von += m["von"]; t_thu += thu_
                print(f"  {r['dai']:<20}{m['ten']:<18} trúng {h:>2}  "
                      f"lãi {thu_-m['von']:+9,.0f}")
                dong_dc.append(f"{r['dai']} — {m['ten']}: trúng {h}, "
                               f"lãi {thu_-m['von']:+,.0f}")
        print("  " + "-" * 74)
        print(f"  TỔNG: vốn {t_von:,} | thu {t_thu:,.0f} | lãi {t_thu-t_von:+,.0f} "
              f"= {(t_thu-t_von)/t_von:+.1%}")
        print(f"  Đây là MỘT ngày. Phương sai ở cỡ mẫu này rất lớn — lãi hay lỗ")
        print(f"  của một ngày đều KHÔNG nói lên điều gì.")
        dong_dc.append(f"<b>TỔNG: lãi {t_thu-t_von:+,.0f} = {(t_thu-t_von)/t_von:+.1%}</b>"
                       f" — một ngày, chưa nói lên điều gì")

    # --- TASK 8: các vùng số ---
    for r in ket:
        for m in r["modules"]:
            print("\n" + "=" * 78)
            print(f"  {r['dai'].upper()}  —  {m['ten']}  ({m['K']} con)")
            print(f"  {thu} {ngay_moc:%d.%m.%Y}  |  {r['mien']}  |  {m['mo_ta']}")
            print(f"  Vốn {m['von']:,}  |  thưởng {m['tra']:.0f}/lần  |  p HỌ {m['p_ho']:.3f}")
            if m["key"] == "4SO":
                md = r["n_ky"] * m["mt"] / 10000
                print(f"  ⚠ MẬT ĐỘ {md:.2f} mẫu/ô — thấp nhất trong mọi module. "
                      f"~{10000*(1-1/10000)**(r['n_ky']*m['mt']):,.0f}/10.000 con chưa từng ra.")
                print(f"  ⚠ EV = {TY_LE_TRA_4SO:.0f}/10000 − 1 = {TY_LE_TRA_4SO/10000-1:+.2%} "
                      f"— tệ nhất trong mọi loại cược.")
            print("=" * 78)
            moi_hang = 10
            for i in range(0, len(m["so"]), moi_hang):
                print("   " + "  ".join(m["so"][i:i+moi_hang]))
    print("\n" + "=" * 78)

    # --- TASK 9: gửi email ---
    if GUI_EMAIL:
        print("\n  Đang gửi email...")
        gui_email_v8(ket, ngay_moc, tong_von, tong_ev, loi, dong_dc)

    globals()["_KET_QUA_LAN_CUOI"] = ket
    return {r["dai"]: {m["ten"]: m["so"] for m in r["modules"]} for r in ket}




# ==============================================================================
#  [19] BACKTEST LÃI/LỖ THẬT — WALK-FORWARD LỒNG NHAU
#
#  VÁ LỖ HỔNG của backtest cũ: hàm toi_uu() chọn bộ trọng số tốt nhất bằng cách
#  chấm trên TOÀN BỘ điểm test — tức là dùng cả dữ liệu tương lai để chọn tham số.
#  Đó là rò rỉ thông tin, và nó thổi phồng hit rate.
#
#  Ở đây, tại mỗi kỳ t: trọng số chỉ được chọn từ dữ liệu TRƯỚC t. Không có
#  bất kỳ thông tin nào của kỳ t hay sau đó tham gia vào quyết định.
#
#  Kèm ĐỐI CHỨNG NGẪU NHIÊN: cùng cơ cấu cược, nhưng bộ số bốc ngẫu nhiên.
#  Hai đường lãi/lỗ đặt cạnh nhau là phép so trực quan nhất, không cần p-value.
# ==============================================================================

def _trong_so_qua_khu(chuoi, pool, g3, t, key, loai, K, cap, min_train, g4=None):
    """Chọn trọng số CHỈ từ dữ liệu trước kỳ t."""
    if loai == "4so":
        w = precompute_windows_4(g4[:t], min_train=min_train, max_pts=None)
        return toi_uu_trong_so_4(w, K, cap)[0] if w else np.ones(len(SIGNALS_4))
    if loai == "3so":
        w = precompute_windows_3(g3[:t], min_train=min_train, max_pts=None)  # cầu tắt khi tối ưu trọng số
        return toi_uu_trong_so_3(w, K, cap)[0] if w else np.ones(len(SIGNALS_3))
    w = cua_so(chuoi[:t], pool[:t], min_train, None)
    return toi_uu(w, K, cap)[0] if w else np.ones(len(SIGNALS))


def backtest_von(toan_giai, Ks, diem, khu="MN", min_train=30, chu_ky_toi_uu=5,
                 n_doi_chung=40, seed=2026, im_lang=False):
    """Mô phỏng lãi/lỗ trên dữ liệu THẬT với đúng cơ cấu cược của bạn.

    Ks   : {"DB": 82, "LO2": 4, "3SO": 50}
    diem : {"DB": 5,  "LO2": 2, "3SO": 1}   điểm cược mỗi con
    chu_ky_toi_uu: chọn lại trọng số mỗi N kỳ (1 = mỗi kỳ, chậm nhất & chuẩn nhất)
    """
    rng = np.random.default_rng(seed)
    n = len(toan_giai)
    n_lo2 = len(toan_giai[0]); n_lo3 = len(lo3_cua_ky(toan_giai[0]))
    pool = [[int(s[-2:]) for s in ky] for ky in toan_giai]
    g3   = [lo3_cua_ky(ky) for ky in toan_giai]
    g4   = [lo4_cua_ky(ky) for ky in toan_giai]
    lo2_tien = SO_LO_2SO_MB if khu == "MB" else SO_LO_2SO_MN
    lo3_tien = SO_LO_3SO_MB if khu == "MB" else SO_LO_3SO_MN
    # (loại, chuỗi nguồn, số lô TÍNH TIỀN, không gian, tỷ lệ trả)
    CH = {"DB":  ("2so", [[k[0]] for k in toan_giai], 1,        100,  TY_LE_TRA_2SO),
          "LO2": ("2so", toan_giai,                   lo2_tien, 100,  TY_LE_TRA_2SO),
          "3SO": ("3so", g3,                          lo3_tien, 1000, TY_LE_TRA_3SO),
          "4SO": ("4so", g4, (SO_LO_4SO_MB if khu == "MB" else SO_LO_4SO_MN),
                  10000, TY_LE_TRA_4SO)}

    ban_ghi = []
    w_luu = {}
    for t in range(min_train, n):
        hang = {"ky": t, "kq": toan_giai[t]}
        for key in ["DB", "LO2", "3SO", "4SO"]:
            if not Ks.get(key): continue
            loai, chuoi, n_mt, KG, tra = CH[key]
            K, d = Ks[key], diem[key]
            cap = (tinh_cap4(K) if loai == "4so"
                   else tinh_cap3(K) if loai == "3so" else tinh_cap(K))
            # --- chọn lại trọng số theo chu kỳ, CHỈ dùng dữ liệu quá khứ ---
            if (t - min_train) % chu_ky_toi_uu == 0 or key not in w_luu:
                w_luu[key] = _trong_so_qua_khu(chuoi, pool, g3, t, key, loai,
                                               K, cap, min_train, g4)
            w = w_luu[key]
            # --- dự báo kỳ t bằng dữ liệu < t ---
            if loai == "4so":
                Z, _ = build_signal_matrix_4(g4[:t])
                sel, _ = select_top4(w @ Z, K, cap)
                thuc = [int(x) for x in g4[t]]
            elif loai == "3so":
                Z, _ = build_signal_matrix_3(g3[:t], toan_giai[:t])
                sel, _ = select_top3(w @ Z, K, cap)
                thuc = [int(x) for x in g3[t]]
            else:
                Z, _ = ma_tran_tin_hieu(chuoi[:t], pool[:t], toan_giai[:t])
                sel, _ = select_top(w @ Z, K, cap)
                thuc = [int(s[-2:]) for s in (toan_giai[t] if key == "LO2"
                                              else [toan_giai[t][0]])]
                bo_nn = rng.choice(100, K, replace=False)
            bo = set(sel.tolist())
            h = sum(1 for x in thuc if x in bo)
            von = K * d * n_mt
            hang[key] = {"trung": h, "von": von, "lai": tra*d*h - von}
            # --- đối chứng ngẫu nhiên: trung bình n_doi_chung bộ bốc bừa ---
            tong = 0
            for _ in range(n_doi_chung):
                b = set(rng.choice(KG, K, replace=False).tolist())
                tong += sum(1 for x in thuc if x in b)
            h_nn = tong / n_doi_chung
            hang[key + "_nn"] = {"trung": h_nn, "lai": tra*d*h_nn - von}
        ban_ghi.append(hang)
        if not im_lang and (t - min_train + 1) % 10 == 0:
            print(f"     ... kỳ {t-min_train+1}/{n-min_train}", end="\r")
    if not im_lang: print(" " * 40, end="\r")
    return ban_ghi


def bao_cao_backtest(bg, Ks, diem, ten_dai=""):
    keys = [k for k in ["DB", "LO2", "3SO", "4SO"] if Ks.get(k)]
    nt = len(bg)
    print("\n" + "=" * 78)
    print(f"  BACKTEST LÃI/LỖ TRÊN DỮ LIỆU THẬT — {ten_dai}")
    print(f"  {nt} kỳ kiểm tra | trọng số chọn CHỈ từ quá khứ | có đối chứng ngẫu nhiên")
    print("=" * 78)
    print(f"  {'Module':<14}{'Vốn/kỳ':>9}{'Tổng vốn':>11}{'Lãi/lỗ':>12}{'%':>8}"
          f"{'Kỳ có lời':>11}{'Đối chứng NN':>14}")
    print("  " + "-" * 74)
    tv = tl = tl_nn = 0
    for k in keys:
        von = bg[0][k]["von"]; tvon = von*nt
        lai = sum(b[k]["lai"] for b in bg)
        lnn = sum(b[k+"_nn"]["lai"] for b in bg)
        col = sum(1 for b in bg if b[k]["lai"] > 0)
        tv += tvon; tl += lai; tl_nn += lnn
        print(f"  {MO_TA[k][0]:<14}{von:>9,}{tvon:>11,}{lai:>+12,.0f}{lai/tvon:>7.1%}"
              f"{col:>7}/{nt:<3}{lnn:>+14,.0f}")
    print("  " + "-" * 74)
    print(f"  {'TỔNG CỘNG':<14}{'':<9}{tv:>11,}{tl:>+12,.0f}{tl/tv:>7.1%}"
          f"{'':>11}{tl_nn:>+14,.0f}")
    print(f"\n  Chênh lệch engine so với bốc ngẫu nhiên: {tl-tl_nn:+,.0f} điểm "
          f"({(tl-tl_nn)/tv:+.2%} trên tổng vốn)")

    # đường vốn & chuỗi thua dài nhất
    luy = 0; dinh = 0; sut_max = 0; thua = 0; thua_max = 0
    for b in bg:
        ngay = sum(b[k]["lai"] for k in keys)
        luy += ngay; dinh = max(dinh, luy); sut_max = max(sut_max, dinh-luy)
        thua = thua+1 if ngay < 0 else 0
        thua_max = max(thua_max, thua)
    print(f"  Sụt vốn sâu nhất : {sut_max:,.0f} điểm")
    print(f"  Chuỗi thua dài nhất: {thua_max} kỳ liên tiếp")
    ev_lt = sum((Ks[k]/(10000 if k=='4SO' else 1000 if k=='3SO' else 100)) *
                (bg[0][k]['von']/(Ks[k]*diem[k])) *
                (TY_LE_TRA_4SO if k=='4SO' else
                 TY_LE_TRA_3SO if k=='3SO' else TY_LE_TRA_2SO)*diem[k]
                - bg[0][k]['von'] for k in keys) * nt
    print(f"  Kỳ vọng lý thuyết: {ev_lt:+,.0f} điểm  ({ev_lt/tv:+.2%})")
    print(f"  → Thực tế {tl:+,.0f} lệch {tl-ev_lt:+,.0f} so với kỳ vọng. "
          f"{'Trong' if abs(tl-ev_lt) < 2*abs(ev_lt) else 'NGOÀI'} biên dao động thường gặp.")
    return {"tong_lai": tl, "tong_lai_nn": tl_nn, "tong_von": tv,
            "sut_max": sut_max, "thua_max": thua_max}


def chay_backtest(chon_dai=1, so_ky=100, Ks=None, diem=None, chu_ky_toi_uu=5):
    """Tải dữ liệu THẬT của 1 đài rồi chạy backtest lãi/lỗ đầy đủ."""
    Ks = Ks or {"DB": SO_CON_DB, "LO2": SO_CON_LO2, "3SO": SO_CON_3SO, "4SO": SO_CON_4SO}
    diem = diem or {"DB": DIEM_DB, "LO2": DIEM_LO2, "3SO": DIEM_3SO, "4SO": DIEM_4SO}
    ten, ma, mien, nd = lay_dai(chon_dai)
    m = lay_tu_master(chon_dai, so_ky)
    if m:
        tg, ngay_full, info = m
        ngay = [d.strftime("%d/%m") for d in ngay_full]
        print(f"  [{chon_dai}] {ten} — lấy từ KHO DỮ LIỆU")
    else:
        print(f"  Đang tải {so_ky} kỳ của [{chon_dai}] {ten} từ web...")
        _, ngay, ngay_full, tg, info = lay_du_lieu(chon_dai, so_ky)
    if not tg:
        raise RuntimeError("không bóc được toàn bảng giải")
    print(f"  {len(tg)} kỳ, từ {fmt_ngay(ngay_full[0], ngay[0])} "
          f"đến {fmt_ngay(ngay_full[-1], ngay[-1])}")
    print(f"  Đang backtest (chọn lại trọng số mỗi {chu_ky_toi_uu} kỳ, chỉ dùng quá khứ)...")
    bg = backtest_von(tg, Ks, diem, khu=info["khu"], chu_ky_toi_uu=chu_ky_toi_uu)
    return bao_cao_backtest(bg, Ks, diem, f"{ten} ({mien})")




# ==============================================================================
#  [20] KHO DỮ LIỆU CHỦ (MASTER DATA)
#
#  Quét 1 lần toàn bộ 36 đài → lưu ra file → các lần sau đọc từ file.
#
#  BA ĐIỂM QUAN TRỌNG:
#  1. Ổ ĐĨA COLAB LÀ TẠM THỜI. Ngắt kết nối là mất sạch. File PHẢI lưu vào
#     Google Drive, nếu không lần sau vẫn phải quét lại từ đầu.
#  2. LẤY 200 KỲ, KHÔNG PHẢI 100. Cùng công sức quét, gấp đôi dữ liệu.
#     Sau này muốn tăng SO_KY lên 200 là có sẵn, không phải quét lại.
#  3. PHẢI CẬP NHẬT HÀNG NGÀY. Dữ liệu cũ 1 ngày là dự báo sai 1 kỳ.
#     cap_nhat_master() chỉ tải phần mới, mất ~30 giây thay vì 3 phút.
# ==============================================================================

MASTER_DRIVE = os.environ.get("MASTER_PATH", "data/master.json")  # trong repo
MASTER_TAM   = "master.json"
_MASTER_MEM  = None


def gan_drive():
    """Trên GitHub Actions: chỉ cần tạo thư mục data/ trong repo.
       File được workflow commit lại nên tồn tại vĩnh viễn giữa các lần chạy."""
    d = os.path.dirname(MASTER_DRIVE)
    if d:
        os.makedirs(d, exist_ok=True)
    return True


def _duong_dan_master():
    d = os.path.dirname(MASTER_DRIVE)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    return MASTER_DRIVE


def tao_master(so_ky=200, chi_dai=None, nghi=0.8):
    """Quét toàn bộ 36 đài, lưu kho dữ liệu chủ. Chạy MỘT LẦN (~3-4 phút)."""
    gan_drive()
    dsach = chi_dai or list(range(1, len(DAI_LIST) + 1))
    kho, loi = {}, []
    print(f"  Quét {len(dsach)} đài × {so_ky} kỳ. Ước tính {len(dsach)*(nghi+2.5)/60:.1f} phút.\n")
    for stt in dsach:
        ten, ma, mien, nd = lay_dai(stt)
        try:
            _, ngay, ngay_full, tg, info = lay_du_lieu(stt, so_ky)
            if not tg:
                raise RuntimeError("không bóc được toàn bảng giải")
            ky = [{"ngay": d.isoformat() if d else None, "giai": g}
                  for d, g in zip(ngay_full, tg)]
            dem = {}
            for d in ngay_full:
                if d: dem[d.weekday()] = dem.get(d.weekday(), 0) + 1
            ng = max(2, int(0.15 * sum(dem.values())))
            cac_thu = sorted(w for w, c in dem.items() if c >= ng) or [max(dem, key=dem.get)]
            kho[str(stt)] = {"ten": ten, "ma": ma, "mien": mien,
                             "khu": "MB" if ma == "xsmb" else "MN",
                             "thu": int(cac_thu[0]), "cac_thu": [int(x) for x in cac_thu],
                             "thu_vn": ", ".join(THU_VN[x] for x in cac_thu), "ky": ky}
            print(f"     ✓ [{stt:>2}] {ten:<24}{len(ky):>4} kỳ  "
                  f"{ky[0]['ngay']} → {ky[-1]['ngay']}")
        except Exception as e:
            loi.append((stt, ten, str(e)))
            print(f"     ✗ [{stt:>2}] {ten:<24}LỖI: {e}")
        time.sleep(nghi)

    # --- Thử lại các đài lỗi, chậm hơn ---
    for vong in range(2):
        if not loi: break
        print(f"\n  Thử lại {len(loi)} đài lỗi (vòng {vong+1}/2)...")
        con = []
        for stt, ten, _ in loi:
            try:
                _, ngay, ngay_full, tg, info = lay_du_lieu(stt, so_ky)
                if not tg: raise RuntimeError("không bóc được toàn bảng giải")
                ky = [{"ngay": d.isoformat() if d else None, "giai": g}
                      for d, g in zip(ngay_full, tg)]
                thu = max({d.weekday() for d in ngay_full if d},
                          key=lambda w: sum(1 for d in ngay_full if d and d.weekday() == w))
                ma = DAI_LIST[stt-1][1]; mien = DAI_LIST[stt-1][2]
                kho[str(stt)] = {"ten": ten, "ma": ma, "mien": mien,
                                 "khu": "MB" if ma == "xsmb" else "MN",
                                 "thu": int(thu), "thu_vn": THU_VN[thu], "ky": ky}
                print(f"     ✓ [{stt:>2}] {ten:<24}{len(ky):>4} kỳ (lần thử lại)")
            except Exception as e:
                con.append((stt, ten, str(e)))
                print(f"     ✗ [{stt:>2}] {ten:<24}{e}")
            time.sleep(2.5)
        loi = con

    kho_full = {"tao_luc": datetime.now(timezone(timedelta(hours=7))).isoformat(),
                "so_ky": so_ky, "dai": kho,
                "loi": [f"{s}. {t}: {e}" for s, t, e in loi]}
    dd = _duong_dan_master()
    json.dump(kho_full, open(dd, "w"), ensure_ascii=False)
    kb = os.path.getsize(dd) / 1024
    print(f"\n  Đã lưu {dd}  ({kb:,.0f} KB, {len(kho)}/{len(dsach)} đài)")

    globals()["_MASTER_MEM"] = kho_full
    return kho_full


def doc_master(bat_buoc_moi=False):
    global _MASTER_MEM
    if _MASTER_MEM is not None and not bat_buoc_moi:
        return _MASTER_MEM
    for dd in [MASTER_DRIVE, MASTER_TAM]:
        if os.path.exists(dd):
            _MASTER_MEM = json.load(open(dd))
            return _MASTER_MEM
    return None


def cap_nhat_master(so_ky_moi=30, nghi=0.8):
    """Chỉ tải các kỳ MỚI rồi ghép vào kho. ~40 giây thay vì 4 phút.
       Chạy mỗi ngày trước khi dự báo."""
    kho = doc_master()
    if kho is None:
        print("  Chưa có kho dữ liệu — chạy tao_master() trước.")
        return None
    them = 0
    print(f"  Cập nhật {len(kho['dai'])} đài (tải {so_ky_moi} kỳ gần nhất mỗi đài)...")
    for stt, d in kho["dai"].items():
        try:
            _, ngay, ngay_full, tg, _ = lay_du_lieu(int(stt), so_ky_moi)
            co = {k["ngay"] for k in d["ky"]}
            moi = [{"ngay": dt.isoformat(), "giai": g}
                   for dt, g in zip(ngay_full, tg) if dt and dt.isoformat() not in co]
            if moi:
                d["ky"] = (d["ky"] + moi)[-kho["so_ky"]:]
                them += len(moi)
                print(f"     + [{stt:>2}] {d['ten']:<22}{len(moi)} kỳ mới "
                      f"→ mới nhất {d['ky'][-1]['ngay']}")
        except Exception as e:
            print(f"     ✗ [{stt:>2}] {d['ten']:<22}{e}")
        time.sleep(nghi)
    kho["tao_luc"] = datetime.now(timezone(timedelta(hours=7))).isoformat()
    json.dump(kho, open(_duong_dan_master(), "w"), ensure_ascii=False)
    print(f"\n  Xong. Thêm {them} kỳ mới.")
    return kho


def lay_tu_master(stt, n_can, ngay_moc=None):
    """Đọc dữ liệu 1 đài từ kho. Trả None nếu kho không có hoặc quá cũ."""
    kho = doc_master()
    if not kho or str(stt) not in kho["dai"]:
        return None
    d = kho["dai"][str(stt)]
    ky = [k for k in d["ky"] if k["ngay"]]
    if ngay_moc:                       # chống rò rỉ: bỏ kỳ TỪ ngày mốc trở đi
        ky = [k for k in ky if _date.fromisoformat(k["ngay"]) < ngay_moc]
    if len(ky) < n_can:
        return None
    ky = ky[-n_can:]
    tg = [k["giai"] for k in ky]
    ngay_full = [_date.fromisoformat(k["ngay"]) for k in ky]
    info = {"ten": d["ten"], "ma": d["ma"], "mien": d["mien"], "khu": d["khu"],
            "lich": d["thu_vn"], "url": f"KHO DỮ LIỆU ({kho['tao_luc'][:16]})",
            "khop": True, "khop3": True, "n_A": len(tg), "n_B": len(tg),
            "n_FA": len(tg), "n_FB": len(tg), "n_digits": len(tg[0][0])}
    return tg, ngay_full, info


def kiem_tra_do_moi(ngay_moc=None):
    """Cảnh báo nếu kho dữ liệu đã cũ — dữ liệu cũ = dự báo sai."""
    kho = doc_master()
    if not kho:
        return
    hom_nay = ngay_moc or datetime.now(timezone(timedelta(hours=7))).date()
    cu = []
    for stt, d in kho["dai"].items():
        ngays = [_date.fromisoformat(k["ngay"]) for k in d["ky"] if k["ngay"]]
        if not ngays: continue
        # kỳ gần nhất ĐÁNG LẼ phải có, theo lịch quay của đài
        cts = d.get("cac_thu", [d["thu"]])
        can_co = max(hom_nay - timedelta(days=((hom_nay.weekday() - t) % 7) or 7)
                     for t in cts)
        if max(ngays) < can_co:
            cu.append(f"[{stt}] {d['ten']}: mới nhất {max(ngays)}, đáng lẽ có {can_co}")
    if cu:
        print(f"\n  ⚠ KHO DỮ LIỆU CŨ — {len(cu)} đài thiếu kỳ gần nhất:")
        for c in cu[:6]: print(f"      {c}")
        if len(cu) > 6: print(f"      ... và {len(cu)-6} đài khác")
        print(f"  → Chạy cap_nhat_master() trước khi dự báo, nếu không bộ số sẽ SAI KỲ.")
    return cu



# ==============================================================================
#  main()                          → hôm nay, tất cả đài (mặc định)
#  main("25.08.2026")              → chạy cho ngày khác
#  main(so_con_3so=0)              → tắt module lô 3 số cho nhanh
#  xay_lich(bat_buoc_lam_moi=True) → dựng lại lịch quay 36 đài
#  in_danh_sach_dai()              → xem bảng 36 đài
# ==============================================================================
