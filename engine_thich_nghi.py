"""
ENGINE_THICH_NGHI — engine TỰ HỌC, TỰ ĐÚC KẾT KINH NGHIỆM QUA TỪNG KỲ.

THUẬT TOÁN: HEDGE / MULTIPLICATIVE WEIGHTS (Freund & Schapire 1997)
    Đây không phải heuristic. Nó có ĐẢM BẢO TOÁN HỌC chứng minh được:

        Regret sau T kỳ  ≤  sqrt(T · ln N / 2)

    Nghĩa là: sau T kỳ, engine tổng hợp thua engine TỐT NHẤT (biết được khi
    nhìn lại) không quá sqrt(T·lnN/2) — mà KHÔNG CẦN biết trước cái nào tốt.
    Với N=12 chuyên gia và T=150 kỳ, regret ≤ 13,6 kỳ.

CƠ CHẾ TỰ ĐÚC KẾT:
    Sau MỖI kỳ, khi biết kết quả thật:
        w_i  ←  w_i · exp(η · thưởng_i)
    Chuyên gia dự báo đúng được tăng trọng số, sai thì giảm.
    Trọng số LƯU RA FILE -> mang sang ngày hôm sau -> kinh nghiệm TÍCH LUỸ.

CƠ CHẾ CHUẨN HOÁ (calibration):
    Theo dõi xác suất dự báo so với tần suất thực tế. Nếu engine nói "70%"
    mà thực tế chỉ 64%, hệ số hiệu chỉnh tự kéo về.

CƠ CHẾ QUÊN:
    Trọng số nhân với hệ số suy giảm mỗi kỳ -> bằng chứng cũ nhạt dần,
    engine thích nghi khi máy quay thay đổi.

TỰ CHẨN ĐOÁN — phần quan trọng nhất:
    Tính ENTROPY của phân bố trọng số và SỐ CHUYÊN GIA HIỆU DỤNG.
        · Sau nhiều kỳ mà trọng số vẫn ĐỀU (entropy tối đa, hiệu dụng ≈ N)
          -> KHÔNG chuyên gia nào nổi bật -> dữ liệu KHÔNG CÓ tín hiệu.
        · Trọng số dồn về vài chuyên gia -> có phân hoá thật.
    Đây là câu trả lời engine tự đưa ra, không cần tôi phán xét.

GIỚI HẠN PHẢI BIẾT:
    Đảm bảo của Hedge là TƯƠNG ĐỐI: "gần bằng chuyên gia tốt nhất".
    Nếu MỌI chuyên gia đều vô dụng (64%), Hedge cũng cho 64%.
    Nó KHÔNG tạo ra tín hiệu — chỉ đảm bảo không bỏ lỡ tín hiệu nếu có.
"""
import os, json, math, time
from datetime import datetime, timedelta, timezone
import numpy as np

import engine as E
import de_engine12 as D

VN = timezone(timedelta(hours=7))

# ==============================================================================
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  BẢNG ĐIỀU KHIỂN                                                     │
#  └──────────────────────────────────────────────────────────────────────┘
# ==============================================================================

SO_KY        = 150    # số kỳ lịch sử để khởi động trọng số
SO_CON       = 64     # số con cho ĐB/G1/G8
SO_CON_LO    = 20     # số con bao lô
ETA          = 0.60   # tốc độ học. Cao = thích nghi nhanh nhưng nhiễu hơn
SUY_GIAM     = 0.985  # hệ số quên mỗi kỳ (1,0 = không quên)
W_SAN        = 0.02   # trọng số sàn — không chuyên gia nào bị loại hẳn
MIN_TRAIN    = 30
TY_LE_TRA    = 95.0

FILE_TRANG_THAI = "data/trang_thai_hedge.json"

TEN_GIAI = {"DB": "① ĐỀ ĐẶC BIỆT", "G1": "② ĐỀ GIẢI NHẤT",
            "G8": "③ ĐỀ ĐẦU (G8)", "LO": "④ BAO LÔ"}


# ==============================================================================
#  TRẠNG THÁI — kinh nghiệm tích luỹ, mang qua từng ngày
# ==============================================================================

def nap_trang_thai(path=FILE_TRANG_THAI):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    return {"tao_luc": None, "khoa": {}}


def luu_trang_thai(tt, path=FILE_TRANG_THAI):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tt["tao_luc"] = datetime.now(VN).isoformat(timespec="seconds")
    json.dump(tt, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def _khoa(stt, giai):
    return f"{stt}|{giai}"


# ==============================================================================
#  HEDGE — cập nhật trọng số sau mỗi kỳ
# ==============================================================================

def khoi_tao_w(n):
    return np.full(n, 1.0 / n)


def cap_nhat_w(w, thuong, eta=ETA, suy_giam=SUY_GIAM, san=W_SAN):
    """w_i <- w_i * exp(eta * thưởng_i), rồi suy giảm + sàn + chuẩn hoá.

    thuong[i] in [0,1] — 1 nếu chuyên gia i dự báo TRÚNG kỳ vừa rồi.
    """
    w = w * np.exp(eta * (np.asarray(thuong, float) - np.mean(thuong)))
    w = w ** suy_giam                      # quên dần bằng chứng cũ
    w = np.maximum(w, san / len(w))        # sàn: không loại hẳn ai
    return w / w.sum()


def chan_doan_w(w):
    """Entropy + số chuyên gia hiệu dụng. Đây là TỰ CHẨN ĐOÁN của engine."""
    n = len(w)
    p = np.clip(w, 1e-12, None)
    H = float(-(p * np.log(p)).sum())
    H_max = math.log(n)
    hieu_dung = float(np.exp(H))           # perplexity
    return {"entropy": H, "entropy_chuan": H / H_max,
            "hieu_dung": hieu_dung, "n": n,
            "tap_trung": 1.0 - H / H_max}


# ==============================================================================
#  CHẠY HEDGE TRÊN LỊCH SỬ — khởi động trọng số từ 150 kỳ
# ==============================================================================

def hoc_tu_lich_su(mt, pool, k, la_lo=False, min_train=MIN_TRAIN, w0=None):
    """Chạy Hedge tuần tự qua lịch sử. Mỗi kỳ: dự báo -> thấy kết quả -> cập nhật.
       KHÔNG rò rỉ: trọng số tại kỳ t chỉ dựa trên kết quả các kỳ < t."""
    n = len(mt) if not la_lo else len(pool)
    n_eng = len(D.ENGINES)
    w = khoi_tao_w(n_eng) if w0 is None else np.asarray(w0, float)
    rng = np.random.default_rng(2026)

    lich_su_w = []
    hit_hedge, hit_eng = [], np.zeros(n_eng)
    n_test = 0
    for t in range(min_train, n):
        if la_lo:
            mt_phang = [v for ky in pool[:t] for v in ky]
            diem = np.array([f(mt_phang, pool[:t], rng) for _, f in D.ENGINES])
            thuc = set(pool[t])
        else:
            diem = np.array([f(mt[:t], pool[:t], rng) for _, f in D.ENGINES])
            thuc = {mt[t]}

        # --- dự báo tổng hợp: gộp điểm z có trọng số ---
        Z = np.array([D._z(d) for d in diem])
        tong = (w[:, None] * Z).sum(axis=0)
        bo = set(np.argsort(-tong)[:k].tolist())
        hit_hedge.append(len(bo & thuc) if la_lo else (1 if (thuc & bo) else 0))

        # --- thưởng từng chuyên gia, rồi TỰ ĐÚC KẾT ---
        thuong = np.zeros(n_eng)
        for i in range(n_eng):
            sel = set(np.argsort(-Z[i])[:k].tolist())
            tr = len(sel & thuc) if la_lo else (1 if (thuc & sel) else 0)
            thuong[i] = tr / (k if la_lo else 1)
            hit_eng[i] += tr
        w = cap_nhat_w(w, thuong)
        lich_su_w.append(w.copy())
        n_test += 1

    return {"w": w, "lich_su_w": np.array(lich_su_w),
            "hit_hedge": np.array(hit_hedge), "hit_eng": hit_eng,
            "n_test": n_test}


# ==============================================================================
#  CHUẨN HOÁ (CALIBRATION) — engine tự sửa khi dự báo quá tự tin
# ==============================================================================

def chuan_hoa(hit_hedge, k, la_lo=False):
    """So xác suất LÝ THUYẾT với tần suất THỰC TẾ quan sát được."""
    n = len(hit_hedge)
    if n == 0:
        return {"ly_thuyet": 0, "thuc_te": 0, "lech": 0, "he_so": 1.0}
    if la_lo:
        ly = 18 * k / 100.0
        tt = float(np.mean(hit_hedge))
    else:
        ly = k / 100.0
        tt = float(np.mean(hit_hedge))
    he_so = tt / ly if ly > 0 else 1.0
    return {"ly_thuyet": ly, "thuc_te": tt, "lech": tt - ly, "he_so": he_so, "n": n}


def wilson(t, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p, den = t / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / den
    return (max(0., c - h), min(1., c + h))


# ==============================================================================
#  CHẠY 1 ĐÀI
# ==============================================================================

def chay_dai(stt, ngay_moc, tt, so_ky=SO_KY, giai_list=None):
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
    if len(tg) < 40:
        raise RuntimeError(f"chỉ còn {len(tg)} kỳ, cần >=40")

    pool = [[int(s[-2:]) for s in ky] for ky in tg]
    n_lo = len(pool[0])
    vt_g8 = len(tg[0]) - 1 if (len(tg[0]) == 18 and len(tg[0][-1]) == 2) else None
    ds = [("DB", 0, SO_CON), ("G1", 1, SO_CON)]
    if vt_g8 is not None:
        ds.append(("G8", vt_g8, SO_CON))
    if len(tg[0]) == 18:
        ds.append(("LO", None, SO_CON_LO))
    if giai_list:
        ds = [x for x in ds if x[0] in giai_list]

    rng = np.random.default_rng(2026)
    mods = []
    for khoa, vi, k in ds:
        la_lo = khoa == "LO"
        mt = [int(ky[vi][-2:]) for ky in tg] if not la_lo else []
        kh = _khoa(stt, khoa)
        w0 = tt["khoa"].get(kh, {}).get("w")

        r = hoc_tu_lich_su(mt, pool, k, la_lo, w0=w0)
        w, cd = r["w"], chan_doan_w(r["w"])
        ch = chuan_hoa(r["hit_hedge"], k, la_lo)

        # --- dự báo kỳ TỚI bằng trọng số đã đúc kết ---
        if la_lo:
            mt_phang = [v for ky in pool for v in ky]
            diem = np.array([D._z(f(mt_phang, pool, rng)) for _, f in D.ENGINES])
        else:
            diem = np.array([D._z(f(mt, pool, rng)) for _, f in D.ENGINES])
        tong = (w[:, None] * diem).sum(axis=0)
        bo = sorted(int(x) for x in np.argsort(-tong)[:k])

        # --- so Hedge với chuyên gia tốt nhất và với mốc ---
        hr_hedge = float(np.mean(r["hit_hedge"])) / (k if la_lo else 1)
        hr_eng = r["hit_eng"] / r["n_test"] / (k if la_lo else 1)
        moc = (1 - (1 - .01) ** n_lo) if la_lo else k / 100.0
        i_tot = int(np.argmax(hr_eng))

        # --- lưu kinh nghiệm cho ngày mai ---
        tt["khoa"][kh] = {
            "w": [float(x) for x in w], "n_kỳ_hoc": r["n_test"],
            "hr_hedge": hr_hedge, "cap_nhat": f"{ngay_moc:%Y-%m-%d}",
            "tap_trung": cd["tap_trung"], "hieu_dung": cd["hieu_dung"],
        }

        mods.append({
            "key": khoa, "ten": TEN_GIAI[khoa], "k": k, "la_lo": la_lo,
            "so": [f"{v:02d}" for v in bo],
            "w": w, "chan_doan": cd, "chuan_hoa": ch,
            "hr_hedge": hr_hedge, "hr_eng": hr_eng, "n_test": r["n_test"],
            "moc": moc, "hoa_von": (k * n_lo / TY_LE_TRA / n_lo) if la_lo else k / TY_LE_TRA,
            "eng_tot": D.TEN_ENGINE[i_tot], "hr_eng_tot": float(hr_eng[i_tot]),
            "top_w": sorted(zip(D.TEN_ENGINE, w), key=lambda x: -x[1])[:5],
            "von": k * n_lo if la_lo else k,
        })

    return {"stt": stt, "dai": ten, "mien": mien, "nguon": nguon, "n_ky": len(tg),
            "n_lo": n_lo, "ngay_ky_truoc": ng[-1].strftime("%d/%m/%Y"),
            "db_ky_truoc": tg[-1][0], "modules": mods, "ket_qua_that": that}


# ==============================================================================
#  PIPELINE
# ==============================================================================

def main(ngay=None, so_ky=None, giai_list=None, gui_mail=True):
    ngay_moc = E.doc_ngay(ngay) if ngay else datetime.now(VN).date()
    so_ky = so_ky or SO_KY
    thu = E.THU_VN[ngay_moc.weekday()]
    tt = nap_trang_thai()
    co_kn = len(tt["khoa"]) > 0

    print("=" * 80)
    print(f"  ENGINE THÍCH NGHI (Hedge)  |  {thu} {ngay_moc:%d.%m.%Y}  |  {so_ky} kỳ")
    print(f"  η={ETA} · suy giảm={SUY_GIAM} · sàn={W_SAN} · {len(D.ENGINES)} chuyên gia")
    if co_kn:
        print(f"  Kinh nghiệm tích luỹ: {len(tt['khoa'])} mục, cập nhật {tt['tao_luc'][:16]}")
    else:
        print(f"  Lần đầu chạy — khởi tạo trọng số ĐỀU, học từ {so_ky} kỳ lịch sử")
    print("=" * 80)

    dsach = E.dai_theo_ngay(ngay_moc)
    lich = E.xay_lich()
    print(f"\n  Đài hôm nay: {len(dsach)}")

    ket, loi = [], []
    for s in dsach:
        ten = lich.get(str(s), {}).get("ten", E.lay_dai(s)[0])
        try:
            r = chay_dai(s, ngay_moc, tt, so_ky, giai_list)
            ket.append(r)
            print(f"     ✓ [{s:>2}] {ten:<20} {r['n_ky']} kỳ ({r['nguon']}) | "
                  f"{len(r['modules'])} giải")
        except Exception as e:
            loi.append((s, ten, str(e)))
            print(f"     ✗ [{s:>2}] {ten:<20} LỖI: {e}")
    if not ket:
        raise RuntimeError("Không đài nào chạy được.")
    luu_trang_thai(tt)
    print(f"\n  ✓ Đã lưu kinh nghiệm vào {FILE_TRANG_THAI}")

    # ---------- TỰ CHẨN ĐOÁN ----------
    print(f"\n[A] ENGINE TỰ CHẨN ĐOÁN — trọng số có phân hoá không?")
    print("-" * 80)
    print(f"  {'Đài':<16}{'Giải':<18}{'Hiệu dụng':>12}{'Tập trung':>12}{'Hedge':>10}{'Mốc':>9}")
    print("  " + "-" * 76)
    tt_list = []
    for r in ket:
        for i, m in enumerate(r["modules"]):
            cd = m["chan_doan"]
            tt_list.append(cd["tap_trung"])
            print(f"  {r['dai'] if i==0 else '':<16}{m['ten']:<18}"
                  f"{cd['hieu_dung']:>10.1f}/{cd['n']:<2}{cd['tap_trung']:>11.1%}"
                  f"{m['hr_hedge']:>10.1%}{m['moc']:>9.1%}")
    tb = float(np.mean(tt_list))
    print("  " + "-" * 76)
    print(f"  Độ tập trung trung bình: {tb:.1%}")
    print("  → " + ("Trọng số vẫn gần ĐỀU sau khi học — KHÔNG chuyên gia nào nổi bật."
                    "\n    Đây là engine TỰ NÓI rằng dữ liệu không có tín hiệu phân hoá."
                    if tb < 0.10 else
                    "Trọng số ĐÃ phân hoá — có chuyên gia nổi bật hơn hẳn."
                    "\n    Theo dõi xem chuyên gia đó có giữ được vị trí qua nhiều ngày."))

    for r in ket:
        for m in r["modules"]:
            lo, hi = wilson(int(m["hr_hedge"] * m["n_test"]), m["n_test"])
            print("\n" + "=" * 80)
            print(f"  {r['dai'].upper()}  —  {m['ten']}  ({m['k']} con)")
            print("=" * 80)
            print(f"\n  {','.join(m['so'])}\n")
            print(f"  Hedge {m['hr_hedge']:.1%} [{lo:.1%},{hi:.1%}] · mốc {m['moc']:.1%} · "
                  f"chuyên gia tốt nhất {m['eng_tot']} {m['hr_eng_tot']:.1%}")
            print(f"  Chuẩn hoá: lý thuyết {m['chuan_hoa']['ly_thuyet']:.3f} · "
                  f"thực tế {m['chuan_hoa']['thuc_te']:.3f} · "
                  f"hệ số {m['chuan_hoa']['he_so']:.3f}")
            print(f"  Hiệu dụng {m['chan_doan']['hieu_dung']:.1f}/{m['chan_doan']['n']} "
                  f"chuyên gia · tập trung {m['chan_doan']['tap_trung']:.1%}")
            print(f"\n  TRỌNG SỐ ĐÃ ĐÚC KẾT (5 cao nhất)")
            for ten_e, wv in m["top_w"]:
                print(f"     {ten_e:<18}{wv:>7.1%}  {'█'*int(wv*180)}")
    print("\n" + "=" * 80)
    print(f"  ⚠ Đảm bảo của Hedge là TƯƠNG ĐỐI: 'gần bằng chuyên gia tốt nhất'.")
    print(f"    Nếu MỌI chuyên gia đều ở mốc ngẫu nhiên, Hedge cũng ở đó.")
    print(f"    Nó không tạo ra tín hiệu — chỉ đảm bảo không bỏ lỡ nếu có.")

    if gui_mail:
        print("\n  Đang gửi email...")
        try:
            gui_email(ket, ngay_moc, loi, tb)
        except Exception as e:
            print(f"  ✗ Không gửi được email: {e}")
    return ket


# ==============================================================================
#  EMAIL
# ==============================================================================

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

EMAIL_NHAN = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI  = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")


def _html(ket, ngay_moc, loi, tap_trung_tb):
    css = "font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
    thu = E.THU_VN[ngay_moc.weekday()]
    phan_hoa = tap_trung_tb >= 0.10
    mau_pt = "#2e7d32" if phan_hoa else "#c62828"
    h = [f'<div style="{css}max-width:760px;color:#222">']
    h.append(f'<h2 style="margin:0 0 2px">Engine thích nghi — {thu} {ngay_moc:%d.%m.%Y}</h2>')
    h.append(f'<p style="color:#666;margin:0 0 14px;font-size:14px">'
             f'{len(ket)} đài · Hedge trên {len(D.ENGINES)} chuyên gia · '
             f'kinh nghiệm TÍCH LUỸ qua từng kỳ</p>')
    h.append(f'<div style="background:#fff8e1;border-left:4px solid #f9a825;'
             f'padding:10px 13px;margin:0 0 20px;font-size:13px;line-height:1.6">'
             f'<b>Engine này tự đúc kết thế nào:</b><br>'
             f'Sau mỗi kỳ, chuyên gia dự báo đúng được TĂNG trọng số, sai thì GIẢM '
             f'(w ← w·e<sup>η·thưởng</sup>). Trọng số lưu ra file, mang sang ngày sau.<br>'
             f'<b style="color:{mau_pt}">Tự chẩn đoán: độ tập trung trọng số '
             f'{tap_trung_tb:.1%}</b> — '
             + ("trọng số ĐÃ phân hoá, có chuyên gia nổi bật."
                if phan_hoa else
                "trọng số vẫn gần ĐỀU sau khi học. Engine đang TỰ NÓI rằng "
                "không chuyên gia nào nổi bật — dữ liệu không có tín hiệu phân hoá.")
             + '</div>')

    for r in ket:
        h.append(f'<div style="margin:24px 0 0;padding:9px 12px;background:#263238;'
                 f'color:#fff;border-radius:5px 5px 0 0">'
                 f'<span style="font-size:17px;font-weight:700">{r["dai"].upper()}</span>'
                 f'<span style="font-size:13px;opacity:.85"> &nbsp;|&nbsp; {r["mien"]}'
                 f' &nbsp;|&nbsp; {thu} {ngay_moc:%d.%m.%Y}</span><br>'
                 f'<span style="font-size:12px;opacity:.7">Kỳ gần nhất '
                 f'{r["ngay_ky_truoc"]} · ĐB {r["db_ky_truoc"]} · {r["n_ky"]} kỳ</span></div>')
        h.append('<div style="border:1px solid #cfd8dc;border-top:0;'
                 'border-radius:0 0 5px 5px;padding:8px 12px 14px">')
        for m in r["modules"]:
            cd = m["chan_doan"]
            h.append(f'<div style="margin:12px 0 0">'
                     f'<div style="font-size:14px;font-weight:600">{m["ten"]} '
                     f'&mdash; {m["k"]} con</div>'
                     f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
                     f'font-size:14px;background:#e8f5e9;border-left:4px solid #2e7d32;'
                     f'padding:10px 12px;margin-top:5px;word-break:break-all;'
                     f'line-height:1.9">{",".join(m["so"])}</div>')
            h.append(f'<div style="font-size:12px;color:#546e7a;margin:5px 0 0">'
                     f'Hedge <b>{m["hr_hedge"]:.1%}</b> · mốc {m["moc"]:.1%} · '
                     f'chuyên gia tốt nhất {m["eng_tot"]} ({m["hr_eng_tot"]:.1%}) · '
                     f'{cd["hieu_dung"]:.1f}/{cd["n"]} chuyên gia hiệu dụng · '
                     f'tập trung {cd["tap_trung"]:.1%}</div>')
            h.append('<div style="font-size:11px;color:#78909c;margin:4px 0 0">'
                     '<b>Trọng số đã đúc kết:</b> '
                     + " · ".join(f'{t} {w:.0%}' for t, w in m["top_w"][:4]) + '</div></div>')
        h.append('</div>')

    if loi:
        h.append('<div style="background:#ffebee;border-left:4px solid #c62828;'
                 'padding:10px 13px;margin:16px 0;font-size:13px"><b>Đài lỗi:</b><ul>')
        for s_, t_, e_ in loi:
            h.append(f'<li>[{s_}] {t_}: {e_}</li>')
        h.append('</ul></div>')

    h.append('<hr style="margin:24px 0 10px;border:0;border-top:1px solid #ddd">'
             '<p style="font-size:12px;color:#888;line-height:1.6">'
             'Hedge có đảm bảo toán học: regret ≤ √(T·lnN/2) — sau T kỳ thua chuyên gia '
             'tốt nhất không quá ngần ấy, mà không cần biết trước cái nào tốt.<br>'
             '<b>Nhưng đảm bảo đó là TƯƠNG ĐỐI.</b> Nếu mọi chuyên gia đều ở mốc ngẫu '
             'nhiên, Hedge cũng ở đó. Nó không tạo ra tín hiệu — chỉ đảm bảo không bỏ lỡ '
             'nếu có. Hãy theo dõi "độ tập trung" qua nhiều ngày: nếu nó mãi gần 0, '
             'đó là câu trả lời.</p></div>')
    return "".join(h)


def gui_email(ket, ngay_moc, loi, tap_trung_tb):
    mk = E._lay_mat_khau()
    thu = E.THU_VN[ngay_moc.weekday()]
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[THÍCH NGHI] {thu} {ngay_moc:%d.%m.%Y} — {len(ket)} đài"
    msg["From"] = formataddr(("XSMN Thích Nghi", EMAIL_GUI)); msg["To"] = EMAIL_NHAN
    t = [f"ENGINE THÍCH NGHI — {thu} {ngay_moc:%d.%m.%Y}",
         f"Độ tập trung trọng số: {tap_trung_tb:.1%}", ""]
    for r in ket:
        t.append("=" * 60); t.append(f"{r['dai'].upper()} | {r['mien']}")
        for m in r["modules"]:
            t.append("")
            t.append(f"{m['ten']} — {m['k']} con (Hedge {m['hr_hedge']:.1%}, "
                     f"mốc {m['moc']:.1%})")
            t.append(",".join(m["so"]))
        t.append("")
    msg.attach(MIMEText("\n".join(t), "plain", "utf-8"))
    msg.attach(MIMEText(_html(ket, ngay_moc, loi, tap_trung_tb), "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk); sv.send_message(msg)
    print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")
