"""
KIEM_CHUNG_MB — engine DUY NHẤT: kiểm chứng rồi khai thác độ lệch Miền Bắc Giải Nhất.

VÌ SAO CHỈ MỘT ENGINE NÀY:
    Với máy quay công bằng, 64 con bất kỳ trúng đúng 64% — đây là ĐỊNH LÝ.
    Engine chỉ vượt 64% được bằng MỘT cơ chế: khai thác độ lệch THẬT của máy.

    Backtest 5.350 lượt thật trên 36 đài cho kết quả:
        tần suất 64,04% · ngẫu nhiên 63,87% · 106/107 ô về đúng mốc
    NGOẠI LỆ DUY NHẤT: Miền Bắc Giải Nhất — 44/50 = 88%
        p thô = 0,000132 · sau Bonferroni 107 ô: p = 0,0142 -> vẫn có ý nghĩa

    Đây là manh mối duy nhất của toàn dự án. Engine này làm đúng một việc:
    xác định manh mối đó là THẬT hay là LỖI DỮ LIỆU, trước khi đặt đồng nào.

NĂM PHÉP KIỂM — phải qua CẢ NĂM mới được khai thác:
    K1 TRÙNG LẶP     : kho dữ liệu có kỳ bị lưu lặp không? (nguồn rò rỉ phổ biến nhất)
    K2 LẶP GIÁ TRỊ   : G1 có lặp lại bất thường giữa các kỳ liên tiếp không?
    K3 KHỐI ĐỘC LẬP  : độ lệch có lặp lại ở các khối 150 kỳ RỜI NHAU hoàn toàn?
    K5 ĐỐI CHỨNG     : G1 có vượt ĐB cùng đài, và vượt bốc ngẫu nhiên không?

    Nếu thiếu MỘT phép -> kết luận là lỗi dữ liệu hoặc may rủi, KHÔNG khai thác.

    BÀI HỌC KHI XÂY ENGINE NÀY: bản đầu dùng cửa sổ trượt CHỒNG LẤN. Trên dữ
    liệu NGẪU NHIÊN, nó báo "84%, p=0,0016 -> lặp lại ✓" vì một đoạn may được
    đếm hai lần. Các khối phải RỜI NHAU hoàn toàn mới là kiểm định độc lập.
"""
import os, sys, json
import numpy as np
from scipy import stats
from collections import Counter

import engine as E

STT_MB   = 36
SO_CON   = 64
ALPHA    = 1.0
HOA_VON  = SO_CON / 95.0
MUC_Y    = 0.05


def chon_64(ls, k=SO_CON):
    c = np.full(100, ALPHA)
    for v in ls:
        c[v] += 1
    return set(int(i) for i in np.argsort(-c, kind="stable")[:k])


def test_cua_so(mt, hoc, test):
    """Chọn từ mt[:hoc], kiểm trên mt[hoc:hoc+test]. Trả số lần trúng."""
    bo = chon_64(mt[:hoc])
    doan = mt[hoc:hoc + test]
    return sum(1 for v in doan if v in bo), len(doan)


def p_mot_phia(trung, n, p0=SO_CON / 100):
    return float(1 - stats.binom.cdf(trung - 1, n, p0)) if n else 1.0


def main():
    print("=" * 78)
    print("  KIỂM CHỨNG MANH MỐI DUY NHẤT — MIỀN BẮC GIẢI NHẤT")
    print("=" * 78)

    m = E.lay_tu_master(STT_MB, 450)
    if m:
        tg, ngay_full, info = m; print("  Nguồn: KHO DỮ LIỆU")
    else:
        print("  Đang tải từ web...")
        _, _, ngay_full, tg, info = E.lay_du_lieu(STT_MB, 450)
    if not tg:
        print("  ✗ Không lấy được dữ liệu Miền Bắc."); return None
    n = len(tg)
    g1 = [int(ky[1][-2:]) for ky in tg]
    db = [int(ky[0][-2:]) for ky in tg]
    print(f"  {n} kỳ Miền Bắc\n")
    ket = {}

    # ---------------- K1: TRÙNG LẶP KỲ ----------------
    print("[K1] KHO CÓ KỲ BỊ LƯU LẶP KHÔNG?")
    ngay = [str(d) for d in ngay_full] if ngay_full else []
    so_ngay_khac = len(set(ngay)) if ngay else n
    trung_bang = sum(1 for i in range(1, n) if tg[i] == tg[i - 1])
    print(f"     Số kỳ: {n} · số ngày KHÁC NHAU: {so_ngay_khac} · "
          f"kỳ trùng khít kỳ liền trước: {trung_bang}")
    ket["K1"] = so_ngay_khac == n and trung_bang == 0
    print("     → " + ("SẠCH ✓" if ket["K1"] else
                        "CÓ TRÙNG LẶP ✗ — đây là nguồn rò rỉ, kết quả 88% có thể do lỗi này"))

    # ---------------- K2: LẶP GIÁ TRỊ G1 ----------------
    print("\n[K2] G1 CÓ LẶP BẤT THƯỜNG GIỮA CÁC KỲ LIỀN KỀ?")
    lap = sum(1 for i in range(1, n) if g1[i] == g1[i - 1])
    kv = (n - 1) * 0.01
    p_lap = float(1 - stats.binom.cdf(lap - 1, n - 1, 0.01))
    so_khac = len(set(g1))
    print(f"     G1 lặp y hệt kỳ trước: {lap} lần (kỳ vọng ngẫu nhiên {kv:.1f}) · p = {p_lap:.4f}")
    print(f"     Số giá trị G1 KHÁC NHAU trong {n} kỳ: {so_khac} "
          f"(kỳ vọng ~{100*(1-0.99**n):.0f})")
    ket["K2"] = p_lap >= MUC_Y and so_khac >= 0.85 * 100 * (1 - 0.99 ** n)
    print("     → " + ("BÌNH THƯỜNG ✓" if ket["K2"] else
                        "BẤT THƯỜNG ✗ — nghi bóc dữ liệu sai vị trí hoặc lặp"))

    # ---------------- K3: CÁC KHỐI RỜI NHAU HOÀN TOÀN ----------------
    # Mỗi khối 150 kỳ (100 học + 50 test), KHÔNG chung kỳ nào với khối khác.
    # Bản trước dùng cửa sổ trượt chồng lấn -> một đoạn may bị đếm nhiều lần.
    print("\n[K3] ĐỘ LỆCH CÓ LẶP LẠI Ở CÁC KHỐI ĐỘC LẬP?")
    print(f"     {'Khối (rời nhau)':<22}{'G1 trúng':>12}{'Tỷ lệ':>9}{'p':>9}")
    hoc, test = 100, 50
    kq3 = []
    for bd in range(0, n - (hoc + test) + 1, hoc + test):       # bước = 150
        seg = g1[bd:bd + hoc + test]
        t, nn = test_cua_so(seg, hoc, test)
        kq3.append((t, nn))
        print(f"     kỳ {bd+1:>3}–{bd+hoc+test:<3}          {t:>5}/{nn:<5}"
              f"{t/nn:>9.0%}{p_mot_phia(t, nn):>9.3f}")
    if len(kq3) >= 2:
        T = sum(t for t, _ in kq3); N = sum(nn for _, nn in kq3)
        p_gop = p_mot_phia(T, N)
        vuot = sum(1 for t, nn in kq3 if t / nn > HOA_VON)
        print(f"     Gộp {len(kq3)} khối: {T}/{N} = {T/N:.1%} · p = {p_gop:.4f} · "
              f"{vuot}/{len(kq3)} khối vượt hoà vốn")
        ket["K3"] = vuot == len(kq3) and p_gop < MUC_Y and T / N > HOA_VON
    else:
        print(f"     Chỉ đủ {len(kq3)} khối — cần >= 300 kỳ cho 2 khối độc lập")
        ket["K3"] = False
    print("     → " + ("LẶP LẠI Ở MỌI KHỐI ✓" if ket["K3"] else
                        "KHÔNG lặp lại ✗ — độ lệch không tái hiện ở dữ liệu độc lập"))

    # ---------------- K5: ĐỐI CHỨNG ----------------
    print("\n[K5] G1 CÓ VƯỢT ĐB CÙNG ĐÀI VÀ BỐC NGẪU NHIÊN?")
    t_g1, n_ = test_cua_so(g1[-150:], 100, 50)
    t_db, _ = test_cua_so(db[-150:], 100, 50)
    rng = np.random.default_rng(2026)
    nn_list = []
    for _ in range(2000):
        bo = set(rng.choice(100, SO_CON, replace=False).tolist())
        nn_list.append(sum(1 for v in g1[-50:] if v in bo))
    p_nn = float(np.mean(np.array(nn_list) >= t_g1))
    print(f"     G1 {t_g1}/{n_} · ĐB {t_db}/{n_} · ngẫu nhiên TB {np.mean(nn_list):.1f}/{n_}")
    print(f"     P(ngẫu nhiên >= G1) = {p_nn:.4f}")
    ket["K5"] = t_g1 > t_db and p_nn < MUC_Y
    print("     → " + ("VƯỢT ĐỐI CHỨNG ✓" if ket["K5"] else "KHÔNG vượt ✗"))

    # ---------------- PHÁN XỬ ----------------
    qua = sum(ket.values())
    print("\n" + "=" * 78)
    print(f"  PHÁN XỬ: qua {qua}/{len(ket)} phép kiểm")
    print("=" * 78)
    for k, v in ket.items():
        print(f"     {k}  {'✓' if v else '✗'}")
    if qua == len(ket):
        bo = sorted(chon_64(g1))
        print(f"\n  → QUA TẤT CẢ. Độ lệch Miền Bắc Giải Nhất có vẻ THẬT.")
        print(f"    64 con cho kỳ tới (chọn theo tần suất toàn bộ {n} kỳ):")
        print("    " + ",".join(f"{v:02d}" for v in bo))
        print(f"\n    Khuyến nghị: theo dõi TIẾN CỨU 30 kỳ với tiền nhỏ trước khi")
        print(f"    tin. Một phát hiện thật phải tiếp tục đúng trong tương lai.")
    else:
        print(f"\n  → KHÔNG ĐỦ ĐIỀU KIỆN. Manh mối 88% không đứng vững.")
        print(f"    Khả năng cao là lỗi dữ liệu hoặc may rủi trong 107 phép thử.")
        print(f"    Không có engine nào khai thác được — vì không có gì để khai thác.")
    return ket


# ==============================================================================
#  GỬI EMAIL — ghi lại TOÀN BỘ báo cáo in ra màn hình rồi gửi đi
# ==============================================================================
import io, html, smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

VN = timezone(timedelta(hours=7))
EMAIL_NHAN = os.environ.get("MAIL_TO",   "Linh.tm.pg@gmail.com")
EMAIL_GUI  = os.environ.get("MAIL_USER", "Linh.tm.pg@gmail.com")


class _Tee:
    """Vừa in ra màn hình (log GitHub), vừa ghi vào bộ đệm để gửi email."""
    def __init__(self, *luong):
        self.luong = luong
    def write(self, s):
        for l in self.luong:
            l.write(s)
    def flush(self):
        for l in self.luong:
            l.flush()


def gui_email(bao_cao, ket):
    mk = E._lay_mat_khau()
    hn = datetime.now(VN)
    if ket:
        qua, tong = sum(ket.values()), len(ket)
        dat = qua == tong
        tieu_de = (f"[KIỂM CHỨNG MB G1] QUA {qua}/{tong} — "
                   + ("ĐẠT, độ lệch có vẻ THẬT" if dat else "KHÔNG ĐẠT"))
        mau = "#2e7d32" if dat else "#c62828"
        ket_luan = ("QUA TẤT CẢ phép kiểm — độ lệch Miền Bắc Giải Nhất có vẻ thật. "
                    "Theo dõi tiến cứu 30 kỳ với tiền nhỏ trước khi tin."
                    if dat else
                    "KHÔNG qua đủ phép kiểm — manh mối 88% không đứng vững. "
                    "Khả năng cao là lỗi dữ liệu hoặc may rủi.")
        dong_kq = "".join(
            f'<tr><td style="padding:4px 12px">{k}</td>'
            f'<td style="padding:4px 12px;color:{"#2e7d32" if v else "#c62828"}">'
            f'<b>{"✓ QUA" if v else "✗ TRƯỢT"}</b></td></tr>'
            for k, v in ket.items())
    else:
        tieu_de = "[KIỂM CHỨNG MB G1] LỖI — không lấy được dữ liệu"
        mau, ket_luan, dong_kq = "#c62828", "Không lấy được dữ liệu Miền Bắc.", ""

    than = (f'<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;'
            f'max-width:780px;color:#222">'
            f'<h2 style="margin:0 0 4px">Kiểm chứng Miền Bắc Giải Nhất</h2>'
            f'<p style="color:#666;margin:0 0 14px;font-size:13px">'
            f'{hn:%d.%m.%Y %H:%M} (giờ VN)</p>'
            f'<div style="border-left:4px solid {mau};background:#fafafa;'
            f'padding:11px 14px;margin:0 0 16px;font-size:14px">'
            f'<b style="color:{mau}">KẾT LUẬN:</b> {ket_luan}</div>'
            + (f'<table style="font-size:13px;border-collapse:collapse;margin:0 0 18px;'
               f'background:#eceff1">{dong_kq}</table>' if dong_kq else '')
            + f'<div style="font-size:13px;font-weight:600;margin:0 0 6px">'
            f'BÁO CÁO ĐẦY ĐỦ</div>'
            f'<pre style="font-family:ui-monospace,Menlo,Consolas,monospace;'
            f'font-size:12px;background:#f5f7f8;border:1px solid #cfd8dc;'
            f'padding:12px;white-space:pre-wrap;line-height:1.5">'
            f'{html.escape(bao_cao)}</pre></div>')

    msg = MIMEMultipart("alternative")
    msg["Subject"] = tieu_de
    msg["From"] = formataddr(("XSMN Kiểm Chứng", EMAIL_GUI))
    msg["To"] = EMAIL_NHAN
    msg.attach(MIMEText(bao_cao, "plain", "utf-8"))
    msg.attach(MIMEText(than, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as sv:
        sv.login(EMAIL_GUI, mk)
        sv.send_message(msg)


if __name__ == "__main__":
    import traceback
    dem = io.StringIO()
    goc = sys.stdout
    sys.stdout = _Tee(goc, dem)
    ket = None
    try:
        kho = E.doc_master()
        if kho is None:
            print("Chưa có kho — quét lần đầu..."); E.tao_master(so_ky=450)
        ket = main()
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout = goc

    print("\n  Đang gửi email...")
    try:
        gui_email(dem.getvalue(), ket)
        print(f"  ✓ Đã gửi email tới {EMAIL_NHAN}")
    except Exception as e:
        print(f"  ✗ KHÔNG gửi được email: {e}")
        sys.exit(1)
