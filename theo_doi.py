"""
THEO DÕI TIẾN CỨU — ghi nhận dự báo TRƯỚC khi quay, chấm điểm SAU khi quay.

VÌ SAO CẦN:
    Mọi backtest đều bị nhiễm bởi việc dò tổ hợp trọng số trên chính dữ liệu
    dùng để đánh giá. Cách duy nhất sạch là ghi dự báo ra file, commit lên Git
    (dấu thời gian không sửa được), rồi chấm điểm sau khi kết quả có.

CÁCH QUY CÔNG CHO TÍN HIỆU:
    Bộ số ensemble là tổng có trọng số của nhiều tín hiệu — không thể tách công.
    Nên mỗi tín hiệu chạy SOLO như một danh mục riêng: nó tự chọn top-K của mình.
    Cộng thêm một danh mục NGẪU NHIÊN làm đối chứng. Tổng cộng 14 danh mục.

FILE:
    theo_doi/YYYY-MM.jsonl   — mỗi dòng 1 bản ghi (ngày × đài × module)
                               Nối thêm, KHÔNG BAO GIỜ ghi đè.
"""
import json, os, glob
from datetime import datetime, timedelta, timezone
import numpy as np
import engine as E

THU_MUC = "theo_doi"
VN = timezone(timedelta(hours=7))


def _duong_dan(ngay):
    os.makedirs(THU_MUC, exist_ok=True)
    return os.path.join(THU_MUC, f"{ngay:%Y-%m}.jsonl")


# ==============================================================================
#  GHI DỰ BÁO (chạy 6h sáng, TRƯỚC khi quay)
# ==============================================================================

def _solo_2so(chuoi, pool, tg, mtc, K, cap):
    """Mỗi tín hiệu 2 số tự chọn top-K của riêng nó."""
    Z, _ = E.ma_tran_tin_hieu(chuoi, pool, tg, mtc)
    ra = {}
    for j, nm in enumerate(E.SIG_NAMES):
        if Z[j].std() < 1e-9:            # tín hiệu suy biến -> bỏ qua
            continue
        w = np.zeros(len(E.SIG_NAMES)); w[j] = 1.0
        sel, _ = E.select_top(w @ Z, K, cap)
        ra[nm] = sorted(f"{i:02d}" for i in sel)
    return ra


def _solo_3so(g3, tg, K, cap):
    Z, _ = E.build_signal_matrix_3(g3, tg)
    ra = {}
    for j, nm in enumerate(E.SIG3_NAMES):
        if Z[j].std() < 1e-9:
            continue
        w = np.zeros(len(E.SIG3_NAMES)); w[j] = 1.0
        sel, _ = E.select_top3(w @ Z, K, cap)
        ra[nm] = sorted(f"{i:03d}" for i in sel)
    return ra


def ghi_du_bao(ket, ngay_moc):
    """ket: danh sách kết quả từ chay_dai_v8. Ghi 14 danh mục cho mỗi đài×module."""
    rng = np.random.default_rng(int(f"{ngay_moc:%Y%m%d}"))
    dong = []
    for r in ket:
        tg = r.get("_toan_giai")
        if tg is None:
            continue
        pool = [[int(x[-2:]) for x in ky] for ky in tg]
        g3 = [E.lo3_cua_ky(ky) for ky in tg]
        for m in r["modules"]:
            key, K = m["key"], m["K"]
            if key == "3SO":
                cap = E.tinh_cap3(K)
                solo = _solo_3so(g3, tg, K, cap)
                nn = sorted(f"{i:03d}" for i in rng.choice(1000, K, replace=False))
            elif key in ("DB", "LO2"):
                cap = E.tinh_cap(K)
                ch = tg if key == "LO2" else [[k[0]] for k in tg]
                mtc = [{int(k[0][-2:])} for k in tg] if key == "DB" else None
                solo = _solo_2so(ch, pool, tg, mtc, K, cap)
                nn = sorted(f"{i:02d}" for i in rng.choice(100, K, replace=False))
            else:
                continue
            dong.append({
                "ngay": f"{ngay_moc:%Y-%m-%d}", "dai": r["stt"], "ten_dai": r["dai"],
                "khu": r["khu"], "module": key, "K": K, "n_ky_train": r["n_ky"],
                "ensemble": m["so"], "trong_so": m["tin_hieu"],
                "solo": solo, "ngau_nhien": nn,
                "ket_qua": None, "n_lo": None,
                "trung_ensemble": None, "trung_ngau_nhien": None, "trung_solo": None,
                "ghi_luc": datetime.now(VN).isoformat(timespec="seconds"),
            })
    if not dong:
        print("  ⚠ Không ghi được bản ghi theo dõi nào.")
        return 0
    with open(_duong_dan(ngay_moc), "a", encoding="utf-8") as f:
        for d in dong:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_dm = len(dong[0]["solo"]) + 2
    print(f"  ✓ Đã ghi {len(dong)} bản ghi theo dõi ({n_dm} danh mục mỗi bản) "
          f"→ {_duong_dan(ngay_moc)}")
    return len(dong)


# ==============================================================================
#  CHẤM ĐIỂM (chạy 22h tối, SAU khi quay)
# ==============================================================================

def _muc_tieu_that(giai, module):
    if module == "DB":   return [giai[0][-2:]]
    if module == "LO2":  return [x[-2:] for x in giai]
    if module == "3SO":  return E.lo3_cua_ky(giai)
    return []


def cham_diem(ngay=None, im_lang=False):
    """Dò kết quả thật rồi điền vào các bản ghi còn trống."""
    hom_nay = ngay or datetime.now(VN).date()
    path = _duong_dan(hom_nay)
    if not os.path.exists(path):
        print(f"  Chưa có {path} — không có gì để chấm."); return 0

    ban_ghi = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    can_cham = [b for b in ban_ghi if b["trung_ensemble"] is None]
    if not can_cham:
        print("  Mọi bản ghi đã được chấm."); return 0

    # --- Tải kết quả thật, mỗi đài chỉ tải 1 lần ---
    cache = {}
    da_cham = 0
    for b in ban_ghi:
        if b["trung_ensemble"] is not None:
            continue
        stt = b["dai"]; ngay_b = b["ngay"]
        khoa = (stt, ngay_b)
        if khoa not in cache:
            try:
                _, ng, nf, tg, _ = E.lay_du_lieu(stt, 12)
                cache[khoa] = None
                for d, g in zip(nf, tg):
                    if d and d.isoformat() == ngay_b:
                        cache[khoa] = g
                        break
            except Exception as e:
                if not im_lang:
                    print(f"     ✗ [{stt}] {b['ten_dai']}: {e}")
                cache[khoa] = None
        giai = cache[khoa]
        if giai is None:
            continue

        thuc = _muc_tieu_that(giai, b["module"])
        tap = set(thuc)
        b["ket_qua"] = thuc
        b["n_lo"] = len(thuc)
        b["trung_ensemble"] = sorted(tap & set(b["ensemble"]))
        b["trung_ngau_nhien"] = sorted(tap & set(b["ngau_nhien"]))
        b["trung_solo"] = {k: sorted(tap & set(v)) for k, v in b["solo"].items()}
        b["cham_luc"] = datetime.now(VN).isoformat(timespec="seconds")
        da_cham += 1

    with open(path, "w", encoding="utf-8") as f:
        for b in ban_ghi:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    if not im_lang:
        print(f"\n  Đã chấm {da_cham}/{len(can_cham)} bản ghi")
        _in_ket_qua_ngay([b for b in ban_ghi if b["ngay"] == str(hom_nay)
                          and b["trung_ensemble"] is not None])
    return da_cham


def _in_ket_qua_ngay(bg):
    if not bg:
        return
    print("\n" + "=" * 78)
    print(f"  KẾT QUẢ NGÀY {bg[0]['ngay']}")
    print("=" * 78)
    print(f"  {'Đài':<18}{'Module':<8}{'Lô':>4}{'Ensemble':>10}{'Ngẫu nhiên':>12}"
          f"   Số trúng")
    print("  " + "-" * 74)
    te = tn = 0
    for b in sorted(bg, key=lambda x: (x["dai"], x["module"])):
        e = len(b["trung_ensemble"]); nn = len(b["trung_ngau_nhien"])
        te += e; tn += nn
        so = ",".join(b["trung_ensemble"][:6]) or "—"
        if len(b["trung_ensemble"]) > 6: so += "..."
        print(f"  {b['ten_dai']:<18}{b['module']:<8}{b['n_lo']:>4}{e:>10}{nn:>12}   {so}")
    print("  " + "-" * 74)
    print(f"  {'TỔNG':<30}{te:>10}{tn:>12}")
    print(f"\n  (Ngẫu nhiên = danh mục đối chứng bốc bừa cùng cỡ, chạy song song)")

    # --- tín hiệu nào trúng nhiều nhất HÔM NAY ---
    dem = {}
    for b in bg:
        for k, v in (b["trung_solo"] or {}).items():
            dem[k] = dem.get(k, 0) + len(v)
    if dem:
        print(f"\n  Tín hiệu solo trúng nhiều nhất hôm nay:")
        for k, v in sorted(dem.items(), key=lambda x: -x[1])[:5]:
            print(f"     {k:<16} {v} lượt")
        print(f"  ⚠ MỘT NGÀY KHÔNG NÓI LÊN ĐIỀU GÌ. Cần ~60 ngày mới đủ mẫu.")
    print("=" * 78)


def thong_ke_nhanh():
    """Tổng hợp mọi tháng — dùng để xem đã tích luỹ bao nhiêu."""
    files = sorted(glob.glob(os.path.join(THU_MUC, "*.jsonl")))
    if not files:
        print("  Chưa có dữ liệu theo dõi."); return
    tong, da_cham, theo_mod = 0, 0, {}
    for f in files:
        for l in open(f, encoding="utf-8"):
            if not l.strip(): continue
            b = json.loads(l); tong += 1
            if b["trung_ensemble"] is not None:
                da_cham += 1
                m = b["module"]
                d = theo_mod.setdefault(m, {"ky": 0, "lo": 0, "ens": 0, "nn": 0})
                d["ky"] += 1; d["lo"] += b["n_lo"]
                d["ens"] += len(b["trung_ensemble"]); d["nn"] += len(b["trung_ngau_nhien"])
    print(f"\n  Tích luỹ: {tong} bản ghi, {da_cham} đã chấm ({len(files)} tháng)")
    if theo_mod:
        print(f"  {'Module':<8}{'Kỳ':>6}{'Quan sát':>10}{'Ensemble':>11}{'Ngẫu nhiên':>13}")
        print("  " + "-" * 50)
        for m, d in theo_mod.items():
            print(f"  {m:<8}{d['ky']:>6}{d['lo']:>10,}"
                  f"{d['ens']/max(d['lo'],1):>10.2%}{d['nn']/max(d['lo'],1):>13.2%}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "thongke":
        thong_ke_nhanh()
    else:
        cham_diem()
