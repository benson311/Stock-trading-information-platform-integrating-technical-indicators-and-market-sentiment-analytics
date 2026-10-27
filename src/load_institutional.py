import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""載入 TWSE 官方法人買賣超 (T86) 與融資融券餘額 (MI_MARGN) 資料。

T86 三大法人買賣超日報 (單位: 股), 欄位版型跨年有變動:
  2015-2017: 外資買賣超股數
  2018+    : 外陸資買賣超股數(不含外資自營商) + 外資自營商買賣超股數
  → foreign_net = 外陸資 + 外資自營商 (2018+) 或 外資 (2015-17), 概念一致
  dealer_net = 「自營商買賣超股數」合計欄 (自行買賣+避險)
  total_net  = 三大法人買賣超股數

MI_MARGN 融資融券彙總-股票 (餘額單位: 交易單位=張):
  margin_bal = 融資今日餘額 (散戶槓桿部位, 台股傳統反指標)
  short_bal  = 融券今日餘額 (放空部位, 軋空潛在動能)

快取 (parquet, 來源有新檔時自動重建):
  {PROCESSED_DIR}/institutional.parquet  (code, date, foreign_net, trust_net, dealer_net, total_net)
  {PROCESSED_DIR}/margin.parquet         (code, date, margin_bal, short_bal)
"""
import glob
import json
import os

import pandas as pd

import config
from src.price_loader import list_all_stock_codes, _to_float

T86_CACHE = os.path.join(config.PROCESSED_DIR, "institutional.parquet")
MARGIN_CACHE = os.path.join(config.PROCESSED_DIR, "margin.parquet")


def _find_col(fields, key, exact=False):
    """在 T86 fields 中找欄位索引; exact=True 需全名相符。找不到回 None。"""
    for i, f in enumerate(fields):
        if (exact and f == key) or (not exact and key in f):
            return i
    return None


def _parse_t86_file(path, universe):
    d = json.load(open(path, encoding="utf-8"))
    if d.get("stat") != "OK" or not d.get("data"):
        return []
    fields = d["fields"]
    i_code = _find_col(fields, "證券代號")
    i_f_main = _find_col(fields, "外陸資買賣超")           # 2018+
    if i_f_main is None:
        i_f_main = _find_col(fields, "外資買賣超")          # 2015-17
    i_f_hf = _find_col(fields, "外資自營商買賣超")          # 2018+
    i_trust = _find_col(fields, "投信買賣超")
    i_dealer = _find_col(fields, "自營商買賣超股數", exact=True)
    i_total = _find_col(fields, "三大法人買賣超")
    if None in (i_code, i_f_main, i_trust, i_dealer, i_total):
        raise ValueError(f"{path}: 無法辨識欄位 {fields}")

    date = os.path.basename(path)[4:12]
    rows = []
    for r in d["data"]:
        code = str(r[i_code]).strip()
        if code not in universe:
            continue
        foreign = _to_float(r[i_f_main]) or 0.0
        if i_f_hf is not None:
            foreign += _to_float(r[i_f_hf]) or 0.0
        rows.append((code, date,
                     foreign,
                     _to_float(r[i_trust]) or 0.0,
                     _to_float(r[i_dealer]) or 0.0,
                     _to_float(r[i_total]) or 0.0))
    return rows


def _parse_margin_file(path, universe):
    d = json.load(open(path, encoding="utf-8"))
    if d.get("stat") != "OK":
        return []
    # 找「融資融券彙總 (股票)」表格
    table = None
    for t in d.get("tables", []):
        if "融資融券彙總" in (t.get("title") or ""):
            table = t
            break
    if table is None or not table.get("data"):
        return []
    # 欄位位置: 0代號 1名稱 | 融資: 2買 3賣 4現償 5前日 6今日 7限額 | 融券: 8買 9賣 10現償 11前日 12今日 13限額 | 14互抵 15註記
    if len(table.get("fields", [])) < 14:
        raise ValueError(f"{path}: 欄位數異常 {table.get('fields')}")
    date = os.path.basename(path).replace("mi_margn_", "").replace(".json", "")
    rows = []
    for r in table["data"]:
        code = str(r[0]).strip()
        if code not in universe or str(r[1]).strip() in ("合計", ""):
            continue
        rows.append((code, date, _to_float(r[6]) or 0.0, _to_float(r[12]) or 0.0))
    return rows


def _newest_mtime(pattern):
    mtimes = [os.path.getmtime(f) for f in glob.glob(pattern)]
    return max(mtimes) if mtimes else 0


def _load_or_build(cache_path, pattern, parser, universe, force=False):
    """快取存在且比來源新 → 直接讀; 否則重解析。"""
    if (not force) and os.path.exists(cache_path) and \
            os.path.getmtime(cache_path) >= _newest_mtime(pattern):
        return pd.read_parquet(cache_path)
    files = sorted(glob.glob(pattern))
    rows = []
    for i, f in enumerate(files):
        try:
            rows.extend(parser(f, universe))
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"  ! 跳過異常檔案 {os.path.basename(f)}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  ...解析 {i+1}/{len(files)} 檔", flush=True)
    df = pd.DataFrame(rows, columns=["code", "date", *COLS[cache_path]])
    df = df.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"])
    df.to_parquet(cache_path, index=False)
    print(f"快取重建: {os.path.basename(cache_path)} {len(df)} 列 ({df['code'].nunique()} 檔) "
          f"{df['date'].min()} ~ {df['date'].max()}")
    return df


# 快取檔 → 資料欄位名
COLS = {
    T86_CACHE: ["foreign_net", "trust_net", "dealer_net", "total_net"],
    MARGIN_CACHE: ["margin_bal", "short_bal"],
}


def load_institutional(force=False) -> pd.DataFrame:
    """三大法人買賣超: (code, date, foreign_net, trust_net, dealer_net, total_net)。"""
    return _load_or_build(T86_CACHE, os.path.join(config.RAW_T86_DIR, "t86_*.json"),
                          _parse_t86_file, set(list_all_stock_codes()), force)


def load_margin(force=False) -> pd.DataFrame:
    """融資融券餘額: (code, date, margin_bal, short_bal)。"""
    return _load_or_build(MARGIN_CACHE, os.path.join(config.RAW_MARGIN_DIR, "mi_margn_*.json"),
                          _parse_margin_file, set(list_all_stock_codes()), force)


if __name__ == "__main__":
    inst = load_institutional()
    marg = load_margin()
    # 一致性抽核: foreign+trust+dealer ≈ total
    chk = inst.sample(min(50000, len(inst)), random_state=42)
    diff = (chk["foreign_net"] + chk["trust_net"] + chk["dealer_net"] - chk["total_net"]).abs()
    print(f"T86 加總一致性: 中位差 {diff.median():.0f} 股, 99分位 {diff.quantile(0.99):.0f} 股")
    for code in ["2330", "2317", "2002"]:
        s = inst[inst["code"] == code]
        m = marg[marg["code"] == code]
        print(f"{code}: 法人 {len(s)} 天 ({s['date'].min()}~{s['date'].max()}), "
              f"融資融券 {len(m)} 天 ({m['date'].min()}~{m['date'].max()})")
