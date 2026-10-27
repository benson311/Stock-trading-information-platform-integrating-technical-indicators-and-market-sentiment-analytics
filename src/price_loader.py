import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""股價資料統一載入器：Yahoo (268 檔) 與 TWSE 官方 (54 檔) 合併。"""
import os
import json
import glob
import re
import datetime as dt

import pandas as pd

from src.taiwan_calendar import roc_to_datetime
import config


def _to_float(v):
    """相容 TWSE 數值字串: 去除逗號、'X' 前綴, '--'/空 回 None。"""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "--", "---"):
        return None
    s = s.replace(",", "")
    s = re.sub(r"^[^0-9.\-]+", "", s)  # 去除前綴 (如 'X')
    if s in ("", "-", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_yahoo_stock(code: str) -> pd.DataFrame:
    """讀取 stocks/{code}/price/yahoo_{code}_{year}.csv 並合併為單一 df。

    欄位: date, open, high, low, close, volume (Yahoo 為還原/調整後價)。
    """
    price_dir = os.path.join(config.RAW_STOCKS_DIR, code, "price")
    files = sorted(glob.glob(os.path.join(price_dir, f"yahoo_{code}_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8")
        df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]  # 去除 BOM
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["close"]).drop_duplicates(subset=["date"]).sort_values("date")
    return out.reset_index(drop=True)


def load_twse_stock(code: str) -> pd.DataFrame:
    """讀取 twse/stock_day/stock_day_{code}_{yyyymmdd}.json 並合併。

    欄位 (民國年): 日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價,
    漲跌價差, 成交筆數, 註記。
    轉為西元年, 補上 volume (成交股數)。
    """
    pattern = os.path.join(config.RAW_TWSE_DIR, f"stock_day_{code}_*.json")
    files = sorted(glob.glob(pattern))
    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("stat") != "OK" or "data" not in data:
            continue
        for row in data.get("data", []):
            if len(row) < 7:
                continue
            records.append({
                "date": roc_to_datetime(row[0]).strftime("%Y-%m-%d"),
                "volume": _to_float(row[1]) or 0,
                "amount": _to_float(row[2]) or 0,
                "open": _to_float(row[3]),
                "high": _to_float(row[4]),
                "low": _to_float(row[5]),
                "close": _to_float(row[6]),
                "change": _to_float(row[7]),
            })
    if not records:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(records)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).drop_duplicates(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def load_best_price(code: str) -> pd.DataFrame:
    """優先採用 TWSE 官方 (含成交筆數、較準), 不足部分用 Yahoo 補。

    回傳欄位: date, open, high, low, close, volume。
    """
    twse = load_twse_stock(code)
    yahoo = load_yahoo_stock(code)

    if twse.empty and yahoo.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    if not twse.empty:
        cols = ["date", "open", "high", "low", "close", "volume"]
        used = twse[cols].copy()
    elif not yahoo.empty:
        used = yahoo[["date", "open", "high", "low", "close", "volume"]].copy()
    else:
        used = twse

    # 若 TWSE 缺早期/晚期資料, 用 Yahoo 補缺口 (Yahoo 涵蓋 2011–2026)
    if not twse.empty and not yahoo.empty:
        twse_min = used["date"].min()
        twse_max = used["date"].max()
        ycols = ["date", "open", "high", "low", "close", "volume"]
        prepend = yahoo[yahoo["date"] < twse_min][ycols]
        append = yahoo[yahoo["date"] > twse_max][ycols]
        used = pd.concat([prepend, used, append], ignore_index=True)

    used = used.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return used


def list_all_stock_codes() -> list[str]:
    """列出所有可用的股票代碼 (Yahoo 目錄 ∪ TWSE 檔案)。"""
    codes = set()
    if os.path.isdir(config.RAW_STOCKS_DIR):
        for d in os.listdir(config.RAW_STOCKS_DIR):
            if os.path.isdir(os.path.join(config.RAW_STOCKS_DIR, d)):
                codes.add(d)
    if os.path.isdir(config.RAW_TWSE_DIR):
        for f in os.listdir(config.RAW_TWSE_DIR):
            # stock_day_{code}_{date}.json
            name = f.replace("stock_day_", "").replace(".json", "")
            code = name.split("_")[0]
            codes.add(code)
    return sorted(codes)


if __name__ == "__main__":
    codes = list_all_stock_codes()
    print(f"可分析股票代碼數: {len(codes)}")
    for c in ["1101", "2330", "2317", "2002", "2618"]:
        if c in codes:
            df = load_best_price(c)
            print(f"{c}: {len(df)} 天, {df['date'].min()} -> {df['date'].max()}")