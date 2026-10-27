import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""特徵建置: 股票價格 + 12 技術指標 + 情緒指數(自身/產業) + 靜態新聞情緒。

產業情緒 (論文 Section 2.1.3 相關產業概念, 台灣改以 TWSE 產業別分群):
  對每股 s, 其「產業情緒」= 同產業別內「其他」股票之情緒加總 (排除自身)。
"""
import os
import json

import numpy as np
import pandas as pd

import config
from src.price_loader import load_best_price, list_all_stock_codes
from src.technical_indicators import compute_technical_indicators
from src.load_institutional import load_institutional, load_margin

TECH_COLS = ["MA10", "MA20", "EMA10", "EMA20", "MACD_6_15_6", "MACD_12_26_9",
             "RSI6", "RSI12", "WILLR14", "MOM14", "OBV", "ADOSC_3_10"]
SENT_COLS = ["sent_pos", "sent_neg"]  # 現在是 w1*news + w2*review 的加權合成
SECTOR_COLS = ["sector_pos", "sector_neg"]
INST_COLS = [
    # 法人買賣超 / 成交量 (單位一致: 股)
    "foreign_net_vol", "trust_net_vol", "dealer_net_vol", "total_net_vol",
    # 法人買賣超 60 日滾動 z-score (跨股規模標準化)
    "foreign_net_z", "trust_net_z", "dealer_net_z", "total_net_z",
    # 融資/融券餘額變化率與 60 日滾動 z-score (台股散戶槓桿反指標 / 軋空動能)
    "margin_chg", "margin_z", "short_chg", "short_z",
]
INST_NETS = ["foreign", "trust", "dealer", "total"]
Z_WINDOW = 60      # 滾動標準化窗長 (交易日)
Z_MIN_PERIODS = 20


def load_sentiment_index():
    """載入「純新聞」情緒指數 (鉅亨+Yahoo, 不含 PTT)。

    欄位對映到下游沿用的名稱:
      news_pos -> review_pos (→ sent_pos)
      news_neg -> review_neg (→ sent_neg)
    """
    df = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "news_sentiment_index.parquet"))
    df["date"] = df["date"].astype(str)
    df = df.rename(columns={"news_pos": "review_pos", "news_neg": "review_neg"})
    return df[["code", "date", "review_pos", "review_neg", "news_count"]]


def load_sector_index():
    """(保留介面; 產業情緒由 build_sector_sentiment 從新聞指數重算。)"""
    path = os.path.join(config.PROCESSED_DIR, "news_sector_sentiment.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["date"] = df["date"].astype(str)
    return df


def build_sector_sentiment(sent_idx: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """回傳 (code, date, sector_pos, sector_neg) — 同產業別其他股票加總。"""
    df = sent_idx[["code", "date", "review_pos", "review_neg"]].copy()
    df["industry"] = df["code"].map(lambda c: meta.get(c, {}).get("industry", "00"))

    # 產業總和 (使用已加權的 review_pos/review_neg，已含新聞權重)
    ind = df.groupby(["industry", "date"], as_index=False).agg(
        ind_pos=("review_pos", "sum"), ind_neg=("review_neg", "sum"))
    df = df.merge(ind, on=["industry", "date"], how="left")
    df["sector_pos"] = df["ind_pos"] - df["review_pos"]
    df["sector_neg"] = df["ind_neg"] - df["review_neg"]
    return df[["code", "date", "sector_pos", "sector_neg"]]


def _rolling_z(s: pd.Series, window: int = Z_WINDOW, min_periods: int = Z_MIN_PERIODS) -> pd.Series:
    """60 日滾動 z-score; 樣本不足或 std=0 → 0 (中性)。"""
    roll = s.rolling(window, min_periods=min_periods)
    z = (s - roll.mean()) / roll.std()
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def add_institutional_features(df: pd.DataFrame, inst_stock: pd.DataFrame,
                                marg_stock: pd.DataFrame) -> pd.DataFrame:
    """併入三大法人買賣超與融資融券特徵 (df 已依 date 排序、含 volume 欄)。

    - *_net_vol : 買賣超股數 / 成交股數 (Yahoo 與 TWSE 官方 volume 皆為「股」)
    - *_net_z   : 買賣超股數 60 日滾動 z-score
    - margin/short_chg : 融資/融券餘額日變化率 (clip ±100%, 防新開放信用交易暴量)
    - margin/short_z   : 餘額 60 日滾動 z-score (散戶槓桿/軋空部位)
    缺資料 (2011-2014 法人、上櫃股、非信用交易股) → 0 (中性)。
    """
    if not inst_stock.empty:
        df = df.merge(inst_stock, on="date", how="left")
    else:
        for c in INST_NETS:
            df[f"{c}_net"] = np.nan
    if not marg_stock.empty:
        df = df.merge(marg_stock, on="date", how="left")
    else:
        df["margin_bal"] = df["short_bal"] = np.nan

    vol = df["volume"].replace(0, np.nan)
    for c in INST_NETS:
        net = df[f"{c}_net"].fillna(0.0)
        df[f"{c}_net_vol"] = (net / vol).fillna(0.0)
        df[f"{c}_net_z"] = _rolling_z(net)

    for bal, chg in (("margin_bal", "margin_chg"), ("short_bal", "short_chg")):
        s = df[bal]
        s_ff = s.ffill()  # 餘額缺日沿用前值 (pct_change 跨 NaN 缺口比對前一有效值)
        df[chg] = (s_ff / s_ff.shift(1) - 1.0).clip(-1.0, 1.0) \
            .replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df[f"{chg[:-4]}_z"] = _rolling_z(s_ff)

    for c in INST_COLS:
        df[c] = df[c].fillna(0.0)
    return df


def build_stock_features(code: str, sent_idx: pd.DataFrame, sector: pd.DataFrame,
                         news_static: pd.DataFrame = None,
                         inst: pd.DataFrame = None,
                         marg: pd.DataFrame = None) -> pd.DataFrame:
    price = load_best_price(code)
    if price.empty:
        return None
    price = compute_technical_indicators(price)

    # 情緒自身
    s = sent_idx[sent_idx["code"] == code][["date", "review_pos", "review_neg"]]
    df = price.merge(s, on="date", how="left")
    df = df.rename(columns={"review_pos": "sent_pos", "review_neg": "sent_neg"})

    # 產業情緒
    sec = sector[sector["code"] == code][["date", "sector_pos", "sector_neg"]]
    df = df.merge(sec, on="date", how="left")

    # 填充無情緒日為中性 (0)
    for c in SENT_COLS + SECTOR_COLS:
        df[c] = df[c].fillna(0.0)

    # 靜態新聞情緒 (常數特徵)
    if news_static is not None and len(news_static):
        row = news_static[news_static["code"] == code]
        if len(row):
            df["news_net"] = float(row["news_net"].iloc[0])
        else:
            df["news_net"] = 0.0
    else:
        df["news_net"] = 0.0

    # 三大法人買賣超 + 融資融券
    inst_stock = inst[inst["code"] == code] if inst is not None else pd.DataFrame()
    marg_stock = marg[marg["code"] == code] if marg is not None else pd.DataFrame()
    df = add_institutional_features(df, inst_stock, marg_stock)

    df = df.dropna(subset=TECH_COLS + ["close"]).reset_index(drop=True)
    return df


def build_all_features():
    meta = json.load(open(os.path.join(config.PROCESSED_DIR, "stock_metadata.json"), encoding="utf-8"))
    sent_idx = load_sentiment_index()
    sector = build_sector_sentiment(sent_idx, meta)
    news_static = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "news_sentiment_static.parquet"))
    # 三大法人 + 融資融券 (日期轉 YYYY-MM-DD 對齊價格表)
    inst = load_institutional()
    marg = load_margin()
    for d in (inst, marg):
        d["date"] = pd.to_datetime(d["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

    codes = list_all_stock_codes()
    feats = []
    for i, code in enumerate(codes):
        df = build_stock_features(code, sent_idx, sector, news_static, inst, marg)
        if df is not None and len(df) >= 120:
            df["code"] = code
            feats.append(df)
        if (i + 1) % 50 == 0:
            print(f"  ...特徵建置 {i+1}/{len(codes)}")

    full = pd.concat(feats, ignore_index=True)
    out = os.path.join(config.PROCESSED_DIR, "features.parquet")
    full.to_parquet(out, index=False)
    print(f"特徵表: {len(full)} 列, {full['code'].nunique()} 檔 -> {out}")
    print(f"特徵欄位: {TECH_COLS + SENT_COLS + SECTOR_COLS + ['news_net'] + INST_COLS}")
    return full


if __name__ == "__main__":
    build_all_features()