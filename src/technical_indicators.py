import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""技術指標計算 (論文 Table 1, 共 12 個)。

趨勢: MA(10), MA(20), EMA(10), EMA(20), MACD(6,15,6), MACD(12,26,9)
動量: RSI(6), RSI(12), WILLR(14), MOM(14)
量能: OBV, ADOSC(3,10)
"""
import numpy as np
import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """輸入含 date/open/high/low/close/volume 的 df, 回傳加入指標之 df。"""
    prices = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    out = df.copy()

    # --- 趨勢 ---
    out["MA10"] = prices.rolling(10).mean()
    out["MA20"] = prices.rolling(20).mean()
    out["EMA10"] = prices.ewm(span=10, adjust=False).mean()
    out["EMA20"] = prices.ewm(span=20, adjust=False).mean()

    out["MACD_6_15_6"] = macd(prices, 6, 15, 6)
    out["MACD_12_26_9"] = macd(prices, 12, 26, 9)

    # --- 動量 ---
    out["RSI6"] = rsi(prices, 6)
    out["RSI12"] = rsi(prices, 12)
    out["WILLR14"] = willr(high, low, prices, 14)
    out["MOM14"] = prices - prices.shift(14)

    # --- 量能 ---
    out["OBV"] = obv(prices, volume)
    out["ADOSC_3_10"] = adosc(high, low, prices, volume, 3, 10)

    return out


def macd(prices, fast, slow, signal):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=signal, adjust=False).mean()
    return (diff - dea) * 2  # MACD 柱


def rsi(prices, period):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def willr(high, low, close, period):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll + 1e-10)


def obv(prices, volume):
    direction = np.sign(prices.diff().fillna(0))
    return (direction * volume).cumsum()


def adosc(high, low, close, volume, fast, slow):
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
    money_flow_volume = money_flow_multiplier * volume
    ad = money_flow_volume.cumsum()
    ema_fast = ad.ewm(span=fast, adjust=False).mean()
    ema_slow = ad.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow