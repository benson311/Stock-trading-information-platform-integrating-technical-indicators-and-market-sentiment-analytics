import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""資料集建置: 滑動窗、正規化、訓練/測試切分 (論文 Section 3.1.3)。

每筆樣本: X = (window=20, n_features), y = 下一交易日收盤價 (正規化)。
正規化: 僅以訓練集統計量 fit (min-max), 避免資料洩漏。
"""
import os

import numpy as np
import pandas as pd

import config
from src.build_features import TECH_COLS, SENT_COLS, SECTOR_COLS, INST_COLS

FEATURE_COLS = TECH_COLS + SENT_COLS + SECTOR_COLS + ["news_net"] + INST_COLS


_FEATURES_CACHE = None


def load_features() -> pd.DataFrame:
    """模組級快取: 302 檔逐股訓練時避免重複讀取近百萬列 parquet。"""
    global _FEATURES_CACHE
    if _FEATURES_CACHE is None:
        _FEATURES_CACHE = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "features.parquet"))
    return _FEATURES_CACHE


def make_sequences_for_stock(df: pd.DataFrame, window: int = config.WINDOW_SIZE,
                             horizon: int = config.FORECAST_HORIZON, target: str = "return"):
    """對單一股票 df (已依 date 排序) 產生 (X, y, dates, close_t)。

    X shape (n_samples, window, n_features)。
    target='return': y = 未來 horizon 日報酬率 (close[t+h]/close[t+h-1] - 1)
    target='price' : y = 未來 horizon 日收盤價
    同時回傳 close_t (預測時點的收盤價), 供還原價格用。
    """
    feats = df[FEATURE_COLS].astype(float).values
    close = df["close"].astype(float).values
    dates = df["date"].values

    X, y, out_dates, close_t = [], [], [], []
    for i in range(len(df) - window - horizon + 1):
        t_predict = i + window - 1           # 最後一個已知日
        t_target = i + window + horizon - 1  # 目標日
        X.append(feats[i:i + window])
        if target == "return":
            y.append(close[t_target] / close[t_target - 1] - 1.0)
        else:
            y.append(close[t_target])
        out_dates.append(dates[t_target])
        close_t.append(close[t_predict])
    return np.array(X), np.array(y), np.array(out_dates), np.array(close_t)


def split_train_test(X, y, dates, train_ratio: float = config.TRAIN_RATIO):
    """時間順序切分 (不 shuffle)。"""
    n = len(X)
    n_train = int(n * train_ratio)
    return (X[:n_train], y[:n_train], dates[:n_train],
            X[n_train:], y[n_train:], dates[n_train:])


def normalize(X_train, X_test=None):
    """min-max 正規化 (fit 於訓練集), 回傳 (X_train_norm, X_test_norm, scaler_params)。"""
    lo = X_train.min(axis=(0, 1), keepdims=True)
    hi = X_train.max(axis=(0, 1), keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1.0
    Xt = (X_train - lo) / rng
    if X_test is not None:
        Xe = (X_test - lo) / rng
    else:
        Xe = None
    return Xt, Xe, (lo, rng)


def normalize_single(X, lo, rng):
    return (X - lo) / rng


def prepare_stock_data(code: str, target: str = "return"):
    """載入單股特徵並切分, 回傳 dict (方便訓練/評估)。"""
    df = load_features()
    sub = df[df["code"] == code].sort_values("date").reset_index(drop=True)
    X, y, dates, close_t = make_sequences_for_stock(sub, target=target)
    if len(X) < config.WINDOW_SIZE:
        return None
    Xtr, ytr, dtr, Xte, yte, dte = split_train_test(X, y, dates)
    ctr = close_t[:len(Xtr)]
    cte = close_t[len(Xtr):]
    Xtr_n, Xte_n, params = normalize(Xtr, Xte)

    if target == "price":
        y_lo, y_hi = ytr.min(), ytr.max()
        yrng = (y_hi - y_lo) or 1.0
        ytr_n = (ytr - y_lo) / yrng
        yte_n = (yte - y_lo) / yrng
    else:
        ytr_n, yte_n = ytr, yte  # 報酬率不另做 min-max (台灣 ±10% 自然有界)
        y_lo, yrng = 0.0, 1.0

    return {
        "code": code, "target": target,
        "X_train": Xtr_n, "y_train": ytr_n, "dates_train": dtr, "close_train": ctr,
        "X_test": Xte_n, "y_test": yte_n, "dates_test": dte, "close_test": cte,
        "y_lo": y_lo, "y_range": yrng, "closes": sub["close"].values,
    }


if __name__ == "__main__":
    d = prepare_stock_data("2330")
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)