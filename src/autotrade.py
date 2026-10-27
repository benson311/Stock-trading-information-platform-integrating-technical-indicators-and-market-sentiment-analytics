import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""自動交易訊號產生器。

基於已訓練模型預測下一日報酬, 套用 DMA+預測策略規則 (含台灣漲跌停),
產出逐股訊號 (BUY/SELL/HOLD) 與加權排名建議。

流程:
  1. 對每股以「全歷史」重訓模型 (或載入現成權重)。
  2. 以最新 window 預測下一日報酬。
  3. 套用策略規則產生訊號。
  4. 依 (預測報酬 × 情緒正向度) 排序, 給出買進清單。
"""
import os
import json

import numpy as np
import pandas as pd

import config
from src.dataset import load_features, FEATURE_COLS
from src.models import build_model
import torch


def predict_next_return(code: str, epochs: int = 60, model_name: str = "lstm_attention"):
    """用全歷史重訓並預測最新下一日報酬, 回傳 (pred_return, last_close, last_date)。"""
    df = load_features()
    sub = df[df["code"] == code].sort_values("date").reset_index(drop=True)
    if len(sub) < config.WINDOW_SIZE:
        return None

    feats = sub[FEATURE_COLS].astype(float).values
    close = sub["close"].astype(float).values

    # 訓練資料: 除了最後 1 點以外的全部
    X, y = [], []
    for i in range(len(sub) - config.WINDOW_SIZE):
        X.append(feats[i:i + config.WINDOW_SIZE])
        y.append(close[i + config.WINDOW_SIZE] / close[i + config.WINDOW_SIZE - 1] - 1.0)
    X = np.array(X)
    y = np.array(y)

    lo = X.min(axis=(0, 1), keepdims=True)
    hi = X.max(axis=(0, 1), keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1.0
    Xn = (X - lo) / rng

    torch.manual_seed(0)
    m = build_model(model_name, Xn.shape[2], config.WINDOW_SIZE)
    opt = torch.optim.Adam(m.parameters(), lr=config.LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(Xt)
    bs = config.BATCH_SIZE
    for _ in range(epochs):
        perm = torch.randperm(n)
        for a in range(0, n, bs):
            idx = perm[a:a + bs]
            opt.zero_grad()
            loss = loss_fn(m(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    # 最新 window 預測
    last_feat = (feats[-config.WINDOW_SIZE:] - lo.squeeze(0)) / rng.squeeze(0)
    with torch.no_grad():
        pred = m(torch.tensor(last_feat[None], dtype=torch.float32)).item()

    # 台股單日 ±10% 漲跌停: 裁切預測報酬至可行域
    pred = float(np.clip(pred, -config.PRICE_LIMIT, config.PRICE_LIMIT))
    return pred, float(close[-1]), sub["date"].iloc[-1]


def generate_signals(codes, epochs: int = 60, model_name: str = "lstm_attention"):
    """對每股產出最新訊號, 回傳 DataFrame。"""
    rows = []
    from src.price_loader import load_best_price
    for code in codes:
        try:
            pred_ret, last_close, last_date = predict_next_return(code, epochs, model_name)
        except Exception as e:
            print(f"  {code} 失敗: {e}")
            continue
        # 計算 MA5 / MA10
        df = load_best_price(code)
        if df.empty:
            continue
        c = df["close"].astype(float)
        ma5 = c.rolling(config.FAST_MA).mean().iloc[-1]
        ma10 = c.rolling(config.SLOW_MA).mean().iloc[-1]
        pred_price = last_close * (1 + pred_ret)

        signal = "HOLD"
        reason = ""
        if pred_price > ma5 * (1 + config.PRED_THRESHOLD):
            signal = "BUY"
            reason += "預測價>MA5x1.05"
        elif pred_price < ma5 * (1 - config.PRED_THRESHOLD):
            signal = "SELL"
            reason += "預測價<MA5x0.95"
        # 交叉訊號
        ma5_prev = c.rolling(config.FAST_MA).mean().iloc[-2]
        ma10_prev = c.rolling(config.SLOW_MA).mean().iloc[-2]
        if ma5 > ma10 and ma5_prev <= ma10_prev:
            signal = "BUY" if signal != "SELL" else signal
            reason += ("" if not reason else "+") + "黃金交叉"
        elif ma5 < ma10 and ma5_prev >= ma10_prev:
            signal = "SELL" if signal != "BUY" else signal
            reason += ("" if not reason else "+") + "死亡交叉"

        rows.append({"code": code, "date": last_date, "pred_return": pred_ret,
                     "last_close": last_close, "pred_price": pred_price,
                     "ma5": ma5, "ma10": ma10, "signal": signal, "reason": reason})

    sig = pd.DataFrame(rows)
    sig["score"] = sig["pred_return"]  # 排名分 (可再加入情緒)
    sig = sig.sort_values("score", ascending=False).reset_index(drop=True)
    out = os.path.join(config.RESULTS_DIR, "auto_trade_signals.csv")
    sig.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"訊號產出 {len(sig)} 檔 -> {out}")
    print(sig[["code", "pred_return", "signal"]].head(15).to_string(index=False))
    return sig


if __name__ == "__main__":
    from src.price_loader import list_all_stock_codes
    codes = list_all_stock_codes()[:20]
    generate_signals(codes, epochs=40)