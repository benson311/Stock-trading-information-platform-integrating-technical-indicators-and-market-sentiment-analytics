import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""由既有全量預測結果提取「最新自動交易訊號」(不重訓)。

讀取 results/predictions_{model}.json 的最後一個交易日預測,
套用 DMA+預測策略規則, 產出 BUY/SELL/HOLD 訊號與加權排名。
"""
import os
import json

import numpy as np
import pandas as pd

import config
from src.price_loader import load_best_price, list_all_stock_codes


def latest_signal_from_predictions(model="lstm_attention"):
    preds_file = os.path.join(config.RESULTS_DIR, f"predictions_{model}.json")
    if not os.path.exists(preds_file):
        print("無預測檔, 請先跑 run_pipeline")
        return None
    with open(preds_file, encoding="utf-8") as f:
        preds = json.load(f)

    rows = []
    for code, p in preds.items():
        dates = p["dates"]
        pred_ret = p["pred_return"]
        pred_price = p["pred_price"]
        if not dates:
            continue
        last_pred_ret = pred_ret[-1]
        last_pred_price = pred_price[-1]

        df = load_best_price(code)
        if df.empty:
            continue
        c = df["close"].astype(float)
        if len(c) < config.SLOW_MA:
            continue
        # 以「最後已知收盤」-> MA 需對齊預測時點
        # 預測檔最後一筆對應測試集最後一日; 取該日期收盤計算 MA
        last_date = dates[-1]
        sub = df[df["date"] <= last_date]
        if sub.empty:
            continue
        closes = sub["close"].astype(float)
        ma5 = closes.rolling(config.FAST_MA).mean().iloc[-1]
        ma10 = closes.rolling(config.SLOW_MA).mean().iloc[-1]
        ma5_prev = closes.rolling(config.FAST_MA).mean().iloc[-2]
        ma10_prev = closes.rolling(config.SLOW_MA).mean().iloc[-2]

        signal = "HOLD"
        reason = ""
        if last_pred_price > ma5 * (1 + config.PRED_THRESHOLD):
            signal = "BUY"
            reason += "預測價>MA5x1.05"
        elif last_pred_price < ma5 * (1 - config.PRED_THRESHOLD):
            signal = "SELL"
            reason += "預測價<MA5x0.95"
        if ma5 > ma10 and ma5_prev <= ma10_prev:
            signal = "BUY" if signal != "SELL" else signal
            reason += ("" if not reason else "+") + "黃金交叉"
        elif ma5 < ma10 and ma5_prev >= ma10_prev:
            signal = "SELL" if signal != "BUY" else signal
            reason += ("" if not reason else "+") + "死亡交叉"

        rows.append({"code": code, "date": last_date, "pred_return": last_pred_ret,
                     "pred_price": last_pred_price, "ma5": ma5, "ma10": ma10,
                     "signal": signal, "reason": reason})

    sig = pd.DataFrame(rows)
    sig["score"] = sig["pred_return"]
    sig = sig.sort_values("score", ascending=False).reset_index(drop=True)
    out = os.path.join(config.RESULTS_DIR, "auto_trade_signals.csv")
    sig.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"自動交易訊號 {len(sig)} 檔 -> {out}")
    print("BUY/SELL/HOLD 分布:", sig["signal"].value_counts().to_dict())
    print("\n預測報酬前 15 (潛在買進):")
    print(sig.head(15)[["code", "pred_return", "signal"]].to_string(index=False))
    print("\n賣出訊號:")
    print(sig[sig["signal"] == "SELL"][["code", "pred_return", "reason"]].head(10).to_string(index=False))
    return sig


if __name__ == "__main__":
    latest_signal_from_predictions()