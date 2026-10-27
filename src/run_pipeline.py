import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""整合執行器: 對每檔股票「訓練一次」, 同時產出:

  1. 準確度指標 (MSE/RMSE/MAE/方向準確率) -> results/accuracy_{model}.csv
  2. 測試集預測 (報酬率) -> results/predictions_{model}.json
  3. 回測 (DMA+預測 vs DMA, 台灣規則) -> results/backtest_{model}.csv

用法: python -m src.run_pipeline [model] [epochs] [limit]
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

import config
from src.price_loader import list_all_stock_codes, load_best_price
from src.train import run_stock
from src.backtest import run_backtest


def run_one(code, model, epochs, save_model=True, load_model=False):
    r = run_stock(code, model_name=model, epochs=epochs, target="return",
                  save_model=save_model, load_model=load_model)
    if r is None:
        return None

    # 準確度
    acc_row = {k: r[k] for k in ("code", "model", "MSE", "RMSE", "MAE", "direction_acc")}

    # 預測
    pred_dates = list(r["dates_test"])
    pred_return = list(np.asarray(r["y_pred"], dtype=float))
    pred_price = list(np.asarray(r["pred_price"], dtype=float))

    # 回測 (需完整收盤序列以算 MA, 與測試集日期對齊)
    df = load_best_price(code)
    if df.empty:
        return {"acc": acc_row, "preds": (pred_dates, pred_return, pred_price)}

    close = df.set_index("date")["close"].astype(float)
    # 回測區間 = 測試集期間的收盤
    sub = close.reindex(pred_dates)
    pr = pd.Series(pred_return, index=pred_dates)
    valid = sub.notna() & pr.notna()
    sub = sub[valid]
    pr = pr[valid]
    bt_metrics = {}
    bt_dma = {}
    if len(sub) > 30:
        _, m_with, eq_with = run_backtest(sub, pr, use_prediction=True)
        _, m_dma, eq_dma = run_backtest(sub, pr, use_prediction=False)
        bt_metrics = m_with
        bt_dma = m_dma

    return {
        "acc": acc_row,
        "preds": (pred_dates, pred_return, pred_price),
        "bt": bt_metrics,
        "bt_dma": bt_dma,
    }


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    model = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "lstm_attention"
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    epochs = int(pos[1]) if len(pos) > 1 else config.EPOCHS
    limit = int(pos[2]) if len(pos) > 2 else None
    save_model = "--no-save" not in flags     # 預設: 自動存權重至 models/
    load_model = "--load-weights" in flags   # 有存檔則載入, 跳過訓練 (快速重評估/回測)

    codes = list_all_stock_codes()
    if limit:
        codes = codes[:limit]

    acc_csv = os.path.join(config.RESULTS_DIR, f"accuracy_{model}.csv")
    bt_csv = os.path.join(config.RESULTS_DIR, f"backtest_{model}.csv")
    preds_json = os.path.join(config.RESULTS_DIR, f"predictions_{model}.json")

    done = set()
    if os.path.exists(acc_csv):
        done = set(pd.read_csv(acc_csv)["code"].astype(str))
    codes = [c for c in codes if c not in done]

    preds_all = {}
    if os.path.exists(preds_json):
        with open(preds_json, encoding="utf-8") as f:
            preds_all = json.load(f)

    print(f"[{model}] pipeline 待處理 {len(codes)} 檔, epochs={epochs}, "
          f"存權重={save_model}, 載入權重={load_model}")
    t0 = time.time()
    acc_header = not os.path.exists(acc_csv)
    bt_header = not os.path.exists(bt_csv)

    for i, code in enumerate(codes):
        try:
            out = run_one(code, model, epochs, save_model=save_model, load_model=load_model)
        except Exception as e:
            print(f"  {code} 失敗: {e}")
            continue
        if out is None:
            continue

        pd.DataFrame([out["acc"]]).to_csv(acc_csv, mode="a", header=acc_header, index=False)
        acc_header = False

        if out.get("bt"):
            row = {"code": code, "model": model,
                   "with_ret": out["bt"]["total_return"],
                   "with_sharpe": out["bt"]["sharpe"],
                   "with_maxdd": out["bt"]["max_drawdown"],
                   "with_final": out["bt"]["final_assets"],
                   "dma_ret": out["bt_dma"]["total_return"],
                   "dma_sharpe": out["bt_dma"]["sharpe"],
                   "dma_maxdd": out["bt_dma"]["max_drawdown"],
                   "dma_final": out["bt_dma"]["final_assets"]}
            pd.DataFrame([row]).to_csv(bt_csv, mode="a", header=bt_header, index=False)
            bt_header = False

        preds_all[code] = {"dates": out["preds"][0],
                           "pred_return": out["preds"][1],
                           "pred_price": out["preds"][2]}
        if (i + 1) % 10 == 0:
            with open(preds_json, "w", encoding="utf-8") as f:
                json.dump(preds_all, f)
            print(f"  [{model}] {i+1}/{len(codes)} done ({time.time()-t0:.0f}s), last {code}")

    with open(preds_json, "w", encoding="utf-8") as f:
        json.dump(preds_all, f)
    print(f"[{model}] pipeline 完成 -> {acc_csv}, {bt_csv}, {preds_json}")


if __name__ == "__main__":
    main()