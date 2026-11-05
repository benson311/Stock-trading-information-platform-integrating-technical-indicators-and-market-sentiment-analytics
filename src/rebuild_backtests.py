import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""以「漲停鎖死不可成交」規則重算回測 (免重訓: 直接用既有 predictions json)。

背景: 模型選中的股票觸及漲跌停的比例顯著高於母體 (外部研究: 8.5 倍),
用次日收盤價無條件成交會嚴重高估可實現績效。

處理:
  1. 29 維現行: results/predictions_lstm_attention.json → backtest_lstm_attention.csv
     (舊檔備份至 results/backup_pre_limit/)
  2. 17 維對照: results/backup_17feat_20261008_2213/predictions_*.json
     → backtest_lstm_attention_17feat_clipped.csv (公平 A/B)
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

import config
from src.backtest import run_backtest
from src.price_loader import load_best_price


def rebuild(json_path, out_csv, backup_dir=None):
    with open(json_path, encoding="utf-8") as f:
        preds = json.load(f)
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        if os.path.exists(out_csv):
            shutil.copy2(out_csv, backup_dir)
    rows = []
    for code, p in preds.items():
        dates = list(p["dates"])
        rets = np.asarray(p["pred_return"], dtype=float)
        price = load_best_price(code)
        if price.empty:
            continue
        close = price.set_index("date")["close"].astype(float)
        sub = close.reindex(dates)
        pr = pd.Series(rets, index=dates)
        valid = sub.notna() & pr.notna()
        sub, pr = sub[valid], pr[valid]
        if len(sub) <= 30:
            continue
        _, m_with, _ = run_backtest(sub, pr, use_prediction=True, skip_limit=True)
        _, m_dma, _ = run_backtest(sub, pr, use_prediction=False, skip_limit=True)
        rows.append({"code": code, "model": "lstm_attention",
                     "with_ret": m_with["total_return"], "with_sharpe": m_with["sharpe"],
                     "with_maxdd": m_with["max_drawdown"], "with_final": m_with["final_assets"],
                     "dma_ret": m_dma["total_return"], "dma_sharpe": m_dma["sharpe"],
                     "dma_maxdd": m_dma["max_drawdown"], "dma_final": m_dma["final_assets"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def summarize(label, df):
    win = (df["with_ret"] > df["dma_ret"]).mean()
    print(f"  [{label}] 預測均報酬 {df['with_ret'].mean():+.2%} | DMA {df['dma_ret'].mean():+.2%} | "
          f"勝過 DMA {win:.1%} | Sharpe {df['with_sharpe'].mean():.2f}")
    return {"pred": round(float(df["with_ret"].mean()), 4), "win": round(float(win), 3)}


def main():
    print("=== 漲停鎖死規則後的回測重算 ===")
    out = {}
    old29 = pd.read_csv(os.path.join(config.RESULTS_DIR, "backtest_lstm_attention.csv"))
    old17 = pd.read_csv(os.path.join(config.RESULTS_DIR, "backtest_lstm_attention_17feat_clipped.csv"))
    print("\n[29 維] 修正前 vs 修正後:")
    out["old29"] = summarize("修正前", old29)
    new29 = rebuild(os.path.join(config.RESULTS_DIR, "predictions_lstm_attention.json"),
                    os.path.join(config.RESULTS_DIR, "backtest_lstm_attention.csv"),
                    backup_dir=os.path.join(config.RESULTS_DIR, "backup_pre_limit"))
    out["new29"] = summarize("修正後", new29)
    print("\n[17 維對照] 修正前 vs 修正後:")
    out["old17"] = summarize("修正前", old17)
    new17 = rebuild(os.path.join(config.RESULTS_DIR, "backup_17feat_20261008_2213",
                                 "predictions_lstm_attention.json"),
                    os.path.join(config.RESULTS_DIR, "backtest_lstm_attention_17feat_clipped.csv"))
    out["new17"] = summarize("修正後", new17)
    print("\n=== 對照摘要 ===")
    print(f"  29 維: 修正前 {out['old29']['pred']:+.2%} → 修正後 {out['new29']['pred']:+.2%} "
          f"(漲停規則影響 {out['new29']['pred'] - out['old29']['pred']:+.2%})")
    print(f"  17 維: 修正前 {out['old17']['pred']:+.2%} → 修正後 {out['new17']['pred']:+.2%} "
          f"(影響 {out['new17']['pred'] - out['old17']['pred']:+.2%})")
    try:
        from src.experiment_log import log_trial
        log_trial("backtest", "limit_up_rule_rebuild", {"rule": "exec_chg>=9.95% skip"},
                  {"29d_old": out["old29"]["pred"], "29d_new": out["new29"]["pred"],
                   "17d_old": out["old17"]["pred"], "17d_new": out["new17"]["pred"]},
                  notes="漲停鎖死不可成交修正 (cookman123 方法論移植)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
