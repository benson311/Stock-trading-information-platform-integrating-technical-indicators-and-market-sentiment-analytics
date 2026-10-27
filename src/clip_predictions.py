import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""對既有 predictions json 做 ±10% 漲跌停裁切, 並重算 accuracy / backtest (無需重訓)。

模型權重不變, 純後處理:
  1. pred_return 裁切至 [-10%, +10%]; pred_price 等比調整 (close_t 未存檔, 用比例還原)
  2. accuracy: y_true 由 features.parquet 依「同框架」重建 (與訓練時完全一致的日序)
  3. backtest: 以裁切後報酬重跑台灣規則回測 (DMA+預測 vs DMA)

內建驗證: 裁切不改變正負號 → 重算之方向準確率必須與原 accuracy CSV 完全一致,
不一致即中止 (代表 y_true 重建有誤), 不會寫出任何檔案。

用法:
  # 主結果 (29 維) 裁切, 原檔備份:
  python3 -m src.clip_predictions --backup-dir results/backup_unclipped_29feat
  # 舊 17 維備份做同樣裁切以便公平 A/B (輸出新檔名, 不動備份):
  python3 -m src.clip_predictions \
      --json results/backup_17feat_20261008_2213/predictions_lstm_attention.json \
      --accuracy results/accuracy_lstm_attention_17feat_clipped.csv \
      --backtest results/backtest_lstm_attention_17feat_clipped.csv \
      --no-backup --keep-json
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

import config
from src.price_loader import load_best_price
from src.train import evaluate
from src.backtest import run_backtest


def build_true_returns():
    """由 features.parquet 重建 (code, date) → 次日報酬 (目標日當天報酬)。

    與 make_sequences_for_stock 同框架: ret[d] = close[d]/close[框架內前一列] - 1。
    """
    df = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "features.parquet"))
    df["date"] = df["date"].astype(str)
    df = df.sort_values(["code", "date"])
    df["ret"] = df.groupby("code")["close"].pct_change()
    return df.set_index(["code", "date"])["ret"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(config.RESULTS_DIR, "predictions_lstm_attention.json"))
    ap.add_argument("--accuracy", default=os.path.join(config.RESULTS_DIR, "accuracy_lstm_attention.csv"))
    ap.add_argument("--backtest", default=os.path.join(config.RESULTS_DIR, "backtest_lstm_attention.csv"))
    ap.add_argument("--backup-dir", default=None, help="備份原 json/accuracy/backtest 至此目錄")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--keep-json", action="store_true", help="不覆寫 json (用於處理備份檔)")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        preds = json.load(f)
    print(f"載入 {args.json}: {len(preds)} 檔")

    # 原 accuracy 供驗證方向準確率
    if os.path.exists(args.accuracy):
        old_acc = pd.read_csv(args.accuracy, dtype={"code": str}).set_index("code")
    else:
        old_acc = None

    true_ret = build_true_returns()
    lim = config.PRICE_LIMIT

    acc_rows, bt_rows, new_preds, n_clipped_pts = [], [], {}, 0
    for code, p in preds.items():
        dates = list(p["dates"])
        old_ret = np.asarray(p["pred_return"], dtype=float)
        old_px = np.asarray(p["pred_price"], dtype=float)
        new_ret = np.clip(old_ret, -lim, lim)
        n_clipped_pts += int((np.abs(old_ret) > lim + 1e-12).sum())
        # close_t = old_px/(1+old_ret) 未存 → 等比調整 new_px = old_px * (1+new)/(1+old)
        denom = 1.0 + old_ret
        scale = np.where(np.abs(denom) > 1e-9, (1.0 + new_ret) / np.where(np.abs(denom) > 1e-9, denom, 1.0), 1.0)
        new_px = old_px * scale

        # y_true (與訓練同框架)
        try:
            y_true = np.array([true_ret.loc[(code, d)] for d in dates], dtype=float)
        except KeyError as e:
            print(f"  ! {code} 日期缺漏 ({e}), 跳過")
            continue
        ok = ~np.isnan(y_true)
        yt, yp = y_true[ok], new_ret[ok]
        met = evaluate(yt, yp)
        d_acc = float(np.mean((yp > 0) == (yt > 0)))

        # 驗證: 方向準確率應與原結果完全一致 (裁切不改號)
        if old_acc is not None and code in old_acc.index:
            if abs(old_acc.loc[code, "direction_acc"] - d_acc) > 1e-9:
                raise AssertionError(
                    f"{code} 方向準確率不一致: 原 {old_acc.loc[code, 'direction_acc']} vs 重算 {d_acc} "
                    "(y_true 重建有誤, 已中止, 未寫出任何檔案)")
        acc_rows.append({"code": code, "model": "lstm_attention",
                         "MSE": met["MSE"], "RMSE": met["RMSE"], "MAE": met["MAE"],
                         "direction_acc": d_acc})

        # 回測 (與 run_one 相同流程)
        price = load_best_price(code)
        if not price.empty:
            close = price.set_index("date")["close"].astype(float)
            sub = close.reindex(dates)
            pr = pd.Series(new_ret, index=dates)
            valid = sub.notna() & pr.notna()
            sub, pr = sub[valid], pr[valid]
            if len(sub) > 30:
                _, m_with, _ = run_backtest(sub, pr, use_prediction=True)
                _, m_dma, _ = run_backtest(sub, pr, use_prediction=False)
                bt_rows.append({"code": code, "model": "lstm_attention",
                                "with_ret": m_with["total_return"], "with_sharpe": m_with["sharpe"],
                                "with_maxdd": m_with["max_drawdown"], "with_final": m_with["final_assets"],
                                "dma_ret": m_dma["total_return"], "dma_sharpe": m_dma["sharpe"],
                                "dma_maxdd": m_dma["max_drawdown"], "dma_final": m_dma["final_assets"]})

        new_preds[code] = {"dates": dates,
                           "pred_return": list(new_ret),
                           "pred_price": list(new_px)}

    # 備份 → 寫出
    if args.backup_dir and not args.no_backup:
        os.makedirs(args.backup_dir, exist_ok=True)
        for f in (args.json, args.accuracy, args.backtest):
            if os.path.exists(f):
                shutil.copy2(f, args.backup_dir)
                print(f"備份 {os.path.basename(f)} -> {args.backup_dir}/")

    acc_df = pd.DataFrame(acc_rows)
    bt_df = pd.DataFrame(bt_rows)
    acc_df.to_csv(args.accuracy, index=False)
    bt_df.to_csv(args.backtest, index=False)
    if not args.keep_json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(new_preds, f)

    clipped_stocks = sum(1 for p in preds.values()
                          if any(abs(r) > lim + 1e-12 for r in p["pred_return"]))
    print(f"\n裁切點共 {n_clipped_pts} 個, 涉及 {clipped_stocks} 檔")
    print(f"accuracy -> {args.accuracy}: {len(acc_df)} 檔, MSE 中位 {acc_df['MSE'].median():.6f}, "
          f"方向準確率均 {acc_df['direction_acc'].mean():.4f} (與原值一致 ✓)")
    print(f"backtest -> {args.backtest}: {len(bt_df)} 檔, 預測均報酬 {bt_df['with_ret'].mean():+.2%}, "
          f"DMA {bt_df['dma_ret'].mean():+.2%}, 勝過 DMA {(bt_df['with_ret']>bt_df['dma_ret']).mean():.1%}")


if __name__ == "__main__":
    main()
