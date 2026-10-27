import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""總結報告產生器: 由 results/*.csv 彙整比較與結論。"""
import os

import pandas as pd

import config


def summarize_accuracy(csv_path):
    df = pd.read_csv(csv_path)
    print("\n========== 準確度摘要 ==========")
    print(f"股票數: {df['code'].nunique()}")
    print(f"平均 MSE={df['MSE'].mean():.6f}, RMSE={df['RMSE'].mean():.6f}, "
          f"MAE={df['MAE'].mean():.6f}")
    print(f"方向準確率: 平均={df['direction_acc'].mean():.3f}, "
          f"最高={df['direction_acc'].max():.3f}, 最低={df['direction_acc'].min():.3f}")
    print("\nMSE 最低 (預測最準) 前 10:")
    print(df.nsmallest(10, "MSE")[["code", "MSE", "RMSE", "MAE", "direction_acc"]]
          .to_string(index=False))
    print("\nMSE 最高 (預測最差) 前 10:")
    print(df.nlargest(10, "MSE")[["code", "MSE", "RMSE", "MAE", "direction_acc"]]
          .to_string(index=False))


def summarize_backtest(csv_path):
    df = pd.read_csv(csv_path)
    print("\n========== 回測摘要 (DMA+預測 vs DMA) ==========")
    print(f"股票數: {df['code'].nunique()}")
    for col in ["with_ret", "dma_ret", "with_sharpe", "dma_sharpe", "with_maxdd", "dma_maxdd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"加權總報酬: 平均={df['with_ret'].mean():.3f} vs DMA={df['dma_ret'].mean():.3f}")
    print(f"Sharpe: 平均={df['with_sharpe'].mean():.3f} vs DMA={df['dma_sharpe'].mean():.3f}")
    print(f"最大回撤: 平均={df['with_maxdd'].mean():.3f} vs DMA={df['dma_maxdd'].mean():.3f}")
    better = (df["with_ret"] > df["dma_ret"]).mean()
    print(f"預測策略勝過 DMA 的比例: {better:.1%}")

    print("\n預測策略總報酬前 10:")
    print(df.nlargest(10, "with_ret")[["code", "with_ret", "with_sharpe", "dma_ret"]]
          .to_string(index=False))


def main():
    model = "lstm_attention"
    acc = os.path.join(config.RESULTS_DIR, f"accuracy_{model}.csv")
    bt = os.path.join(config.RESULTS_DIR, f"backtest_{model}.csv")
    if os.path.exists(acc):
        summarize_accuracy(acc)
    if os.path.exists(bt):
        summarize_backtest(bt)


if __name__ == "__main__":
    main()