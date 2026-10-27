import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""新 (29 維, 含法人/融資融券) vs 舊 (17 維) 全量結果對照摘要。

用法: python3 -m src.compare_old_new <舊結果備份目錄>
"""
import os
import sys

import pandas as pd

import config


def main():
    if len(sys.argv) < 2:
        print("用法: python3 -m src.compare_old_new <backup_dir>")
        sys.exit(1)
    bk = sys.argv[1]

    old_acc = pd.read_csv(os.path.join(bk, "accuracy_lstm_attention.csv"))
    new_acc = pd.read_csv(os.path.join(config.RESULTS_DIR, "accuracy_lstm_attention.csv"))
    print("=" * 72)
    print("302 檔全面重訓對照 (LSTM+Attention, epochs=60, 測試集 2023–2026)")
    print("=" * 72)
    rows = []
    for label, df in [("舊 17 維 (無法人)", old_acc), ("新 29 維 (含法人+融資融券)", new_acc)]:
        rows.append({
            "方案": label, "檔數": len(df),
            "MSE中位": df["MSE"].median(), "RMSE均": df["RMSE"].mean(),
            "MAE均": df["MAE"].mean(), "方向準確率均": df["direction_acc"].mean(),
        })
    acc = pd.DataFrame(rows)
    print(acc.to_string(index=False))

    # 逐股配對比較 (同 code)
    m = old_acc.merge(new_acc, on="code", suffixes=("_old", "_new"))
    better = (m["MSE_new"] < m["MSE_old"]).mean()
    d_impr = m["direction_acc_new"] - m["direction_acc_old"]
    print(f"\n逐股配對 (n={len(m)}): 新特徵 MSE 較佳比例 {better:.1%}, "
          f"方向準確率平均變化 {d_impr.mean():+.4f} "
          f"({(d_impr > 0).mean():.1%} 的股票方向準確率提升)")

    ob = pd.read_csv(os.path.join(bk, "backtest_lstm_attention.csv"))
    nb = pd.read_csv(os.path.join(config.RESULTS_DIR, "backtest_lstm_attention.csv"))
    print("\n回測 (DMA+預測 vs 純 DMA, 台灣規則 T+2/±10%/稅費)")
    for label, df in [("舊", ob), ("新", nb)]:
        win = (df["with_ret"] > df["dma_ret"]).mean()
        print(f"[{label}] 預測策略平均報酬 {df['with_ret'].mean():+.2%}, DMA {df['dma_ret'].mean():+.2%}, "
              f"預測勝過 DMA 比例 {win:.1%}, 平均 Sharpe {df['with_sharpe'].mean():.2f}")

    out = os.path.join(config.RESULTS_DIR, "old_vs_new_29feat_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(acc.to_string(index=False) + "\n")
        f.write(f"\n逐股配對 (n={len(m)}): 新特徵 MSE 較佳比例 {better:.1%}, "
                f"方向準確率平均變化 {d_impr.mean():+.4f}\n")
    print(f"\n已存 -> {out}")


if __name__ == "__main__":
    main()
