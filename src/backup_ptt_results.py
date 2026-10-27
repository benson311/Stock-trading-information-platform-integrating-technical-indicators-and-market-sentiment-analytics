"""備份 PTT 時代的舊產出 → *_ptt.*  (供最終比較用), 讓新聞版重建時不覆蓋。"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import shutil

import config

MOVES = [
    ("data/processed/features.parquet", "data/processed/features_ptt.parquet"),
    ("results/accuracy_lstm_attention.csv", "results/accuracy_lstm_attention_ptt.csv"),
    ("results/backtest_lstm_attention.csv", "results/backtest_lstm_attention_ptt.csv"),
    ("results/predictions_lstm_attention.json", "results/predictions_lstm_attention_ptt.json"),
    ("results/model_comparison.csv", "results/model_comparison_ptt.csv"),
    ("results/feature_ablation.csv", "results/feature_ablation_ptt.csv"),
    ("results/auto_trade_signals.csv", "results/auto_trade_signals_ptt.csv"),
]


def main():
    base = config.BASE_DIR
    n = 0
    for src, dst in MOVES:
        s = os.path.join(base, src)
        d = os.path.join(base, dst)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.move(s, d)
            print(f"備份 {src} -> {dst}")
            n += 1
    print(f"備份完成: {n} 項 (舊 PTT 結果保留供比較)")


if __name__ == "__main__":
    main()