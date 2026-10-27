"""最終綜合報告: 新聞情緒版 vs PTT 版 完整對照。"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os

import pandas as pd

import config


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def load_or_none(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    R = config.RESULTS_DIR

    # 1. 新聞語料統計
    section("1. 新聞語料與情緒指數")
    idx_path = os.path.join(config.PROCESSED_DIR, "news_sentiment_index.parquet")
    if os.path.exists(idx_path):
        idx = pd.read_parquet(idx_path)
        print(f"新聞情緒指數: {len(idx):,} 列 (個股×日), 涵蓋 {idx['code'].nunique()} 檔")
        print(f"日期範圍: {idx['date'].min()} ~ {idx['date'].max()}")
        print(f"平均個股日淨情緒: {idx['net'].mean():.4f}")
        cnt = idx.groupby("date")["news_count"].sum()
        print(f"每日新聞均量: {cnt.mean():.0f} 篇/日 (全股票合計)")

    # 2. 情緒預測力 (核心)
    section("2. 情緒預測力 (新聞 vs PTT)")
    pp = load_or_none(os.path.join(R, "news_vs_ptt_predictive_power.csv"))
    if pp is not None:
        cols = ["label", "n_stocks", "corr_mean", "hi_lo_diff", "diff_t", "diff_p",
                "pos_corr_ratio"]
        print(pp[cols].round(4).to_string(index=False))
        for _, r in pp.iterrows():
            sig = "顯著!" if r["diff_p"] < 0.05 else ("邊際" if r["diff_p"] < 0.1 else "不顯著")
            print(f"  [{r['label']}] 高低情緒日報酬差 {r['hi_lo_diff']:+.4%} "
                  f"(p={r['diff_p']:.4f}) -> {sig}")

    # 3. 預測準確度對照
    section("3. 預測準確度 (新聞版 vs PTT 版)")
    for label, fn in [("新聞版", "accuracy_lstm_attention.csv"),
                      ("PTT版", "accuracy_lstm_attention_ptt.csv")]:
        acc = load_or_none(os.path.join(R, fn))
        if acc is not None:
            print(f"[{label}] {acc['code'].nunique()} 檔, "
                  f"MSE 中位={acc['MSE'].median():.6f}, "
                  f"方向準確率={acc['direction_acc'].mean():.3f}")

    # 4. 特徵消融對照
    section("4. 特徵消融 (情緒是否有助, 新聞版)")
    fab_news = load_or_none(os.path.join(R, "feature_ablation.news.csv"))
    if fab_news is not None:
        print(fab_news.round(6).to_string(index=False))
    fab_ptt = load_or_none(os.path.join(R, "feature_ablation_ptt.csv"))
    if fab_ptt is not None:
        print("\n[對照 PTT 版]")
        print(fab_ptt.round(6).to_string(index=False))

    # 5. 模型比較
    section("5. 模型比較 (新聞版)")
    mc = load_or_none(os.path.join(R, "model_comparison.news.csv"))
    if mc is not None:
        print(mc.groupby("model")["MSE"].mean().round(6).sort_values().to_string())

    # 6. 回測對照
    section("6. 回測 (DMA+預測 vs DMA, 新聞版 vs PTT 版)")
    for label, fn in [("新聞版", "backtest_lstm_attention.csv"),
                      ("PTT版", "backtest_lstm_attention_ptt.csv")]:
        bt = load_or_none(os.path.join(R, fn))
        if bt is not None:
            for c in ["with_ret", "dma_ret", "with_sharpe", "dma_sharpe"]:
                bt[c] = pd.to_numeric(bt[c], errors="coerce")
            print(f"[{label}] 總報酬: 預測={bt['with_ret'].mean():.2%} "
                  f"vs DMA={bt['dma_ret'].mean():.2%} | "
                  f"勝率={(bt['with_ret'] > bt['dma_ret']).mean():.1%}")

    # 7. 自動交易訊號
    section("7. 自動交易訊號 (新聞版 Top 買進)")
    sig = load_or_none(os.path.join(R, "auto_trade_signals.news.csv"))
    if sig is not None:
        print("BUY/SELL/HOLD:", sig["signal"].value_counts().to_dict())
        buys = sig[sig["signal"] == "BUY"].head(10)
        print(buys[["code", "pred_return"]].to_string(index=False))


if __name__ == "__main__":
    main()