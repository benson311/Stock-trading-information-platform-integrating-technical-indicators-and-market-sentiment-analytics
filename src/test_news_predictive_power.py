"""Phase 6 關鍵檢定: 新聞情緒對次日報酬的預測力 (對照 PTT 的 p=0.27)。

檢定:
  1. 逐股 Pearson 相關: 當日情緒 net vs 次日報酬
  2. 高/低情緒日報酬差 t-test (全體)
  3. 與 PTT 基準並列比較
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import numpy as np
import pandas as pd
from scipy import stats

import config
from src.price_loader import load_best_price

NEWS_INDEX = os.path.join(config.PROCESSED_DIR, "news_sentiment_index.parquet")
PTT_INDEX = os.path.join(config.PROCESSED_DIR, "sentiment_index_review.parquet")


def test_predictive_power(index_path, label):
    idx = pd.read_parquet(index_path)
    idx["date"] = idx["date"].astype(str)
    # 欄位正規化: review_net (PTT) / net (新聞) -> net
    if "net" not in idx.columns and "review_net" in idx.columns:
        idx = idx.rename(columns={"review_net": "net"})

    cors, diffs = [], []
    for code, g in idx.groupby("code"):
        df = load_best_price(code)
        if df.empty:
            continue
        df["ret"] = df["close"].pct_change().shift(-1)
        m = df.merge(g[["date", "net"]], on="date", how="inner").dropna(
            subset=["ret", "net"])
        if len(m) < 30:
            continue
        cors.append(m["net"].corr(m["ret"]))
        q_hi, q_lo = m["net"].quantile(0.7), m["net"].quantile(0.3)
        hi_ret = m[m["net"] > q_hi]["ret"]
        lo_ret = m[m["net"] < q_lo]["ret"]
        # 樣本太少或分位數切不開 (全部同值) 則跳過該股的 hi-lo
        if len(hi_ret) >= 3 and len(lo_ret) >= 3:
            diffs.append(hi_ret.mean() - lo_ret.mean())

    cors = np.array(cors)
    diffs = np.array(diffs)
    if len(cors) == 0:
        print(f"[{label}] 無足夠樣本")
        return None
    t_c = stats.ttest_1samp(cors, 0)
    t_d = stats.ttest_1samp(diffs, 0)
    res = {
        "label": label,
        "n_stocks": len(cors),
        "corr_mean": cors.mean(),
        "corr_median": np.median(cors),
        "corr_t": t_c.statistic, "corr_p": t_c.pvalue,
        "hi_lo_diff": diffs.mean(),
        "diff_t": t_d.statistic, "diff_p": t_d.pvalue,
        "pos_corr_ratio": (cors > 0).mean(),
    }
    return res


def main():
    results = []
    r_news = test_predictive_power(NEWS_INDEX, "鉅亨+Yahoo 新聞")
    if r_news:
        results.append(r_news)
    r_ptt = test_predictive_power(PTT_INDEX, "PTT (基準)")
    if r_ptt:
        results.append(r_ptt)

    print("\n" + "=" * 68)
    print("情緒預測力比較 (當日情緒 vs 次日報酬)")
    print("=" * 68)
    df = pd.DataFrame(results)
    print(df.round(4).to_string(index=False))
    print()
    for r in results:
        sig = "顯著!" if r["diff_p"] < 0.05 else ("邊際" if r["diff_p"] < 0.1 else "不顯著")
        print(f"[{r['label']}] 高低情緒日報酬差 {r['hi_lo_diff']:+.4%}, "
              f"t={r['diff_t']:.2f}, p={r['diff_p']:.4f} -> {sig}")

    out = os.path.join(config.RESULTS_DIR, "news_vs_ptt_predictive_power.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已存 -> {out}")


if __name__ == "__main__":
    main()