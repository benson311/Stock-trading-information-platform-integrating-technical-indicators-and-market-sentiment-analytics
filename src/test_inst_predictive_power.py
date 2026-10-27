import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""法人買賣超/融資融券特徵對次日報酬的預測力檢定 (比照新聞/PTT 檢定方法論)。

檢定 (逐特徵):
  1. 逐股 Pearson 相關: 當日特徵值 vs 次日報酬 → 全體 t 檢定
  2. 高/低特徵日 (70%/30% 分位) 報酬差 → 全體 t 檢定
對照: 新聞情緒 (corr=0.042, p<0.00001) 與 PTT (corr=0.002, p=0.56)。
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import config

FEATURES = os.path.join(config.PROCESSED_DIR, "features.parquet")
TEST_COLS = [
    "foreign_net_vol", "trust_net_vol", "dealer_net_vol", "total_net_vol",
    "foreign_net_z", "trust_net_z", "dealer_net_z", "total_net_z",
    "margin_chg", "margin_z", "short_chg", "short_z",
]


def main():
    df = pd.read_parquet(FEATURES)
    df["date"] = df["date"].astype(str)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["ret"] = df.groupby("code")["close"].pct_change().shift(-1)
    df = df.dropna(subset=["ret"])

    results = []
    for col in TEST_COLS:
        if col not in df.columns:
            print(f"! 特徵 {col} 不在 features.parquet, 跳過")
            continue
        cors, diffs = [], []
        for code, g in df.groupby("code"):
            g = g[g[col] != 0]  # 0 = 無資料 (2011-14 法人 / 非信用交易股), 排除
            if len(g) < 60:
                continue
            c = g[col].corr(g["ret"])
            if pd.notna(c):
                cors.append(c)
            q_hi, q_lo = g[col].quantile(0.7), g[col].quantile(0.3)
            hi, lo = g[g[col] > q_hi]["ret"], g[g[col] < q_lo]["ret"]
            if len(hi) >= 3 and len(lo) >= 3:
                diffs.append(hi.mean() - lo.mean())
        if not cors:
            continue
        cors, diffs = np.array(cors), np.array(diffs)
        t_c = stats.ttest_1samp(cors, 0)
        t_d = stats.ttest_1samp(diffs, 0) if len(diffs) else (np.nan, np.nan)
        results.append({
            "feature": col, "n_stocks": len(cors),
            "corr_mean": cors.mean(), "corr_median": np.median(cors),
            "corr_t": t_c.statistic, "corr_p": t_c.pvalue,
            "pos_corr_ratio": (cors > 0).mean(),
            "hi_lo_diff": diffs.mean() if len(diffs) else np.nan,
            "diff_t": t_d[0], "diff_p": t_d[1],
        })

    res = pd.DataFrame(results)
    print("\n法人/融資融券特徵預測力 (當日特徵 vs 次日報酬)")
    print("=" * 100)
    print(res.round(4).to_string(index=False))
    print()
    for _, r in res.iterrows():
        sig = "顯著!" if r["diff_p"] < 0.05 else ("邊際" if r["diff_p"] < 0.1 else "不顯著")
        print(f"[{r['feature']}] corr={r['corr_mean']:+.4f} (p={r['corr_p']:.4f}), "
              f"高低差 {r['hi_lo_diff']:+.4%} (p={r['diff_p']:.4f}) -> {sig}")

    out = os.path.join(config.RESULTS_DIR, "inst_margin_predictive_power.csv")
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已存 -> {out}")


if __name__ == "__main__":
    main()
