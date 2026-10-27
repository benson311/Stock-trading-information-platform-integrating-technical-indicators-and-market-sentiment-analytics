"""A′ 盤中版情緒策略: 時間感知指數 + 回測對照 (A′ vs A vs B)。

流程:
  1. 合併「有標記文章」與「發布時間戳」(backfill_cnyes_times 產出)
  2. 詞典評分 → 展開個股 → 依台灣時間分 盤中(≤13:30)/盤後(>13:30)
  3. 聚合 (code, date): net_pre (盤中新聞) / net_all (全部)
  4. 回測三版本:
       A′ : 盤中訊號, t 收盤進 → t+1 收盤出 (零前視, 可執行)
       A  : 全部新聞, t 收盤進 → t+1 出 (含盤後前視, 對齊檢定1)
       B  : 全部新聞, t+1 收盤進 → t+2 出 (保守)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys
import numpy as np
import pandas as pd

import config
from src.price_loader import load_best_price
from src.sentiment import SentimentAnalyzer
from src.backtest_sentiment_strategy import daily_legs, perf_stats, ROUND_TRIP

FROZEN = os.path.join(config.PROCESSED_DIR, "cnyes_news_frozen.parquet")
TIMES = os.path.join(config.PROCESSED_DIR, "cnyes_publish_times.parquet")
OUT_IDX = os.path.join(config.PROCESSED_DIR, "timed_sentiment_index.parquet")

MIN_PRE_SCORE = 1   # A′ 訊號至少需要 1 篇盤中新聞


def build_timed_index():
    frozen = pd.read_parquet(FROZEN)
    tagged = frozen[frozen["stock_codes"].str.len() > 0].drop(columns=["date"])
    times = pd.read_parquet(TIMES)
    df = tagged.merge(times[["aid", "date", "pre_close"]], on="aid", how="inner")
    print(f"有標記且有時間戳: {len(df):,} / {len(tagged):,} 篇")
    print(f"盤中(≤13:30)佔比: {df['pre_close'].mean():.1%}")

    # 評分
    analyzer = SentimentAnalyzer(mode="lexicon")
    texts = df["title"].fillna("") + " " + df["content"].fillna("")
    pos, neg = [], []
    for t in texts:
        p, n = analyzer.score(str(t))
        pos.append(p)
        neg.append(n)
    df["pos"], df["neg"] = pos, neg

    # 展開個股
    rows = []
    for _, r in df.iterrows():
        for code in str(r["stock_codes"]).split(","):
            if code.strip():
                rows.append((code.strip(), r["date"], r["pre_close"],
                             r["pos"], r["neg"]))
    ex = pd.DataFrame(rows, columns=["code", "date", "pre_close", "pos", "neg"])

    def agg(sub, suffix):
        g = (sub.groupby(["code", "date"], as_index=False)
             .agg(**{f"pos_{suffix}": ("pos", "sum"), f"neg_{suffix}": ("neg", "sum"),
                     f"n_{suffix}": ("pos", "count")}))
        g[f"net_{suffix}"] = (g[f"pos_{suffix}"] - g[f"neg_{suffix}"]) / (
            g[f"pos_{suffix}"] + g[f"neg_{suffix}"] + 1e-6)
        return g

    pre = agg(ex[ex["pre_close"]], "pre")
    all_ = agg(ex, "all")

    idx = all_.merge(pre, on=["code", "date"], how="left")
    for c in ["net_pre", "pos_pre", "neg_pre", "n_pre"]:
        idx[c] = idx[c].fillna(0)
    idx = idx[["code", "date", "net_pre", "net_all", "n_pre", "n_all"]]
    idx.to_parquet(OUT_IDX, index=False)
    print(f"時間感知情緒指數: {len(idx):,} 列 (個股×日) -> {OUT_IDX}")
    return idx


def build_stock_days(idx, sig_col, min_n=0):
    """訊號 → 股票×日 (retA = t收→t+1收, retB = t+1收→t+2收)。"""
    sub = idx[(idx[sig_col].notna()) & (idx[sig_col] != 0) & (idx.get("n_pre", 1) >= min_n)] \
        if min_n else idx[idx[sig_col].notna()]
    rows = []
    for code, g in sub.groupby("code"):
        price = load_best_price(code)
        if price.empty or len(price) < 5:
            continue
        dates = price["date"].values
        closes = price["close"].astype(float).values
        pmap = {d: i for i, d in enumerate(dates)}
        for _, r in g.iterrows():
            i = pmap.get(r["date"])
            if i is None or i + 2 >= len(closes):
                continue
            rows.append((code, r["date"], r[sig_col],
                         closes[i + 1] / closes[i] - 1,
                         closes[i + 2] / closes[i + 1] - 1))
    return pd.DataFrame(rows, columns=["code", "date", "net", "retA", "retB"])


def run_variant(sd, variant, label):
    daily = daily_legs(sd, variant)
    if daily.empty:
        print(f"  [{label}] 無有效日")
        return None
    spread = daily["long_ret"] - daily["short_ret"]
    gross = perf_stats(spread / 2)
    net_w = perf_stats(spread / 2, cost_per_day=ROUND_TRIP / 5)
    bench = perf_stats(daily["bench_ret"])
    print(f"  [{label}] 有效日 {len(daily)}")
    print(f"    年化(毛): {gross['ann_ret']:+.1%} | Sharpe {gross['sharpe']:.2f} | "
          f"勝率 {gross['hit_rate']:.1%} | 回撤 {gross['max_dd']:.1%} | "
          f"淨值(週再平衡): {net_w['ann_ret']:+.1%} | 基準 {bench['ann_ret']:+.1%}")
    return gross


def main():
    print("=== 步驟 1: 時間感知情緒指數 ===")
    idx = build_timed_index()

    print("\n=== 步驟 2: 三版本回測 ===")
    # A′: 盤中訊號, A 時序
    sd_pre = build_stock_days(idx, "net_pre")
    print(f"\nA′ 樣本 (有盤中訊號的 股票×日): {len(sd_pre):,}")
    g_aprime = run_variant(sd_pre, "A", "A′ 盤中訊號, t收盤進 (零前視)")

    # A: 全部新聞, A 時序
    sd_all = build_stock_days(idx, "net_all")
    print(f"\nA 樣本: {len(sd_all):,}")
    g_a = run_variant(sd_all, "A", "A  全部新聞, t收盤進 (含盤後前視)")

    # B: 全部新聞, B 時序
    g_b = run_variant(sd_all, "B", "B  全部新聞, t+1收盤進 (保守)")

    print("\n" + "=" * 70)
    print("結論對照 (年化毛報酬 / Sharpe)")
    print("=" * 70)
    for lab, g in [("A′ 盤中(可執行)", g_aprime), ("A 全部(前視上限)", g_a),
                   ("B 延遲一天", g_b)]:
        if g:
            print(f"  {lab:<18} 年化 {g['ann_ret']:+7.1%} | Sharpe {g['sharpe']:+.2f}")
    print("\nA′ vs A 差距 = 盤後新聞的前視貢獻; A′ 本身 = 真實可執行的 alpha")


if __name__ == "__main__":
    main()