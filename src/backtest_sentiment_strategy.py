"""橫斷面情緒選股策略回測 (Long High-Sentiment / Short Low-Sentiment)。

動機: 檢定證實「高情緒日 vs 低情緒日」次日報酬差 +0.57% (p<0.0001)。
本模組把該價差直接做成策略:

  訊號: 個股 t 日新聞淨情緒 (net)
  分組: 當日有新聞的股票中, 前 30% (做多) vs 後 30% (做空), 等權
  版本 B (主, 可交易): t+1 收盤進場, t+2 收盤出場 (訊號完全已知)
  版本 A (對照, 樂觀): t 收盤進場, t+1 收盤出場 (對齊檢定 1)
  成本: 手續費 0.1425%/邊 + 賣出/回補稅 0.3% → 每腿全周轉 0.585%

輸出: 逐日績效 CSV + 摘要 (新聞版 vs PTT 版 對照)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import numpy as np
import pandas as pd

import config
from src.price_loader import load_best_price, list_all_stock_codes

FEE = 0.001425          # 手續費 (買/賣各)
TAX = 0.003             # 證券交易稅 (賣出/融券回補)
ROUND_TRIP = FEE * 2 + TAX   # 單腿全周轉 = 0.585%
TOP_Q = 0.70            # 前後 30% 分位
MIN_NAMES = 5           # 每腿最少股票數 (不足則跳過該日)


def build_stock_day_returns(index_path: str) -> pd.DataFrame:
    """將情緒指數對齊個股報酬, 產生 (code, date, net, retA, retB)。"""
    idx = pd.read_parquet(index_path)
    if "net" not in idx.columns:
        idx = idx.rename(columns={"review_net": "net"})
    idx["date"] = idx["date"].astype(str)
    idx = idx[["code", "date", "net"]].dropna(subset=["net"])

    codes = idx["code"].unique()
    rows = []
    for code in codes:
        price = load_best_price(code)
        if price.empty or len(price) < 5:
            continue
        dates = price["date"].values
        closes = price["close"].astype(float).values
        pos_map = {d: i for i, d in enumerate(dates)}

        sub = idx[idx["code"] == code]
        for _, r in sub.iterrows():
            i = pos_map.get(r["date"])
            if i is None or i + 2 >= len(closes):
                continue
            retA = closes[i + 1] / closes[i] - 1        # t收→t+1收
            retB = closes[i + 2] / closes[i + 1] - 1    # t+1收→t+2收
            rows.append((code, r["date"], r["net"], retA, retB))

    df = pd.DataFrame(rows, columns=["code", "date", "net", "retA", "retB"])
    return df


def daily_legs(sd: pd.DataFrame, variant: str) -> pd.DataFrame:
    """逐日分組: 回傳每日 做多腿/做空腿/基準(當日有訊號股等權) 報酬。"""
    ret_col = "retA" if variant == "A" else "retB"
    out = []
    for date, g in sd.groupby("date"):
        g = g.sort_values("net")
        n = len(g)
        k = max(MIN_NAMES, int(n * (1 - TOP_Q)))
        if n < MIN_NAMES * 2:
            continue
        longs = g.tail(k)          # net 最高
        shorts = g.head(k)          # net 最低
        out.append({
            "date": date,
            "n": n,
            "long_ret": longs[ret_col].mean(),
            "short_ret": shorts[ret_col].mean(),
            "bench_ret": g[ret_col].mean(),
        })
    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


def perf_stats(daily: pd.Series, cost_per_day: float = 0.0) -> dict:
    r = daily - cost_per_day
    if len(r) == 0:
        return {}
    equity = (1 + r).cumprod()
    total = equity.iloc[-1] - 1
    ann = (1 + total) ** (252 / len(r)) - 1 if len(r) > 30 else np.nan
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(252)
    peak = equity.cummax()
    mdd = ((equity - peak) / peak).min()
    return {
        "days": len(r),
        "ann_ret": ann,
        "total_ret": total,
        "sharpe": sharpe,
        "max_dd": mdd,
        "hit_rate": (r > 0).mean(),
        "avg_daily": r.mean(),
    }


def run_strategy(index_path: str, label: str) -> dict:
    print(f"\n=== {label} ===")
    sd = build_stock_day_returns(index_path)
    print(f"股票×日 樣本: {len(sd):,} ({sd['code'].nunique()} 檔, "
          f"{sd['date'].nunique()} 日)")

    results = {}
    for variant, name in [("B", "B: 可交易 (t+1進 t+2出)"),
                          ("A", "A: 樂觀 (t進 t+1出, 對齊檢定1)")]:
        daily = daily_legs(sd, variant)
        spread = daily["long_ret"] - daily["short_ret"]
        ls_net = spread / 2          # 50%資金做多 + 50%做空 (dollar-neutral)
        long_only = daily["long_ret"]

        gross = perf_stats(spread / 2, cost_per_day=0)
        net_daily = perf_stats(ls_net, cost_per_day=ROUND_TRIP)
        net_weekly = perf_stats(ls_net, cost_per_day=ROUND_TRIP / 5)
        lo = perf_stats(long_only, cost_per_day=ROUND_TRIP)
        bench = perf_stats(daily["bench_ret"], cost_per_day=0)

        results[f"{label}_{variant}"] = {
            "gross": gross, "net_daily": net_daily,
            "net_weekly": net_weekly, "long_only_net": lo,
        }
        print(f"\n  [{name}] 有效日數 {len(daily)}")
        print(f"    L/S 毛報酬 (無成本): 年化 {gross['ann_ret']:+.1%}, "
              f"Sharpe {gross['sharpe']:.2f}, 勝率 {gross['hit_rate']:.1%}, "
              f"最大回撤 {gross['max_dd']:.1%}")
        print(f"    L/S 淨值 (日再平衡, 台灣成本0.585%/日): 年化 {net_daily['ann_ret']:+.1%}, "
              f"Sharpe {net_daily['sharpe']:.2f}")
        print(f"    L/S 淨值 (週再平衡, 成本/5): 年化 {net_weekly['ann_ret']:+.1%}, "
              f"Sharpe {net_weekly['sharpe']:.2f}, "
              f"總報酬 {net_weekly['total_ret']:+.1%}, 回撤 {net_weekly['max_dd']:.1%}")
        print(f"    純做多高情緒 (含成本): 年化 {lo['ann_ret']:+.1%}, "
              f"Sharpe {lo['sharpe']:.2f}")
        print(f"    對照基準 (當日有訊號股等權): 年化 {bench['ann_ret']:+.1%}")

        # 逐日輸出存檔
        tag = "news" if "新聞" in label else "ptt"
        out = os.path.join(config.RESULTS_DIR,
                           f"sentiment_strategy_{tag}_{variant}.csv")
        daily.to_csv(out, index=False)

    return results


def main():
    news_idx = os.path.join(config.PROCESSED_DIR, "news_sentiment_index.parquet")
    ptt_idx = os.path.join(config.PROCESSED_DIR, "sentiment_index_review.parquet")

    res_news = run_strategy(news_idx, "鉅亨+Yahoo 新聞")
    res_ptt = run_strategy(ptt_idx, "PTT (基準)")

    # 對照總表
    print("\n" + "=" * 72)
    print("橫斷面情緒選股策略: 新聞 vs PTT 對照總表 (可交易版 B, 週再平衡含成本)")
    print("=" * 72)
    rows = []
    for label, res in [("新聞", res_news), ("PTT", res_ptt)]:
        s = res[f"{ {'新聞':'鉅亨+Yahoo 新聞','PTT':'PTT (基準)'}[label] }_B"]["net_weekly"]
        g = res[f"{ {'新聞':'鉅亨+Yahoo 新聞','PTT':'PTT (基準)'}[label] }_B"]["gross"]
        rows.append({
            "訊號來源": label,
            "年化(毛)": f"{g['ann_ret']:+.1%}",
            "年化(淨,週再平衡)": f"{s['ann_ret']:+.1%}",
            "Sharpe(毛)": f"{g['sharpe']:.2f}",
            "勝率": f"{g['hit_rate']:.1%}",
            "最大回撤": f"{s['max_dd']:.1%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    out = os.path.join(config.RESULTS_DIR, "sentiment_strategy_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已存 -> {out}")


if __name__ == "__main__":
    main()