"""最終交付: 事件組合回測 (營收創高 + 評等上調) — 完全可交易版本。

關鍵設計 (整合順位1的教訓 — 發布時間決定可交易性):
  觸發文章發布時間 ≤13:30 (盤中)  → 當日收盤進場, 賺 rets[i+1..i+H]
  觸發文章發布時間 >13:30 (盤後)  → 次日收盤進場, 賺 rets[i+2..i+1+H]  (零前視)

組合規則:
  - 事件: 營收創高/成長、目標價/評等上調 (成本可行性篩選後)
  - 持有 H=10 個交易日, 等權, 同股不重複進場 (有倉位時忽略新事件)
  - 成本: 買 0.1425% + 賣 0.1425% + 證交稅 0.3% = 0.585%/來回
  - 三版本對照: 交易日誌(主) / 全部當日收盤(前視上界) / 全部次日收盤(保守)

輸出: 每日淨值 CSV + 績效摘要 + 年度分解
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import re
import numpy as np
import pandas as pd

import config
from src.event_study import EVENT_RULES, build_market_returns, HORIZONS

FROZEN = os.path.join(config.PROCESSED_DIR, "cnyes_news_raw.parquet")
TIMES = os.path.join(config.PROCESSED_DIR, "cnyes_publish_times.parquet")
FEAT = os.path.join(config.PROCESSED_DIR, "features.parquet")
OUT = os.path.join(config.RESULTS_DIR, "event_portfolio_backtest.csv")

TARGET_EVENTS = ["營收創高/成長", "目標價/評等上調"]
H = 10
FEE_BUY, FEE_SELL, TAX = 0.001425, 0.001425, 0.003
CLOSE_T = 13 * 60 + 30  # 13:30


def extract_events_with_time():
    """抽出目標事件, 保留 aid 以便查發布時間; 回傳 (code, date, event, first_min)。"""
    df = pd.read_parquet(FROZEN)
    df = df[df["stock_codes"].str.len() > 0]
    times = pd.read_parquet(TIMES)
    tmap = dict(zip(times["aid"], times["publish_dt"]))

    recs = {}
    for _, r in df.iterrows():
        text = (r["title"] or "") + " " + (r["content"] or "")[:500]
        for cat, direction, pat in EVENT_RULES:
            if cat not in TARGET_EVENTS:
                continue
            if re.search(pat, text):
                pdt = tmap.get(int(r["aid"]), "9999-01-01 23:59")
                hh, mm = int(pdt[11:13]), int(pdt[14:16])
                minutes = hh * 60 + mm
                for code in str(r["stock_codes"]).split(","):
                    code = code.strip()
                    if not code:
                        continue
                    key = (code, r["date"], cat)
                    if key not in recs or minutes < recs[key]:
                        recs[key] = minutes

    ev = pd.DataFrame(
        [(c, d, cat, m) for (c, d, cat), m in recs.items()],
        columns=["code", "date", "event", "first_min"])
    ev = ev.sort_values(["code", "event", "date"]).reset_index(drop=True)

    # 獨立事件: 同股同類 10 日內去重 (保留最早觸發)
    keep, last = [], {}
    for _, r in ev.iterrows():
        key = (r["code"], r["event"])
        if key in last and r["date"] <= last[key]:
            continue
        keep.append(r)
        last[key] = (pd.Timestamp(r["date"]) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    ev = pd.DataFrame(keep).reset_index(drop=True)
    print(f"目標事件 (獨立): {len(ev):,}")
    print(ev["event"].value_counts().to_string())
    pre = (ev["first_min"] <= CLOSE_T)
    print(f"\n盤中觸發 (≤13:30, 可當日收盤進場): {pre.mean():.1%}")
    return ev


def build_positions(ev, dates, dpos, rets, timing="log"):
    """建倉位表: (code, entry_idx, exit_idx) 依 timing 規則。"""
    positions = []
    n_skipped = 0
    for _, r in ev.iterrows():
        i = dpos.get(r["date"])
        if i is None:
            n_skipped += 1
            continue
        code = r["code"]
        if code not in rets.columns:
            n_skipped += 1
            continue
        pre_close = r["first_min"] <= CLOSE_T
        if timing == "log":
            e = i if pre_close else i + 1      # 交易日誌規則
        elif timing == "opt":                   # 全部當日收盤 (前視上界)
            e = i
        else:                                   # 全部次日收盤 (保守)
            e = i + 1
        x = e + H
        if e + 1 >= len(dates) or x > len(dates):
            n_skipped += 1
            continue
        positions.append({"code": code, "e": e, "x": min(x, len(dates) - 1),
                          "event": r["event"]})
    pos = pd.DataFrame(positions)
    # 同股不重複進場: 有活躍倉位時略過新事件
    pos = pos.sort_values("e").reset_index(drop=True)
    keep = []
    active_until = {}
    for _, p in pos.iterrows():
        until = active_until.get(p["code"], -1)
        if p["e"] <= until:
            n_skipped += 1
            continue
        keep.append(p)
        active_until[p["code"]] = p["x"]
    pos = pd.DataFrame(keep).reset_index(drop=True)
    print(f"倉位數: {len(pos)} (無法執行/同股重複略過: {n_skipped})")
    return pos


def simulate(pos, rets, dates):
    """逐日組合模擬 (等權活躍倉位; 進場日扣買入手續費, 出場日扣賣出+稅)。"""
    n_days = len(dates)
    port = np.zeros(n_days)
    counts = np.zeros(n_days)
    rets_v = rets.to_numpy()
    cols = {c: j for j, c in enumerate(rets.columns)}

    for _, p in pos.iterrows():
        j = cols[p["code"]]
        for d in range(p["e"] + 1, p["x"] + 1):
            r = rets_v[d, j]
            if np.isnan(r):
                r = 0.0
            if d == p["e"] + 1:
                r -= FEE_BUY
            if d == p["x"]:
                r -= FEE_SELL + TAX
            port[d] += r
            counts[d] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        port = np.where(counts > 0, port / np.maximum(counts, 1), 0.0)
    return port, counts


def perf(port, dates, market, label):
    eq = np.cumprod(1 + port)
    total = eq[-1] - 1
    years = len(port) / 252
    cagr = (1 + total) ** (1 / years) - 1
    sharpe = port.mean() / (port.std() + 1e-12) * np.sqrt(252)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    mkt = np.nansum(market)
    print(f"\n[{label}]")
    print(f"  總報酬 {total:+.1%} | 年化 {cagr:+.1%} | Sharpe {sharpe:.2f} | "
          f"最大回撤 {mdd:.1%} | 同期市場等權 {mkt:+.1%}")
    return eq


def yearly(port, dates, market):
    df = pd.DataFrame({"date": dates, "ret": port, "mkt": market})
    df["year"] = df["date"].str[:4]
    g = df.groupby("year").agg(
        策略年化=("ret", lambda s: (1 + s).prod() - 1),
        市場年化=("mkt", lambda s: (1 + s.fillna(0)).prod() - 1))
    g["超額"] = g["策略年化"] - g["市場年化"]
    return g


def main():
    print("=== 步驟 1: 抽取事件 (含發布時間) ===")
    ev = extract_events_with_time()

    print("\n=== 步驟 2: 市場報酬矩陣 ===")
    rets, market = build_market_returns()
    dates = rets.index.values
    dpos = {d: i for i, d in enumerate(dates)}
    market_v = market.values

    # 三版本
    for timing, label in [("opt", "上界: 全部當日收盤 (含前視)"),
                          ("cons", "保守: 全部次日收盤"),
                          ("log", "主版本: 交易日誌 (盤中→當日, 盤後→次日)")]:
        print(f"\n=== 步驟 3: 倉位建置 [{label}] ===")
        pos = build_positions(ev, dates, dpos, rets, timing=timing)
        if len(pos) == 0:
            print("  無倉位")
            continue
        port, counts = simulate(pos, rets, dates)
        eq = perf(port, dates, market_v, label)
        act = counts[counts > 0]
        print(f"  平均同時持倉數: {act.mean():.1f} | 有倉位日佔比: {(counts>0).mean():.0%}")
        if timing == "log":
            g = yearly(port, dates, market_v)
            print("\n  年度分解 (主版本):")
            print((g * 1).round(3).to_string())
            out = pd.DataFrame({"date": dates, "strategy_ret": port,
                                "equity": eq, "n_positions": counts})
            out.to_csv(OUT, index=False, encoding="utf-8-sig")
            print(f"\n每日明細已存 -> {OUT}")

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()