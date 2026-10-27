"""盤前新聞 → 開盤進場: 漂移的「隔夜 vs 盤中」解剖 + 開盤策略回測。

時點設計 (台股 09:00 開盤 / 13:30 收盤):
  交易日 t 的「盤前資訊」= 前一交易日 13:30 之後 ~ 今日 09:00 之前發布的新聞
  (含前晚盤後新聞、週末新聞、今日清晨外資晨報)
  進場: 今日 09:00 開盤價 (資訊完全已知, 零前視)

解剖 (訊號日的第一日報酬拆解):
  gap     = open[t]/close[t-1] - 1   隔夜跳空 (新聞後、開盤前, 搶不到)
  intraday = close[t]/open[t] - 1     開盤→收盤 (開盤進場搶得到!)
  CAR(1)   ≈ gap + intraday

回測:
  事件版: 營收創高 / 評等上調 (僅取「盤前窗口」觸發的文章)
  情緒版: 盤前淨情緒橫斷面多空
  成本: 0.585%/腿來回; 對照「收盤進場」版本
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import bisect
import re
import datetime as dt
import numpy as np
import pandas as pd
from scipy import stats

import config
from src.event_study import EVENT_RULES

FROZEN = os.path.join(config.PROCESSED_DIR, "cnyes_news_raw.parquet")
TIMES = os.path.join(config.PROCESSED_DIR, "cnyes_publish_times.parquet")
FEAT = os.path.join(config.PROCESSED_DIR, "features.parquet")
OUT = os.path.join(config.RESULTS_DIR, "premarket_backtest.csv")

TARGET_EVENTS = ["營收創高/成長", "目標價/評等上調"]
RT_COST = 0.00585
MIN_NAMES = 5


def load_and_map():
    """載入文章+發布時間, 映射到「盤前所屬交易日」, 並標記事件。"""
    art = pd.read_parquet(FROZEN)
    art = art[art["stock_codes"].str.len() > 0]
    times = pd.read_parquet(TIMES)
    art = art.merge(times[["aid", "publish_dt"]], on="aid", how="inner")
    art["dt"] = pd.to_datetime(art["publish_dt"])

    # 交易日曆 (由價格 panel 的日期聯集)
    df = pd.read_parquet(FEAT, columns=["date"])
    tdays = sorted(df["date"].astype(str).unique())
    t09 = [dt.datetime.strptime(d, "%Y-%m-%d") + dt.timedelta(hours=9) for d in tdays]
    t13 = [dt.datetime.strptime(d, "%Y-%m-%d") + dt.timedelta(hours=13, minutes=30)
           for d in tdays]

    def map_row(dtime):
        i = bisect.bisect_left(t09, dtime.to_pydatetime())
        if i >= len(t09):
            return None
        if i > 0 and dtime.to_pydatetime() <= t13[i - 1]:
            return None        # 盤中/更早 → 非本實驗範圍
        return tdays[i]

    art["entry_day"] = art["dt"].map(map_row)
    pre = art.dropna(subset=["entry_day"])
    n_discard = len(art) - len(pre)
    print(f"有標記文章 {len(art):,} 篇 (含時間), 其中「盤前窗口」: {len(pre):,} 篇 "
          f"(其餘 {n_discard:,} 為盤中/超範圍)")
    return pre, tdays


def score_events(pre):
    """對盤前文章做事件標記 + 情緒評分。"""
    from src.sentiment import SentimentAnalyzer
    ana = SentimentAnalyzer(mode="lexicon")
    rows = []
    for _, r in pre.iterrows():
        text = (r["title"] or "") + " " + (r["content"] or "")[:500]
        p, n = ana.score(str(text))
        cats = [c for c, _, pat in EVENT_RULES
                if c in TARGET_EVENTS and re.search(pat, str(text))]
        for code in str(r["stock_codes"]).split(","):
            code = code.strip()
            if code:
                rows.append({"code": code, "entry_day": r["entry_day"],
                             "pos": p, "neg": n,
                             "event": ";".join(cats) if cats else ""})
    sd = pd.DataFrame(rows)
    print(f"展開後 (文章×個股): {len(sd):,} 列")
    ev = sd[sd["event"] != ""]
    print(f"含目標事件: {len(ev):,}")
    return sd, ev


def build_panel():
    """開盤/收盤價矩陣 (共同交易日曆)。"""
    df = pd.read_parquet(FEAT, columns=["code", "date", "open", "close"])
    df["date"] = df["date"].astype(str)
    opens = df.pivot_table(index="date", columns="code", values="open")
    closes = df.pivot_table(index="date", columns="code", values="close")
    return opens, closes


def analyze_events(ev, opens, closes, tdays):
    """事件: 隔夜 vs 盤中 解剖 + 開盤進場多日持有。"""
    dpos = {d: i for i, d in enumerate(tdays)}
    n = len(tdays)
    o = opens.values
    c = closes.values
    cols = {cc: j for j, cc in enumerate(closes.columns)}

    # 當日全市場等權基準 (同窗口)
    def mkt_ret(i, h, use_open):
        if i < 0 or i + h >= n:
            return np.nan
        if use_open:
            r = (c[i + h] / o[i]) - 1
        else:
            r = (c[i + h] / c[i]) - 1
        return np.nanmean(np.clip(r, -0.5, 0.5))

    recs = []
    ev = ev.drop_duplicates(subset=["code", "entry_day", "event"])
    # 同股同事件 10 日內去重 (獨立事件)
    ev = ev.sort_values(["code", "event", "entry_day"])
    keep, last = [], {}
    for _, r in ev.iterrows():
        k = (r["code"], r["event"])
        if k in last and r["entry_day"] <= last[k]:
            continue
        keep.append(r)
        d0 = pd.Timestamp(r["entry_day"])
        last[k] = (d0 + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    ev = pd.DataFrame(keep)
    print(f"\n獨立盤前事件: {len(ev):,}")

    for _, r in ev.iterrows():
        i = dpos.get(r["entry_day"])
        if i is None or i < 1 or i + 6 >= n:
            continue
        j = cols.get(r["code"])
        if j is None:
            continue
        pc_prev, op, cl = c[i - 1, j], o[i, j], c[i, j]
        if np.isnan(pc_prev) or np.isnan(op) or np.isnan(cl):
            continue
        gap = op / pc_prev - 1
        intra = cl / op - 1
        # 開盤進場持有 1/5 日 (開盤價 → 未來收盤)
        h1 = c[i, j] / op - 1
        h5 = c[i + 5, j] / op - 1
        # 對照: 收盤進場 (close[i] → close[i+5])
        c5 = c[i + 5, j] / cl - 1
        # 市場調整
        recs.append({
            "event": r["event"],
            "gap": gap, "intraday": intra,
            "open_h1": h1, "open_h5": h5, "close_h5": c5,
            "open_h1_ar": h1 - mkt_ret(i, 0, True),
            "open_h5_ar": h5 - mkt_ret(i + 5 - 5, 5, True) if False else h5 - _mkt5(o, c, i, 5),
            "close_h5_ar": c5 - _mkt5(c, c, i, 5),
        })
    ed = pd.DataFrame(recs)
    print(f"有效事件: {len(ed):,}")

    print("\n=== 事件漂移解剖 (bp, 開盤進場 vs 收盤進場) ===")
    rows = []
    for ev_name, g in ed.groupby("event"):
        row = {"event": ev_name, "n": len(g)}
        for col, lab in [("gap", "隔夜跳空(搶不到)"), ("intraday", "盤中開→收(搶得到)"),
                         ("open_h1", "開盤進場持1日"), ("open_h5", "開盤進場持5日"),
                         ("open_h5_ar", "開盤持5日(市場調整)"),
                         ("close_h5", "收盤進場持5日"), ("close_h5_ar", "收盤持5日(市場調整)")]:
            v = g[col].dropna()
            v = v.clip(-0.5, 0.5)
            t = stats.ttest_1samp(v, 0).statistic if len(v) > 20 else np.nan
            row[lab] = f"{v.mean()*1e4:+.1f} (t={t:.2f})" if len(v) > 20 else "—"
        net5 = (g["open_h5"].clip(-0.5, 0.5).mean() - RT_COST)
        row["開盤持5日淨值"] = f"{net5*1e4:+.1f}bp"
        rows.append(row)
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    return ed


def _mkt5(o, c, i, h):
    """全市場等權: 開盤[i] → 收盤[i+h]。"""
    n = len(c)
    if i < 0 or i + h >= n:
        return np.nan
    r = (c[i + h] / o[i]) - 1
    return np.nanmean(np.clip(r, -0.5, 0.5))


def sentiment_ls(sd, opens, closes, tdays):
    """盤前淨情緒橫斷面多空: 開盤進場 (H=0 當日, H=5) vs 收盤進場。"""
    agg = (sd.groupby(["code", "entry_day"])
           .agg(pos=("pos", "sum"), neg=("neg", "sum"), n=("pos", "count"))
           .reset_index())
    agg["net"] = (agg["pos"] - agg["neg"]) / (agg["pos"] + agg["neg"] + 1e-6)

    dpos = {d: i for i, d in enumerate(tdays)}
    n = len(tdays)
    o, c = opens.values, closes.values
    cols = {cc: j for j, cc in enumerate(closes.columns)}

    spreads_h0, spreads_h5, spreads_c5 = [], [], []
    for day, g in agg.groupby("entry_day"):
        i = dpos.get(day)
        if i is None or i + 5 >= n or i < 1:
            continue
        g = g.sort_values("net")
        k = max(MIN_NAMES, int(len(g) * 0.3))
        if len(g) < MIN_NAMES * 2:
            continue
        hi, lo = [], []
        for _, r in g.tail(k).iterrows():
            j = cols.get(r["code"])
            if j is None or np.isnan(o[i, j]):
                continue
            hi.append(c[i, j] / o[i, j] - 1)          # 開盤→當日收盤
        for _, r in g.head(k).iterrows():
            j = cols.get(r["code"])
            if j is None or np.isnan(o[i, j]):
                continue
            lo.append(c[i, j] / o[i, j] - 1)
        if len(hi) >= 3 and len(lo) >= 3:
            spreads_h0.append(np.mean(hi) - np.mean(lo))
        # H=5 開盤 / 收盤 進場
        hi5, lo5, hc5, lc5 = [], [], [], []
        for _, r in pd.concat([g.tail(k), g.head(k)]).iterrows():
            j = cols.get(r["code"])
            if j is None or np.isnan(o[i, j]) or np.isnan(c[i, j]):
                continue
            r5o = c[i + 5, j] / o[i, j] - 1
            r5c = c[i + 5, j] / c[i, j] - 1
            r5o, r5c = np.clip(r5o, -0.5, 0.5), np.clip(r5c, -0.5, 0.5)
            if r["net"] >= g["net"].quantile(0.7):
                hi5.append(r5o); hc5.append(r5c)
            else:
                lo5.append(r5o); lc5.append(r5c)
        if len(hi5) >= 3 and len(lo5) >= 3:
            spreads_h5.append(np.mean(hi5) - np.mean(lo5))
            spreads_c5.append(np.mean(hc5) - np.mean(lc5))

    print("\n=== 盤前情緒橫斷面多空 ===")
    for label, arr, cost in [("開盤→當日收盤 (H=0)", spreads_h0, RT_COST),
                             ("開盤進場持5日", spreads_h5, RT_COST),
                             ("收盤進場持5日 (對照)", spreads_c5, RT_COST)]:
        if not arr:
            continue
        s = np.array(arr)
        t = stats.ttest_1samp(s, 0)
        print(f"  [{label}] {len(s)} 日 | 價差 {s.mean()*1e4:+.1f}bp "
              f"(t={t.statistic:.2f}, p={t.pvalue:.3f}) | "
              f"淨值(扣58.5bp) {(s.mean()-cost)*1e4:+.1f}bp")


def main():
    print("=== 盤前新聞 → 開盤進場 實驗 ===\n")
    pre, tdays = load_and_map()
    sd, ev = score_events(pre)
    opens, closes = build_panel()
    ed = analyze_events(ev, opens, closes, tdays)
    sentiment_ls(sd, opens, closes, tdays)
    ed.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n事件明細已存 -> {OUT}")


if __name__ == "__main__":
    main()