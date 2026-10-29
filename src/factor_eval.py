import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""因子評估: Rank IC + 五分位組合 (吃 29 維特徵表, 取代「只看 MSE」的爭論)。

每個特徵 (橫斷面, 逐日):
  - Rank IC      : 該日特徵排序 vs 次日報酬排序的 Spearman 相關
                   → 全樣本 mean / t 值 / ICIR / IC>0 比例
  - 產業中性 IC  : 特徵與報酬先對 (日期, 產業) 去均值再算 IC (剔除產業共同波動)
  - 五分位價差   : 每日按特徵分 5 組等權, Q5(高) − Q1(低) 的次日報酬價差
                   → 日均 / 年化 / t 檢定

資料原則: INST 類特徵的 0 代表「無資料」(2015 前/未覆蓋) → 該日剔除該股 (每日 ≥15 檔才計算)。
輸出: results/factor_eval.csv (+ 寫入 trials.jsonl)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.build_features import INST_COLS
from src.dataset import FEATURE_COLS

FEAT = os.path.join(config.PROCESSED_DIR, "features.parquet")
OUT = os.path.join(config.RESULTS_DIR, "factor_eval.csv")
MIN_STOCKS = 15          # 每日橫斷面最少檔數
MIN_INST_COL = INST_COLS  # 0 = 無資料語意的特徵 (逐日剔除 0)


def load_panel():
    df = pd.read_parquet(FEAT)
    df["date"] = df["date"].astype(str)
    meta = json.load(open(os.path.join(config.PROCESSED_DIR, "stock_metadata.json"),
                          encoding="utf-8"))
    df["industry"] = df["code"].map(lambda c: meta.get(c, {}).get("industry", "00"))
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["ret"] = df.groupby("code")["close"].pct_change().shift(-1)
    return df.dropna(subset=["ret"])


def daily_ic(sub, col, neutral=False):
    """逐日 Rank IC (向量化): 回傳以 date 為 index 的 IC 序列。"""
    x, y = col, "ret"
    if neutral:
        g = sub.groupby(["date", "industry"])[[x, y]].transform("mean")
        sub = sub.assign(**{x: sub[x] - g[x], y: sub[y] - g[y]})
    sub = sub.assign(
        rf=sub.groupby("date")[x].rank(),
        rr=sub.groupby("date")[y].rank(),
    )
    sub = sub.assign(xy=sub["rf"] * sub["rr"],
                     xx=sub["rf"] ** 2, yy=sub["rr"] ** 2)
    g = sub.groupby("date")[["rf", "rr", "xy", "xx", "yy"]].mean()
    g = g.rename(columns={"rf": "mx", "rr": "my", "xy": "mxy", "xx": "mxx", "yy": "myy"})
    g["n"] = sub.groupby("date").size()
    g = g[g["n"] >= MIN_STOCKS]
    sd = np.sqrt(np.maximum(g["mxx"] - g["mx"] ** 2, 1e-12) *
                 np.maximum(g["myy"] - g["my"] ** 2, 1e-12))
    g["ic"] = (g["mxy"] - g["mx"] * g["my"]) / sd
    return g["ic"].dropna()


def daily_quintile_spread(sub, col):
    """逐日 Q5−Q1 等權價差序列。"""
    q = sub.groupby("date")[col].rank(pct=True)
    bucket = np.minimum(5, np.maximum(1, np.ceil(q * 5).astype(int)))
    sub = sub.assign(bucket=bucket)
    m = sub.groupby(["date", "bucket"])["ret"].mean().unstack()
    m = m.reindex(columns=[1, 2, 3, 4, 5])
    m = m.dropna(subset=[1, 5])
    cnt = sub.groupby("date").size()
    m = m[cnt >= MIN_STOCKS]
    return (m[5] - m[1]).dropna()


def eval_feature(df, col):
    sub = df[["date", "code", "industry", col, "ret"]].copy()
    if col in MIN_INST_COL:
        sub = sub[sub[col] != 0]
    if sub.empty:
        return None
    ic = daily_ic(sub, col)
    ic_neu = daily_ic(sub, col, neutral=True)
    spread = daily_quintile_spread(sub, col)
    t_ic = stats.ttest_1samp(ic, 0) if len(ic) > 5 else (np.nan, np.nan)
    t_n = stats.ttest_1samp(ic_neu, 0) if len(ic_neu) > 5 else (np.nan, np.nan)
    t_s = stats.ttest_1samp(spread, 0) if len(spread) > 5 else (np.nan, np.nan)
    return {
        "feature": col,
        "n_days": len(ic),
        "stocks_med": int(sub.groupby("date").size().median()),
        "ic_mean": ic.mean(), "ic_t": t_ic[0], "ic_ir": ic.mean() / (ic.std() + 1e-12),
        "ic_pos": (ic > 0).mean(),
        "ic_neu_mean": ic_neu.mean(), "ic_neu_t": t_n[0],
        "q5_q1_daily": spread.mean(), "q5_q1_ann": spread.mean() * 252,
        "q5_q1_t": t_s[0],
    }


def main():
    df = load_panel()
    print(f"面板: {len(df):,} 列, {df['code'].nunique()} 檔, {df['date'].min()} ~ {df['date'].max()}")
    rows = []
    for i, col in enumerate(FEATURE_COLS, 1):
        r = eval_feature(df, col)
        if r:
            rows.append(r)
        if i % 10 == 0:
            print(f"  ...{i}/{len(FEATURE_COLS)}", flush=True)
    res = pd.DataFrame(rows).sort_values("ic_t", ascending=False)
    res.to_csv(OUT, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 160)
    print("\n=== Rank IC 排行 (|t| 前 10) ===")
    show = res.reindex(res["ic_t"].abs().sort_values(ascending=False).index)
    print(show[["feature", "n_days", "ic_mean", "ic_t", "ic_pos",
                "ic_neu_t", "q5_q1_ann", "q5_q1_t"]].head(10).round(4).to_string(index=False))
    inst = res[res["feature"].isin(INST_COLS)]
    print(f"\n法人/融資融券 12 維中 |IC t| > 2 的: {(inst['ic_t'].abs() > 2).sum()} 個")
    print(f"已存 -> {OUT}")

    try:
        from src.experiment_log import log_trial
        top = res.iloc[0]
        log_trial("factor_eval", "rank_ic_quintile",
                  {"n_features": len(res)},
                  {"top_feature": top["feature"], "top_ic_t": round(float(top["ic_t"]), 2),
                   "top_q5q1_ann": round(float(top["q5_q1_ann"]), 4)},
                  notes="IC+五分位評估 (results/factor_eval.csv)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
