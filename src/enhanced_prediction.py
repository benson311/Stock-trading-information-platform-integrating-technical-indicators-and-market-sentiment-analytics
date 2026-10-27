"""增強預測: 特徵升級 + 換預測目標 — 回答「怎樣才能讓預測準確」。

證據動機:
  - 樹模型 TECH-only 已達 AUC 0.5448 (LSTM 僅 0.514) → 模型換對了
  - 特徵重要度: 短期動能/量能主導 → 往動能與市場結構特徵升級
  - 文獻: 波動率遠比報酬方向可預測 (vol clustering)

增強特徵 (35 維, 全部 t 時點可知, 零前視):
  動能: ret_1d/5d/10d/20d/60d
  波動: rv_5/10/20, ATR14%, 日振幅, 跳空
  量能: 量比(5/20), �金額 z-score, 量變
  市場: 大盤 ret_1d/5d, 大盤 rv_20, 相關性 beta_60
  產業: 產業 ret_1d/5d, 個股-產業相對強弱
  橫斷面排名: ret_5d/RSI12/rv_20 的全體日百分位
  技術: 既有 12 指標 | 新聞: net_pre (對照)

預測目標:
  A. 明日方向 (基準 0.5448)
  B. 未來一週方向 (5 日) — 動能因子更強、成本更友善
  C. 未來一週已實現波動 (回歸) — 展示「真正可預測的東西」
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score

import config
from src.build_features import TECH_COLS

FEAT = os.path.join(config.PROCESSED_DIR, "features.parquet")
TIMED = os.path.join(config.PROCESSED_DIR, "timed_sentiment_index.parquet")
META = os.path.join(config.PROCESSED_DIR, "stock_metadata.json")
OUT = os.path.join(config.RESULTS_DIR, "enhanced_prediction_results.csv")

CLIP = 0.10  # 台灣漲跌停


def build_enhanced_features():
    df = pd.read_parquet(FEAT)
    df["date"] = df["date"].astype(str)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # ---------- 個股層特徵 (groupby rolling, 零前視) ----------
    g = df.groupby("code", sort=False)
    df["ret_1d"] = g["close"].pct_change().clip(-CLIP, CLIP)
    for n in (5, 10, 20, 60):
        df[f"ret_{n}d"] = g["close"].pct_change(n).clip(-CLIP * 3, CLIP * 3)
    for n in (5, 10, 20):
        df[f"rv_{n}"] = g["ret_1d"].transform(lambda s: s.rolling(n).std())
    prev_close = g["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_close).abs(),
                               (df["low"] - prev_close).abs()))
    df["atr14"] = tr.groupby(df["code"]).transform(lambda s: s.rolling(14).mean())
    df["atr14_pct"] = df["atr14"] / df["close"]
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["gap_pct"] = (df["open"] / prev_close - 1).clip(-CLIP, CLIP)
    # 量能
    vma5 = g["volume"].transform(lambda s: s.rolling(5).mean())
    vma20 = g["volume"].transform(lambda s: s.rolling(20).mean())
    df["vol_ratio"] = vma5 / (vma20 + 1)
    df["vol_chg"] = g["volume"].pct_change().clip(-5, 5)
    dvol = df["volume"] * df["close"]
    dm = dvol.groupby(df["code"]).transform(lambda s: s.rolling(60).mean())
    ds = dvol.groupby(df["code"]).transform(lambda s: s.rolling(60).std())
    df["dvol_z"] = ((dvol - dm) / (ds + 1)).clip(-10, 10)

    # ---------- 市場層特徵 ----------
    dates_all = np.array(sorted(df["date"].unique()))
    dpos = {d: i for i, d in enumerate(dates_all)}
    panel_ret = df.pivot_table(index="date", columns="code", values="ret_1d")
    mkt_ret = panel_ret.mean(axis=1)
    mkt_rv20 = mkt_ret.rolling(20).std()
    mkt_r = {}
    mkt_r["mkt_ret_1d"] = mkt_ret
    mkt_r["mkt_ret_5d"] = (1 + mkt_ret).rolling(5).apply(np.prod, raw=True) - 1
    mkt_r["mkt_rv_20"] = mkt_rv20
    mkt_df = pd.DataFrame(mkt_r)
    df = df.merge(mkt_df, left_on="date", right_index=True, how="left")

    # beta_60: 個股 vs 大盤 60 日相關 (向量化滾動)
    df["_xy"] = df["ret_1d"] * df["mkt_ret_1d"]
    df["_y2"] = df["mkt_ret_1d"] ** 2
    g = df.groupby("code", sort=False)
    cov = g["_xy"].transform(lambda s: s.rolling(60).mean()) - \
        g["ret_1d"].transform(lambda s: s.rolling(60).mean()) * \
        g["mkt_ret_1d"].transform(lambda s: s.rolling(60).mean())
    var = g["_y2"].transform(lambda s: s.rolling(60).mean()) - \
        g["mkt_ret_1d"].transform(lambda s: s.rolling(60).mean()) ** 2
    df["beta_60"] = (cov / (var + 1e-12)).clip(-5, 5)
    df = df.drop(columns=["_xy", "_y2"])

    # ---------- 產業層特徵 ----------
    meta = json.load(open(META, encoding="utf-8"))
    df["industry"] = df["code"].map(lambda c: meta.get(c, {}).get("industry", "00"))
    sec_ret = df.groupby(["industry", "date"])["ret_1d"].mean().rename("sec_ret_1d")
    df = df.merge(sec_ret, on=["industry", "date"], how="left")
    sec5 = df.groupby(["industry", "date"])["sec_ret_1d"].first().reset_index()
    sec5["sec_ret_5d"] = sec5.groupby("industry")["sec_ret_1d"].transform(
        lambda s: (1 + s).rolling(5).apply(np.prod, raw=True) - 1)
    df = df.merge(sec5[["industry", "date", "sec_ret_5d"]], on=["industry", "date"],
                  how="left")
    df["rel_sec_5d"] = df["ret_5d"] - df["sec_ret_5d"]

    # ---------- 橫斷面排名 (每日全體百分位) ----------
    by_date = df.groupby("date", sort=False)
    df["cs_rank_ret5"] = by_date["ret_5d"].rank(pct=True)
    df["cs_rank_rsi"] = by_date["RSI12"].rank(pct=True)
    df["cs_rank_rv"] = by_date["rv_20"].rank(pct=True)

    # ---------- 新聞特徵 (對照組) ----------
    timed = pd.read_parquet(TIMED)[["code", "date", "net_pre"]]
    timed["date"] = timed["date"].astype(str)
    df = df.merge(timed, on=["code", "date"], how="left")
    df["net_pre"] = df["net_pre"].fillna(0)

    # ---------- 目標 ----------
    g = df.groupby("code", sort=False)
    df["fwd_ret_1d"] = g["ret_1d"].shift(-1)
    df["fwd_ret_5d"] = g["close"].shift(-5) / df["close"] - 1
    df["y_1d"] = (df["fwd_ret_1d"] > 0).astype(int)
    df["y_5d"] = (df["fwd_ret_5d"] > 0).astype(int)
    df["fwd_rv_5"] = g["ret_1d"].shift(-1).rolling(5).std().reset_index(drop=True) \
        if False else g["ret_1d"].transform(lambda s: s.shift(-1).rolling(5).std())

    return df


FEAT_GROUPS = {
    "TECH": TECH_COLS,
    "MOM": ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d"],
    "VOL": ["rv_5", "rv_10", "rv_20", "atr14_pct", "range_pct", "gap_pct"],
    "FLOW": ["vol_ratio", "vol_chg", "dvol_z", "OBV", "ADOSC_3_10"],
    "MKT": ["mkt_ret_1d", "mkt_ret_5d", "mkt_rv_20", "beta_60"],
    "SEC": ["sec_ret_1d", "sec_ret_5d", "rel_sec_5d"],
    "CS": ["cs_rank_ret5", "cs_rank_rsi", "cs_rank_rv"],
    "NEWS": ["net_pre"],
}


def main():
    print("=== 步驟 1: 增強特徵建置 ===")
    df = build_enhanced_features()
    all_feats = sum(FEAT_GROUPS.values(), [])
    keep = [c for c in all_feats if c in df.columns]
    print(f"特徵數: {len(keep)} 維 | 樣本: {len(df):,}")

    df = df.dropna(subset=keep + ["fwd_ret_1d", "fwd_ret_5d", "fwd_rv_5"]).reset_index(drop=True)
    dates = np.array(sorted(df["date"].unique()))
    cut = dates[int(len(dates) * 0.75)]
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    print(f"訓練 {len(train):,} | 測試 {len(test):,} (>= {cut})")

    results = []

    print("\n=== 步驟 2A: 明日方向 (基準 AUC 0.5448) ===")
    m1 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                        max_depth=6, min_samples_leaf=200,
                                        random_state=0)
    m1.fit(train[keep], train["y_1d"])
    p1 = m1.predict_proba(test[keep])[:, 1]
    auc1 = roc_auc_score(test["y_1d"], p1)
    acc1 = accuracy_score(test["y_1d"], p1 > 0.5)
    print(f"  AUC {auc1:.4f} | 準確率 {acc1:.3f}")
    results.append(["明日方向", auc1, acc1, None])

    print("\n=== 步驟 2B: 未來一週方向 ===")
    m2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                        max_depth=6, min_samples_leaf=200,
                                        random_state=0)
    m2.fit(train[keep], train["y_5d"])
    p5 = m2.predict_proba(test[keep])[:, 1]
    auc5 = roc_auc_score(test["y_5d"], p5)
    acc5 = accuracy_score(test["y_5d"], p5 > 0.5)
    print(f"  AUC {auc5:.4f} | 準確率 {acc5:.3f}")
    results.append(["一週方向", auc5, acc5, None])

    print("\n=== 步驟 2C: 未來一週已實現波動 (回歸) ===")
    m3 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                       max_depth=6, min_samples_leaf=200,
                                       random_state=0)
    m3.fit(train[keep], train["fwd_rv_5"])
    vhat = m3.predict(test[keep])
    r2 = r2_score(test["fwd_rv_5"], vhat)
    corr = np.corrcoef(vhat, test["fwd_rv_5"])[0, 1]
    print(f"  R² {r2:.3f} | 相關係數 {corr:.3f}   <- 「可預測性」的量級對比")
    results.append(["一週波動率", None, None, r2])

    # ---------- 步驟 3: 一週方向的多空價差 (可執行性) ----------
    print("\n=== 步驟 3: 一週方向訊號的多空價差 ===")
    t2 = test[["code", "date", "fwd_ret_5d"]].copy()
    t2["p"] = p5
    t2 = t2.dropna()
    spreads = []
    for d, gdf in t2.groupby("date"):
        gdf = gdf.sort_values("p")
        if len(gdf) < 20:
            continue
        k = max(5, int(len(gdf) * 0.25))
        spreads.append(gdf.tail(k)["fwd_ret_5d"].mean()
                       - gdf.head(k)["fwd_ret_5d"].mean())
    s = np.array(spreads)
    weekly_cost = 0.00585 * 2          # 多空兩腿來回
    net_weekly = s.mean() - weekly_cost
    print(f"  週多空價差 (毛): {s.mean()*1e4:+.1f}bp/週 | "
          f"年化 {(1+s.mean())**52-1:+.1%}")
    print(f"  扣台灣兩腿成本 (117bp/週): 淨 {net_weekly*1e4:+.1f}bp/週 | "
          f"年化 {(1+net_weekly)**52-1:+.1%}")
    print(f"  勝率 {(s > 0).mean():.1%} | 有效週數 {len(s)}")

    # ---------- 步驟 4: 特徵重要度 (一週方向模型) ----------
    print("\n=== 步驟 4: 特徵重要度 (permutation, 一週方向) ===")
    from sklearn.inspection import permutation_importance
    sample = test.sample(40000, random_state=0)
    pi = permutation_importance(m2, sample[keep], sample["y_5d"],
                                scoring="roc_auc", n_repeats=3,
                                random_state=0, n_jobs=-1)
    imp = pd.DataFrame({"feature": keep, "imp": pi.importances_mean})
    imp["group"] = imp["feature"].map(
        lambda f: next(gg for gg, lst in FEAT_GROUPS.items() if f in lst))
    imp = imp.sort_values("imp", ascending=False)
    print(imp.head(12).to_string(index=False))
    group_imp = imp.groupby("group")["imp"].sum().sort_values(ascending=False)
    print("\n特徵組貢獻:")
    print(group_imp.to_string())

    # 儲存
    out = pd.DataFrame(results, columns=["目標", "AUC", "準確率", "R²"])
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n已存 -> {OUT}")


if __name__ == "__main__":
    main()