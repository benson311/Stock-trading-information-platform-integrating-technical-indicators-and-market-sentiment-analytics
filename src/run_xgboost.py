"""順位 2: 樹模型 (HistGBM) 橫斷面「方向分類」— 新聞特徵的真實貢獻。

設計:
  樣本: 302 檔 × 全部交易日 (個股×日), 特徵在 t、預測 close[t]→close[t+1] 方向
  特徵組:
    TECH : 12 技術指標 + ret_1d + ret_5d
    NEWS : 盤中淨情緒 net_pre + 篇數 (「可交易」時效, 取自時間感知指數)
    SECTOR: 產業情緒 (全新聞, 含盤後成分)
  消融: TECH / TECH+NEWS / TECH+NEWS+SECTOR / NEWS-only → 各自 Test AUC
  細部: 新聞日子集 AUC (訊號只可能出現在哪裡)
  特徵重要度: permutation importance (TECH+NEWS 模型)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance

import config
from src.build_features import TECH_COLS, INST_COLS

FEAT = os.path.join(config.PROCESSED_DIR, "features.parquet")
TIMED = os.path.join(config.PROCESSED_DIR, "timed_sentiment_index.parquet")
OUT = os.path.join(config.RESULTS_DIR, "xgboost_inst_importance.csv")

SEED = 0


def build_dataset():
    df = pd.read_parquet(FEAT)
    df["date"] = df["date"].astype(str)

    # 目標: 次日方向 (per stock, 由時間排序後 shift)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["ret_1d"] = df.groupby("code")["close"].pct_change()
    df["ret_5d"] = df.groupby("code")["close"].pct_change(5)
    grp = df.groupby("code")
    df["next_ret"] = grp["close"].shift(-1) / df["close"] - 1
    df["y"] = (df["next_ret"] > 0).astype(int)

    # 併入時間感知新聞特徵 (盤中 net_pre / 篇數)
    timed = pd.read_parquet(TIMED)
    timed["date"] = timed["date"].astype(str)
    timed = timed[["code", "date", "net_pre", "n_pre", "net_all", "n_all"]]
    df = df.merge(timed, on=["code", "date"], how="left")
    for c in ["net_pre", "n_pre", "net_all", "n_all"]:
        df[c] = df[c].fillna(0)
    df["has_news"] = (df["n_pre"] > 0).astype(int)
    df["log_n"] = np.log1p(df["n_pre"])

    df = df.dropna(subset=TECH_COLS + ["next_ret", "ret_1d", "ret_5d"]).reset_index(drop=True)

    # 時間切分: 前 75% 交易日訓練, 後 25% 測試 (對齊 LSTM 慣例)
    dates = np.array(sorted(df["date"].unique()))
    cut = dates[int(len(dates) * 0.75)]
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    print(f"樣本: 全 {len(df):,} | 訓練 {len(train):,} "
          f"({train['date'].min()} ~ <{cut}) | 測試 {len(test):,} (>= {cut})")
    return train, test, cut


def fit_gbm(train, test, cols, label):
    Xtr, ytr = train[cols], train["y"]
    Xte, yte = test[cols], test["y"]
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=SEED)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    acc = accuracy_score(yte, p > 0.5)
    # 新聞日子集 (有盤中新聞)
    mask = test["n_pre"].values > 0
    auc_news = roc_auc_score(yte[mask], p[mask]) if mask.sum() > 500 else np.nan
    print(f"  [{label:<22}] Test AUC {auc:.4f} | 準確率 {acc:.3f} | "
          f"新聞日 AUC {auc_news:.4f} (n={mask.sum():,})")
    return m, p, auc, acc, auc_news


def main():
    print("=== 順位2: 樹模型橫斷面方向分類 ===\n")
    print("步驟 1: 建資料集")
    train, test, cut = build_dataset()

    tech = TECH_COLS + ["ret_1d", "ret_5d"]
    news = ["net_pre", "log_n", "has_news"]
    sector = ["sector_pos", "sector_neg"]
    inst = INST_COLS  # 法人買賣超 + 融資融券 12 特徵

    groups = [
        ("TECH only", tech),
        ("TECH + NEWS(盤中)", tech + news),
        ("TECH + INST(法人)", tech + inst),
        ("TECH+NEWS+SECTOR+INST(全)", tech + news + sector + inst),
        ("NEWS only", news),
        ("INST only", inst),
    ]
    print("\n步驟 2: 特徵組消融 (Test = 時間外樣本)")
    results = {}
    for label, cols in groups:
        _, p, auc, acc, auc_news = fit_gbm(train, test, cols, label)
        results[label] = {"auc": auc, "acc": acc, "auc_news": auc_news, "p": p}

    # 線性基準
    print("\n步驟 3: 線性基準 (LogisticRegression, 標準化)")
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cols = tech + news + sector
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(train[cols], train["y"])
    p_lr = lr.predict_proba(test[cols])[:, 1]
    auc_lr = roc_auc_score(test["y"], p_lr)
    print(f"  [Logistic (全部特徵)]   Test AUC {auc_lr:.4f}")

    # 特徵重要度 (permutation, TECH+NEWS+INST 模型 — 看法人特徵排名)
    print("\n步驟 4: 特徵重要度 (permutation importance, TECH+NEWS+INST 模型)")
    cols_main = tech + news + inst
    m_main, _, _, _, _ = fit_gbm(train, test, cols_main, "主模型(重訓)")
    sample = test.sample(min(60000, len(test)), random_state=SEED)
    pi = permutation_importance(m_main, sample[cols_main], sample["y"],
                                scoring="roc_auc", n_repeats=5,
                                random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": cols_main, "importance": pi.importances_mean})
    imp = imp.sort_values("importance", ascending=False)
    print(imp.head(15).to_string(index=False))

    # 總結
    print("\n" + "=" * 66)
    print("結論: 法人特徵與新聞特徵的增量預測力")
    print("=" * 66)
    base = results["TECH only"]
    for label in ["TECH + NEWS(盤中)", "TECH + INST(法人)", "TECH+NEWS+SECTOR+INST(全)"]:
        d = results[label]["auc"] - base["auc"]
        dn = results[label]["auc_news"] - base["auc_news"]
        print(f"  {label:<24} 全樣本 AUC Δ {d:+.4f} | "
              f"新聞日 AUC Δ {dn:+.4f}")
    di = results["TECH+NEWS+SECTOR+INST(全)"]["auc"] - results["TECH + NEWS(盤中)"]["auc"]
    print(f"  {'INST 增量 (全 - TECH+NEWS)':<24} 全樣本 AUC Δ {di:+.4f}")
    imp.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n特徵重要度已存 -> {OUT}")

    # 模型訊號的多空價差 (訊號強度參考, 含與原始情緒同源的時效限制)
    p_main = results["TECH + NEWS(盤中)"]["p"]
    t2 = test[["code", "date", "next_ret", "n_pre"]].copy()
    t2["p"] = p_main
    t2 = t2.dropna(subset=["next_ret"])
    spread_days = []
    for d, g in t2.groupby("date"):
        g = g.sort_values("p")
        if len(g) < 17:
            continue
        k = max(5, int(len(g) * 0.3))
        spread_days.append(g.tail(k)["next_ret"].mean() - g.head(k)["next_ret"].mean())
    if spread_days:
        s = np.array(spread_days)
        ann = s.mean() * 252
        shp = s.mean() / (s.std() + 1e-12) * np.sqrt(252)
        print(f"\n模型機率多空價差 (訊號強度, t收盤時序=含前視): "
              f"年化 {ann:+.1%}, Sharpe {shp:.2f} ({len(s)} 日)")

    # 步驟 5: 年度分解 — 驗證「新聞訊號跨年衰減」假說
    print("\n步驟 5: 盤中情緒多空價差 (A′式, 毛報酬) 逐年分解")
    from src.run_aprime import build_stock_days
    from src.backtest_sentiment_strategy import daily_legs as _legs
    idx = pd.read_parquet(TIMED)
    sd = build_stock_days(idx, "net_pre")
    daily = _legs(sd, "A")
    if not daily.empty:
        daily["year"] = daily["date"].str[:4]
        daily["spread"] = daily["long_ret"] - daily["short_ret"]
        by_year = (daily.groupby("year")
                   .agg(日均價差_bp=("spread", lambda s: s.mean() * 1e4),
                        有效日數=("spread", "size"),
                        年化=("spread", lambda s: s.mean() * 252),
                        t值=("spread", lambda s: s.mean() / (s.std() / np.sqrt(len(s))))))
        print(by_year.round(2).to_string())
        print("\n(價差 = 每日 前30%高情緒股 減 後30%低情緒股 的次日報酬; 年化=日均×252)")


if __name__ == "__main__":
    main()