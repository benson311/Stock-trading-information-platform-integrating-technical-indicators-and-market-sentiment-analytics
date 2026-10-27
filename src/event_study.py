"""順位 3: 事件研究 / PEAD 漂移 — 找出「持有期以週計」的訊號。

動機: 情緒訊號半衰期<1日且跨年不穩定, 台灣日頻成本無法變現。
但學術文獻中「財報後漂移 (PEAD)」以「週」為單位存在 —
若某類事件的 t+1~t+20 異常報酬顯著 > 周轉成本, 即為可執行策略。

方法:
  1. 規則式事件分類 (10 類, 各含方向)
  2. 事件日 t (新聞日), t+1 收盤進場 (零前視), 持有 H 日
  3. 異常報酬 AR = 個股累積報酬 - 市場(302檔等權)累積報酬
  4. CAR(H) 按事件類別聚合, t 檢定, 並減成本看淨值
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import re
import numpy as np
import pandas as pd

import config
from src.build_features import TECH_COLS

FROZEN = os.path.join(config.PROCESSED_DIR, "cnyes_news_raw.parquet")
FEAT = os.path.join(config.PROCESSED_DIR, "features.parquet")
OUT = os.path.join(config.RESULTS_DIR, "event_study_car.csv")

# -------------------- 事件分類 (規則式) --------------------
EVENT_RULES = [
    # (類別, 方向, regex) — 標題 + 內文前 500 字
    ("營收創高/成長", +1,
     r"營收創高|營收新高|營收.{0,6}創.{0,2}新高|歷史新高|累計營收.{0,4}(創高|新高|成長)|"
     r"營收.{0,8}(年增|月增|雙增|成長)|月營收.{0,4}(創|改寫).{0,2}高"),
    ("營收衰退/下滑", -1,
     r"營收.{0,8}(年減|月減|衰退|下滑|雙降|走跌)|連.{0,2}(黑|跌)"),
    ("獲利亮眼", +1,
     r"(獲利|盈餘|EPS).{0,10}(創高|新高|成長|攀升|大增|暴增|優於預期|超乎預期|亮眼)|"
     r"三率三升|單季賺|大賺|獲利能力.{0,4}提升"),
    ("虧損/獲利衰退", -1,
     r"(虧損|轉虧|由盈轉虧|獲利.{0,6}(衰退|下滑|腰斬|重挫)|EPS.{0,6}(衰退|下滑)|"
     r"低於預期|不如預期|遜於預期)"),
    ("法人/投信買超", +1, r"(外資|投信|法人).{0,8}買超"),
    ("法人/投信賣超", -1, r"(外資|投信|法人).{0,8}賣超"),
    ("目標價/評等上調", +1, r"目標價.{0,4}(上調|調升|調高|調升至)|評等.{0,4}(上調|調升|調高)|升評"),
    ("目標價/評等下調", -1, r"目標價.{0,4}(下調|調降|調低|調降至)|評等.{0,4}(下調|調降)|降評"),
    ("接單/得標/認證", +1, r"(接單|得標|搶單|標下|訂單.{0,6}(成長|滿載|能見度高)|打入.{0,8}供應鏈|"
                            r"取得.{0,8}(認證|藥證|證書|資格)|FDA.{0,6}(核准|通過))"),
    ("併購/投資擴產", +1, r"(併購|收購|合併|擴產|新建廠|加碼投資|投資.{0,8}億元|進軍)"),
    ("庫藏股/減資", +1, r"(庫藏股|買回股份|減資.{0,4}(彌補|現金)|現金減資)"),
    ("利空事故/處分", -1, r"(火災|爆炸|停工|停產|訴訟|求償|違約|警示股|處分|裁罰|洩漏|食安|回收.{0,3}(下架)?)"),
]

HORIZONS = [1, 5, 10, 20]
ROUND_TRIP = 0.00585   # 台灣單腿全周轉 (手續費x2 + 證交稅)


def classify_events():
    """從有標記文章抽取 (code, date, 類別, 方向) 事件表。"""
    df = pd.read_parquet(FROZEN)
    df = df[df["stock_codes"].str.len() > 0]
    events = []
    for _, r in df.iterrows():
        text = (r["title"] or "") + " " + (r["content"] or "")[:500]
        matched = []
        for cat, direction, pat in EVENT_RULES:
            if re.search(pat, text):
                matched.append((cat, direction))
        if not matched:
            continue
        # 同篇多類 → 全部記錄 (事件研究允許重疊分類)
        for cat, direction in matched:
            for code in str(r["stock_codes"]).split(","):
                if code.strip():
                    events.append((code.strip(), r["date"], cat, direction))
    ev = pd.DataFrame(events, columns=["code", "date", "event", "direction"])
    ev = ev.drop_duplicates(subset=["code", "date", "event"])  # 同股同日同類去重

    # 獨立事件數: 同股同類 10 個交易日內只留第一筆 (避免連續日報導膨脹 t 值)
    ev = ev.sort_values(["code", "event", "date"]).reset_index(drop=True)
    keep = []
    last = {}
    for _, r in ev.iterrows():
        key = (r["code"], r["event"])
        if key in last and r["date"] <= last[key]:
            continue
        keep.append(r)
        last[key] = (pd.Timestamp(r["date"]) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    ev = pd.DataFrame(keep).reset_index(drop=True)

    print(f"事件數 (獨立數, 10日內同股同類去重): {len(ev):,} "
          f"({ev['event'].nunique()} 類, {ev['code'].nunique()} 檔)")
    print(ev["event"].value_counts().to_string())
    return ev


def build_market_returns():
    """由 features.parquet 建個股日報酬矩陣 + 市場等權日報酬 (共同日曆)。

    台灣漲跌停 ±10%: 單日報酬裁切至 [-0.10, +0.10] (合法界內),
    同時消滅 1752 南光等「減資跳空」資料異常的污染。
    """
    df = pd.read_parquet(FEAT, columns=["code", "date", "close"])
    df["date"] = df["date"].astype(str)
    panel = df.pivot_table(index="date", columns="code", values="close")
    panel = panel.ffill(limit=5)          # 停牌: 前值填補
    rets = panel.pct_change(fill_method=None)
    # 台灣 ±10% 漲跌停裁切 (原則性去異常)
    rets = rets.clip(-0.10, 0.10)
    market = rets.mean(axis=1)             # 市場 = 302 檔等權 (裁切後)
    return rets, market


def event_study(ev, rets, market):
    """每事件計算 CAR(H) = 累積個股報酬 - 累積市場報酬 (t+1 進場)。"""
    dates = rets.index.values
    dpos = {d: i for i, d in enumerate(dates)}

    rows = []
    n_missing = 0
    for _, e in ev.iterrows():
        i = dpos.get(e["date"])
        if i is None:
            n_missing += 1
            continue
        code = e["code"]
        s = rets[code].values if code in rets.columns else None
        if s is None:
            n_missing += 1
            continue
        row = {"code": code, "date": e["date"], "event": e["event"],
               "direction": e["direction"]}
        for H in HORIZONS:
            end = i + 1 + H
            if end > len(dates):
                row[f"car_{H}"] = np.nan
                continue
            # 個股累積報酬 (t+1 ~ t+1+H), NaN(停牌) 視為 0
            s_seg = s[i + 1:end]
            m_seg = market.values[i + 1:end]
            s_cum = np.nansum(s_seg)
            m_cum = np.nansum(m_seg)
            row[f"car_{H}"] = s_cum - m_cum
        rows.append(row)

    ed = pd.DataFrame(rows)
    print(f"\n有效事件: {len(ed):,} (無法對齊價格: {n_missing:,})")
    return ed


def summarize(ed):
    """按事件類別聚合 CAR, 檢定, 成本可行性。"""
    from scipy import stats
    out = []
    for (ev_name, direction), g in ed.groupby(["event", "direction"]):
        row = {"event": ev_name, "dir": direction, "n": len(g)}
        for H in HORIZONS:
            c = g[f"car_{H}"].dropna()
            if len(c) < 20:
                row[f"car_{H}"] = np.nan
                row[f"t_{H}"] = np.nan
                continue
            t, p = stats.ttest_1samp(c, 0)
            row[f"car_{H}"] = c.mean()
            row[f"t_{H}"] = t
            row[f"p_{H}"] = p
        out.append(row)
    s = pd.DataFrame(out).sort_values("car_10", ascending=False)
    return s


def main():
    print("=== 步驟 1: 事件抽取 ===")
    ev = classify_events()

    print("\n=== 步驟 2: 建市場報酬矩陣 ===")
    rets, market = build_market_returns()
    print(f"共同日曆: {len(rets)} 日 × {rets.shape[1]} 檔")

    print("\n=== 步驟 3: 事件研究 (CAR) ===")
    ed = event_study(ev, rets, market)

    print("\n=== 步驟 4: 各事件類別 CAR 統計 (bp) ===")
    s = summarize(ed)
    disp = s.copy()
    for H in HORIZONS:
        disp[f"car_{H}"] = (disp[f"car_{H}"] * 1e4).round(1)
        disp[f"t_{H}"] = disp[f"t_{H}"].round(2)
    cols = ["event", "dir", "n"] + [f"car_{h}" for h in HORIZONS] + [f"t_{h}" for h in HORIZONS]
    print(disp[cols].to_string(index=False))

    # 成本可行性: 持有10日, 淨值 = CAR(10) - 0.585%
    print("\n=== 步驟 5: 成本可行性 (持有 10 日, 單腿成本 58.5bp) ===")
    viable = []
    for _, r in s.iterrows():
        if pd.isna(r.get("car_10")) or r["n"] < 20:
            continue
        net10_bp = (r["car_10"] - ROUND_TRIP) * 1e4
        sig = "***" if (r.get("p_10", 1) or 1) < 0.01 else ("**" if (r.get("p_10", 1) or 1) < 0.05 else ("*" if (r.get("p_10", 1) or 1) < 0.1 else ""))
        viable.append({"event": r["event"], "dir": r["dir"], "n": r["n"],
                       "CAR10_bp": round(r["car_10"] * 1e4, 1),
                       "淨CAR10_bp": round(net10_bp, 1),
                       "年化(1-H循環)": f"{net10_bp/1e4 * (252/10):+.1%}",
                       "t": round(r.get("t_10", np.nan), 2),
                       "sig": sig})
    v = pd.DataFrame(viable).sort_values("淨CAR10_bp", ascending=False)
    print(v.to_string(index=False))
    print("\n(淨CAR10>0 = 該事件訊號活得過台灣成本)")

    s.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n已存 -> {OUT}")

    # 方向一致性檢查
    print("\n=== 附註: 方向一致性 ===")
    for _, r in s.iterrows():
        if pd.isna(r.get("car_10")):
            continue
        agree = "✓一致" if np.sign(r["car_10"]) == np.sign(r["dir"]) else "✗反向"
        print(f"  {r['event']:<14} 預期方向 {r['dir']:+d} | 實際 CAR10 {r['car_10']*1e4:+.1f}bp {agree}")


if __name__ == "__main__":
    main()