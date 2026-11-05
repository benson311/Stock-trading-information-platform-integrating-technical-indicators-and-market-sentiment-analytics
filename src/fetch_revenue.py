import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""抓取 TWSE OpenAPI 月營收快照 (t187ap05_L: 全市場最新一個月, 含年增率)。

→ data/processed/revenue_latest.parquet (code, month, revenue, yoy_pct, mom_pct)

用途: webapp「財報基本面」策略之最新月營收年增排名。
(歷史逐月版需 MOPS t187ap05_L1 逐月爬蟲 —— 見 DATA_SOURCES.md roadmap)

學術依據: 營收年增率為本計畫事件研究中效應最強之事件類別 (營收創高 CAR p<1e-21);
外部研究 (DAViS 延伸) 顯示營收年增 十分位多空 扣成本年化約 +13% (t 2.36)。
"""
import json
import os

import pandas as pd
import requests

import config

URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
OUT = os.path.join(config.PROCESSED_DIR, "revenue_latest.parquet")


def fetch() -> pd.DataFrame:
    r = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    rows = r.json()
    recs = []
    for x in rows:
        def num(k):
            v = str(x.get(k, "")).replace(",", "").strip()
            try:
                return float(v)
            except ValueError:
                return None
        recs.append({
            "code": str(x.get("公司代號", "")).strip(),
            "name": str(x.get("公司名稱", "")).strip(),
            "month": str(x.get("資料年月", "")),
            "revenue": num("營業收入-當月營收"),
            "yoy_pct": num("營業收入-去年同月增減(%)"),
            "mom_pct": num("營業收入-上月比較增減(%)"),
        })
    df = pd.DataFrame(recs)
    df = df[df["code"].str.match(r"^\d{4}[A-Z]?$", na=False)]  # 只留股票代碼
    return df


def main():
    df = fetch()
    df.to_parquet(OUT, index=False)
    pos = df[df["yoy_pct"].notna()]
    print(f"月營收快照: {len(df)} 檔 (有年增率 {len(pos)}), 月份 {df['month'].iloc[0] if len(df) else '-'}")
    print(f"年增率中位數 {pos['yoy_pct'].median():.2f}% | 正成長比例 {(pos['yoy_pct']>0).mean():.1%}")
    try:
        from src.experiment_log import log_trial
        log_trial("data", "mops_revenue_snapshot",
                  {"month": str(df["month"].iloc[0]) if len(df) else ""},
                  {"n": len(df), "yoy_pos_ratio": round(float((pos["yoy_pct"] > 0).mean()), 3)})
    except Exception:
        pass


if __name__ == "__main__":
    main()
