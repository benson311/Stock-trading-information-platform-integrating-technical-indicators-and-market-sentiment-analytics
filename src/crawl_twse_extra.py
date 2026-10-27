"""爬取 TWSE 官方三大法人買賣超 (T86) 與融資融券餘額 (MI_MARGN) 日資料。

端點 (TWSE rwd JSON API):
  三大法人: https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&response=json
  融資融券: https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date=YYYYMMDD
            &response=json&selectType=STOCK   (STOCK=僅股票, 排除 ETF/受益憑證)

輸出 (與既有 t86 檔案同格式, 存於原始資料目錄):
  {RAW_DIR}/twse/t86/t86_YYYYMMDD.json          (補 2025-01-01 → 今)
  {RAW_DIR}/twse/mi_margn/mi_margn_YYYYMMDD.json (2011-01-01 → 今)

特性:
  - 僅請求週一~週五 (週末/節日 TWSE 回「很抱歉」, 存檔後不再重抓)
  - 已存在檔案自動跳過 → 可中斷重跑 (checkpoint)
  - 禮貌性延遲 1.0~1.4s + 失敗退避, 避免被 TWSE 封鎖
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import datetime as dt
import json
import random
import time

import requests

import config

T86_DIR = config.RAW_T86_DIR
MARG_DIR = config.RAW_MARGIN_DIR

T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
MARG_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "application/json",
}


def fetch_json(url: str, params: dict, max_retry: int = 3) -> dict | None:
    """GET JSON; 失敗退避重試。回傳 None 表示連續失敗 (呼叫端應長睡)。"""
    for attempt in range(1, max_retry + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=(5, 15))
            if r.status_code == 200:
                return r.json()
            print(f"    HTTP {r.status_code}, 第 {attempt} 次重試...")
        except Exception as e:  # noqa: BLE001
            print(f"    請求失敗 ({type(e).__name__}: {e}), 第 {attempt} 次重試...")
        time.sleep(10 * attempt)  # 退避 10s, 20s, 30s
    return None


def date_range(start: str, end: str):
    d = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    while d <= last:
        if d.weekday() < 5:  # 週一~週五
            yield d.strftime("%Y%m%d")
        d += dt.timedelta(days=1)


def crawl(report: str, start: str, end: str):
    """report: 't86' 或 'margin'。"""
    if report == "t86":
        url, out_dir, prefix = T86_URL, T86_DIR, "t86"
        extra = {}
    else:
        url, out_dir, prefix = MARG_URL, MARG_DIR, "mi_margn"
        extra = {"selectType": "STOCK"}
    os.makedirs(out_dir, exist_ok=True)

    dates = list(date_range(start, end))
    todo = [d for d in dates
            if not os.path.exists(os.path.join(out_dir, f"{prefix}_{d}.json"))]
    n = len(todo)
    print(f"[{report}] {start} ~ {end}: 共 {len(dates)} 個工作日, 待抓 {n} 檔")
    if n == 0:
        return

    t0 = time.time()
    n_ok = n_nodata = 0
    fail_streak = 0
    for i, d in enumerate(todo, 1):
        out = os.path.join(out_dir, f"{prefix}_{d}.json")
        data = fetch_json(url, {"date": d, "response": "json", **extra})
        if data is None:
            fail_streak += 1
            print(f"[{report}] {d} 連續失敗 {fail_streak} 次, 暫停 120s...")
            if fail_streak >= 5:
                print(f"[{report}] 連續 5 次失敗, 長暫停 300s")
                time.sleep(300)
                fail_streak = 0
            else:
                time.sleep(120)
            continue
        fail_streak = 0
        # 存檔 (含「很抱歉」的空回應, 避免日後重抓)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        if data.get("stat") == "OK" and (("data" in data and data["data"])
                                        or any(t.get("data") for t in data.get("tables", []))):
            n_ok += 1
        else:
            n_nodata += 1

        if i % 25 == 0 or i == n:
            el = time.time() - t0
            eta = el / i * (n - i)
            print(f"[{report}] {i}/{n} ({i/n:.0%}) 有資料 {n_ok} / 無資料 {n_nodata}, "
                  f"已花 {el/60:.1f} 分, 預估還要 {eta/60:.1f} 分", flush=True)
        time.sleep(1.0 + random.random() * 0.4)
    print(f"[{report}] 完成: 有資料 {n_ok}, 無資料 (假日) {n_nodata}, 失敗略過 {n - n_ok - n_nodata}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", choices=["t86", "margin", "both"], default="both")
    ap.add_argument("--t86-start", default="2025-01-01",
                    help="T86 起始日 (既有檔案到 2024-12-31, 預設補 2025 起)")
    ap.add_argument("--margin-start", default="2011-01-01",
                    help="MI_MARGN 起始日 (與 Yahoo 股價起點一致)")
    ap.add_argument("--end", default=dt.date.today().isoformat(),
                    help="結束日 (預設今天)")
    args = ap.parse_args()

    if args.report in ("margin", "both"):
        crawl("margin", args.margin_start, args.end)
    if args.report in ("t86", "both"):
        crawl("t86", args.t86_start, args.end)


if __name__ == "__main__":
    main()
