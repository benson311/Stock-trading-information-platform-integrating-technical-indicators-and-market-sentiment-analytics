"""Phase 2B: Yahoo 股票新聞時間戳回填。

使用者已有的 stocks/{code}/news/yahoo_news_{code}.csv (12,365 則, 標題+網址)
缺時間戳。逐則抓文章頁, 從 JSON-LD 取 datePublished; 死連結退回 Wayback。

輸出: data/processed/yahoo_news_dated.parquet (code, title, url, date)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import csv
import re
import sys
import time
import json as jsonlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import config

OUT_PATH = os.path.join(config.PROCESSED_DIR, "yahoo_news_dated.parquet")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
WORKERS = 5


def list_all_news() -> pd.DataFrame:
    """彙整所有股票的 Yahoo 新聞 CSV。"""
    rows = []
    for code in sorted(os.listdir(config.RAW_STOCKS_DIR)):
        ndir = os.path.join(config.RAW_STOCKS_DIR, code, "news")
        if not os.path.isdir(ndir):
            continue
        for fn in os.listdir(ndir):
            fpath = os.path.join(ndir, fn)
            with open(fpath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames
                tcol = next((c for c in cols if c.lstrip("\ufeff").strip() == "title"), None)
                ucol = next((c for c in cols if c.strip().lower() == "url"), None)
                if not (tcol and ucol):
                    continue
                for row in reader:
                    url = (row.get(ucol) or "").strip()
                    if url.startswith("http"):
                        rows.append({"code": code,
                                    "title": (row.get(tcol) or "").strip(),
                                    "url": url})
    df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df


def extract_date(html: str) -> str | None:
    """從 Yahoo 文章頁抽 datePublished (JSON-LD 或 meta)。"""
    m = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', html)
    if not m:
        m = re.search(r'property="article:published_time"[^>]*content="(\d{4}-\d{2}-\d{2})', html)
    if not m:
        m = re.search(r'"publishDate"\s*:\s*"?(\d{4}-\d{2}-\d{2})', html)
    return m.group(1) if m else None


def fetch_one(url: str, max_retry=2) -> str | None:
    """抓 Yahoo 文章頁取日期; 404/timeout 退回 Wayback。"""
    for attempt in range(max_retry):
        try:
            r = requests.get(url, timeout=20, headers=HEADERS)
            if r.status_code == 200:
                d = extract_date(r.text)
                if d:
                    return d
                return None  # 200 但無日期
            if r.status_code in (404, 410):
                break
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    # Wayback 備援
    try:
        wb = f"http://archive.org/wayback/available?url={url}"
        r = requests.get(wb, timeout=15, headers=HEADERS)
        snap = r.json().get("archived_snapshots", {}).get("closest", {})
        if snap and snap.get("available"):
            ts = snap["timestamp"]
            r2 = requests.get(
                f"http://web.archive.org/web/{ts}/{url}", timeout=30, headers=HEADERS)
            if r2.status_code == 200:
                return extract_date(r2.text)
    except Exception:
        pass
    return None


def load_done() -> set:
    if not os.path.exists(OUT_PATH):
        return set()
    return set(pd.read_parquet(OUT_PATH)["url"])


def run(workers=WORKERS, limit=None):
    all_news = list_all_news()
    done = load_done()
    todo = all_news[~all_news["url"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"Yahoo 新聞總數: {len(all_news)}, 已完成: {len(done)}, 本次: {len(todo)}")

    results = []
    t0 = time.time()
    n = 0
    got_dates = 0

    def work(row):
        d = fetch_one(row["url"])
        return {**row.to_dict(), "date": d}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, row) for _, row in todo.iterrows()]
        for fut in as_completed(futs):
            r = fut.result()
            n += 1
            if r.get("date"):
                got_dates += 1
            results.append(r)
            if n % 100 == 0:
                el = time.time() - t0
                speed = n / el
                print(f"  {n}/{len(todo)}, 有日期 {got_dates} ({got_dates/n:.0%}), "
                      f"{speed:.1f} 篇/s, 剩 {(len(todo)-n)/speed/3600:.2f}h", flush=True)
                _save(results)

    _save(results, final=True)
    print(f"\n完成: {n} 篇, 含日期 {got_dates} ({got_dates/max(n,1):.0%}), "
          f"耗時 {(time.time()-t0)/60:.1f} 分")


def _save(results, final=False):
    if not results:
        return
    df_new = pd.DataFrame(results)
    if os.path.exists(OUT_PATH):
        old = pd.read_parquet(OUT_PATH)
        df_all = pd.concat([old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)
    else:
        df_all = df_new
    df_all.to_parquet(OUT_PATH, index=False)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit=limit)