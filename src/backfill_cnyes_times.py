"""A′ 步驟 1: 為「有個股標記」的鉅亨文章補抓發布時間戳。

從官網逐篇抓 meta article:published_time (UTC) → 轉台灣時區 (+8),
標記 pre_close (發布於交易日收盤 13:30 前)。

輸出: data/processed/cnyes_publish_times.parquet (aid, publish_dt, date, pre_close)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import re
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import config

FROZEN = os.path.join(config.PROCESSED_DIR, "cnyes_news_frozen.parquet")
OUT = os.path.join(config.PROCESSED_DIR, "cnyes_publish_times.parquet")
HEADERS = {"User-Agent": "research-crawler/grad-project"}
WORKERS = 6
CLOSE_H, CLOSE_M = 13, 30   # 台股收盤 13:30


def fetch_publish_time(aid, timestamp=None, original_url=None, max_retry=3):
    """抓文章頁, 回傳台灣時間 datetime 或 None。官網優先, Wayback 備援。"""
    for attempt in range(max_retry):
        try:
            r = requests.get(f"https://news.cnyes.com/news/id/{aid}",
                             timeout=25, headers=HEADERS)
            if r.status_code == 200 and len(r.content) > 5000:
                m = re.search(
                    r'property="article:published_time"[^>]*content="(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):\d{2}',
                    r.text)
                if m:
                    d = dt.datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}",
                                             "%Y-%m-%d %H:%M")
                    if ".000Z" in r.text[m.start():m.start()+80] or "Z" in (m.group(0)):
                        d += dt.timedelta(hours=8)  # UTC → 台灣
                    return d
                # JSON-LD 備援
                m2 = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})',
                               r.text)
                if m2:
                    d = dt.datetime.strptime(f"{m2.group(1)} {m2.group(2)}:{m2.group(3)}",
                                             "%Y-%m-%d %H:%M")
                    if '"datePublished"' in r.text and 'Z"' in r.text[m2.start():m2.end()+15]:
                        d += dt.timedelta(hours=8)
                    return d
                return None
            if r.status_code in (404, 410):
                break
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    # Wayback 備援
    if timestamp and original_url:
        try:
            r = requests.get(f"http://web.archive.org/web/{timestamp}/{original_url}",
                             timeout=45, headers=HEADERS)
            if r.status_code == 200:
                m = re.search(
                    r'property="article:published_time"[^>]*content="(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})',
                    r.text)
                if m:
                    d = dt.datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}",
                                             "%Y-%m-%d %H:%M")
                    if "+08:00" not in r.text[m.start():m.start()+80]:
                        d += dt.timedelta(hours=8)
                    return d
        except Exception:
            pass
    return None


def load_done():
    if not os.path.exists(OUT):
        return set()
    return set(pd.read_parquet(OUT)["aid"].astype(int))


def main():
    frozen = pd.read_parquet(FROZEN)
    tagged = frozen[frozen["stock_codes"].str.len() > 0].copy()
    todo = tagged[~tagged["aid"].isin(load_done())]
    print(f"有標記文章: {len(tagged):,} 篇, 待補時間: {len(todo):,}")

    t0 = time.time()
    n, got = 0, 0
    rows = []

    def work(r):
        snap = r["snapshot"] if pd.notna(r.get("snapshot")) else None
        orig = f"https://news.cnyes.com/news/id/{int(r['aid'])}"
        dtw = fetch_publish_time(int(r["aid"]), snap, orig)
        return (int(r["aid"]), dtw)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, r) for _, r in todo.iterrows()]
        for fut in as_completed(futs):
            aid, dtw = fut.result()
            n += 1
            if dtw is not None:
                got += 1
                pre = (dtw.hour < CLOSE_H) or (dtw.hour == CLOSE_H and dtw.minute <= CLOSE_M)
                rows.append({"aid": aid, "publish_dt": dtw.strftime("%Y-%m-%d %H:%M"),
                             "date": dtw.strftime("%Y-%m-%d"), "pre_close": pre})
            if n % 200 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(todo)}, 有時間 {got} ({got/n:.0%}), "
                      f"{n/el:.1f} 篇/s, 剩 {(len(todo)-n)/(n/el)/3600:.2f}h", flush=True)
                _save(rows)

    _save(rows, final=True)
    print(f"\n完成: {got}/{n} ({got/max(n,1):.0%}) 有時間戳, 耗時 {(time.time()-t0)/60:.0f} 分")


def _save(rows, final=False):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if os.path.exists(OUT):
        old = pd.read_parquet(OUT)
        df_all = (pd.concat([old, df_new], ignore_index=True)
                    .drop_duplicates(subset=["aid"], keep="last"))
    else:
        df_all = df_new
    df_all.to_parquet(OUT, index=False)


if __name__ == "__main__":
    main()