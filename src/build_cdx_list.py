"""Phase 1: 建置鉅亨網 Wayback Machine CDX 存檔清單。

向 web.archive.org CDX API 分頁拉取所有 news.cnyes.com/news/* 存檔網址,
本地分類 (舊式 id 文章 / 新式 slug 文章 / aigc / 其他), 去重後存 parquet。
輸出: data/processed/cdx_cnyes_list.parquet
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import re
import time
import urllib.parse

import pandas as pd
import requests

import config

CDX_API = "http://web.archive.org/cdx/search/cdx"
PATTERN = "news.cnyes.com/news/*"
OUT_PATH = os.path.join(config.PROCESSED_DIR, "cdx_cnyes_list.parquet")

COLS = ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]


def fetch_cdx_page(pattern, resume_key=None, limit=15000, max_retry=4):
    """抓一頁 CDX 結果 (含 resumeKey 分頁支援)。"""
    base = {
        "url": pattern,
        "output": "json",
        "collapse": "urlkey",
        "limit": limit,
        "showResumeKey": "true",
    }
    qs = urllib.parse.urlencode(base)
    qs += "&filter=statuscode:200&filter=mimetype:text/html"
    if resume_key:
        qs += "&resumeKey=" + urllib.parse.quote(resume_key)

    for attempt in range(max_retry):
        try:
            r = requests.get(f"{CDX_API}?{qs}", timeout=180,
                             headers={"User-Agent": "research-crawler/grad-project"})
            if r.status_code == 200 and r.text.strip():
                return r.json()
            print(f"  HTTP {r.status_code} (attempt {attempt+1})")
        except Exception as e:
            print(f"  請求失敗 (attempt {attempt+1}): {type(e).__name__}")
        time.sleep(8 * (attempt + 1))
    return None


def fetch_all(pattern, limit=15000, checkpoint_path=None):
    """分頁拉取全部 CDX 列表直到沒有 resumeKey。"""
    rows = []
    resume = None
    page = 0
    t0 = time.time()
    while True:
        data = fetch_cdx_page(pattern, resume_key=resume, limit=limit)
        if data is None:
            print("CDX 頁面抓取失敗 (重試後仍失敗), 以目前已取得資料繼續")
            break

        new_rows = []
        new_resume = None
        for item in data[1:]:
            if not item:            # [] 分隔列
                continue
            if len(item) == 1:      # resumeKey 列
                new_resume = item[0]
                continue
            if len(item) >= 7:
                new_rows.append(item[:7])

        rows.extend(new_rows)
        page += 1
        el = time.time() - t0
        print(f"  page {page}: +{len(new_rows)} 列 (累計 {len(rows)}, 耗時 {el/60:.1f} 分)")

        if checkpoint_path and rows:
            pd.DataFrame(rows, columns=COLS).to_parquet(checkpoint_path, index=False)

        if new_resume is None:
            print("  已到最後一頁")
            break
        resume = new_resume
        time.sleep(2)
    return rows


def classify_url(url: str) -> str:
    """分類存檔網址。"""
    try:
        path = urllib.parse.urlparse(url).path
    except Exception:
        return "other"
    # 舊式文章: /news/id/{數字}
    if re.match(r"^/news/id/\d+$", path):
        return "old_article"
    # AI 生效內容
    if path.startswith("/news/aigc/"):
        return "aigc"
    # 分類/列表/標籤頁 (非文章)
    junk_prefixes = ("/news/cat/", "/news/cat_", "/news/list", "/news/tag",
                     "/news/search", "/news/author", "/news/topic", "/news/projects",
                     "/news/section", "/news/tw-stock", "/news/us-stock")
    for j in junk_prefixes:
        if path.startswith(j):
            return "junk"
    # 新式文章: /news/{slug} (1~2 段中文 slug)
    segs = [s for s in path.split("/") if s]
    if len(segs) >= 2 and segs[0] == "news":
        if len(segs) == 2:
            return "new_article"
        if len(segs) == 3 and "%" in urllib.parse.quote(segs[-1]):
            return "new_article"
        if len(segs) == 3:
            return "new_article"
    return "other"


def main():
    print("=== Phase 1: 鉅亨網 CDX 存檔清單建置 ===")
    print(f"查詢樣式: {PATTERN}")

    rows = fetch_all(PATTERN, limit=15000, checkpoint_path=OUT_PATH)
    if not rows:
        print("沒有取得任何資料!")
        return

    df = pd.DataFrame(rows, columns=COLS)
    df = df.drop_duplicates(subset=["original"]).reset_index(drop=True)
    df["kind"] = df["original"].map(classify_url)
    df["snapshot_date"] = df["timestamp"].str[:8]

    df.to_parquet(OUT_PATH, index=False)

    print(f"\n=== 統計 ===")
    print(f"總列數 (去重後): {len(df)}")
    print("\n類型分布:")
    print(df["kind"].value_counts().to_string())
    art = df[df["kind"].isin(["old_article", "new_article"])]
    if len(art):
        art_years = art["snapshot_date"].str[:4].value_counts().sort_index()
        print(f"\n文章頁數量: {len(art)}")
        print("各年份 (快照) 分布:")
        print(art_years.to_string())
    print(f"\n已存 -> {OUT_PATH}")


if __name__ == "__main__":
    main()