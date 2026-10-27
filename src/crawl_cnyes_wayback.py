"""Phase 2A: 鉅亨網 5 年歷史文章回填爬蟲。

策略: 官網完整保留所有舊文章 (實測 id/4000034 = 2017 年文章可直接抓),
故優先直接爬官網 news.cnyes.com/news/id/{aid} (快, ~0.5s/篇),
失敗時退回 Wayback Machine 存檔頁。

特性:
  - 斷點續傳 (每 50 篇存檔, 重跑自動跳過已完成)
  - 禮貌並行 (6 workers), 失敗退避重試
  - 新文章優先 (aid 由大到小)
  - 支援新舊版面解析 (meta tags / JSON-LD / h1 / main-content)

用法:
  python3 -m src.crawl_cnyes_wayback benchmark   # 測速 20 篇
  python3 -m src.crawl_cnyes_wayback run [N]     # 執行 (可選只跑前 N 篇)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import re
import sys
import json
import time
import html as htmllib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import config
from src.news_matcher import build_news_matcher
from src.map_ptt_to_stocks import build_matchers, load_stock_meta

# -------------------- 參數 --------------------
CDX_LIST = os.path.join(config.PROCESSED_DIR, "cdx_cnyes_list.parquet")
OUT_PATH = os.path.join(config.PROCESSED_DIR, "cnyes_news_raw.parquet")
ID_MIN = 4_600_000          # 近 5 年 (2021+) 文章 ID 門檻
WORKERS = 6
CHECKPOINT_EVERY = 50
BENCH_N = 20
MIN_LEN = 2500

HEADERS = {"User-Agent": "research-crawler/grad-project (contact: student)"}


# -------------------- 網頁請求 --------------------
LIVE_BASE = "https://news.cnyes.com/news/id/{}"

def fetch_article(aid: int, timestamp: str = None, original_url: str = None,
                  max_retry=3) -> str | None:
    """抓文章頁: 優先官網 (快), 失敗退回 Wayback 存檔。"""
    for attempt in range(max_retry):
        try:
            r = requests.get(LIVE_BASE.format(aid), timeout=30, headers=HEADERS)
            if r.status_code == 200 and len(r.content) > 5000:
                r.encoding = "utf-8"
                return r.text
            if r.status_code in (404, 410):
                break  # 確定不存在, 直接走 Wayback
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    # Wayback 備援
    if timestamp and original_url:
        wayback_url = f"http://web.archive.org/web/{timestamp}/{original_url}"
        for attempt in range(2):
            try:
                r = requests.get(wayback_url, timeout=60, headers=HEADERS)
                if r.status_code == 200 and len(r.content) > 3000:
                    r.encoding = r.apparent_encoding or "utf-8"
                    return r.text
                if r.status_code in (404, 410):
                    return None
            except Exception:
                pass
            time.sleep(3 * (attempt + 1))
    return None


# -------------------- 解析 --------------------
def _clean_text(s: str) -> str:
    s = re.sub(r"<script[^>]*>.*?</script>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<style[^>]*>.*?</style>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    # 雙重轉義實體 (&#10; 等) 需解兩次
    for _ in range(2):
        prev = s
        s = htmllib.unescape(s)
        if s == prev:
            break
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _utc_to_tw_date(iso: str) -> str:
    """ISO 時間 (UTC) → 台灣時區日期 (UTC+8)。"""
    import datetime as dt
    m = re.match(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})", iso)
    if not m:
        return iso[:10]
    d = dt.datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}", "%Y-%m-%d %H:%M")
    if iso.rstrip().endswith("Z") or "+" not in iso[10:]:
        d += dt.timedelta(hours=8)  # UTC → Asia/Taipei
    return d.strftime("%Y-%m-%d")


def parse_article(html: str) -> dict:
    """從文章頁解析 (date, title, content)。

    策略 (新版官網有乾淨的 JSON-LD, 優先):
      1. JSON-LD NewsArticle: articleBody / datePublished / headline
      2. meta tags (Wayback 舊頁): article:published_time / og:title
      3. h1 / main-content fallback
    """
    out = {"date": None, "title": None, "content": None, "date_approx": False}

    # ---- 1) JSON-LD (官網新版, 內文零雜訊) ----
    for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>',
                            html, flags=re.S):
        try:
            data = json.loads(block)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue
            body = (it.get("articleBody") or "").strip()
            title = (it.get("headline") or it.get("name") or "").strip()
            if body or title:
                out["content"] = _clean_text(body)[:3000] or None
                out["title"] = _clean_text(title) or None
                dp = it.get("datePublished") or it.get("dateModified") or ""
                if dp:
                    out["date"] = _utc_to_tw_date(dp)
                return out

    # ---- 2) meta tags (Wayback 舊版 React 頁) ----
    m = re.search(r'property="article:published_time"[^>]*content="([^"]+)"', html)
    if not m:
        m = re.search(r'content="([^"]+)"[^>]*property="article:published_time"', html)
    if not m:
        m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
    if m:
        out["date"] = _utc_to_tw_date(m.group(1))

    m = re.search(r'property="og:title"[^>]*content="([^"]+)"', html)
    if not m:
        m = re.search(r"<h1[^>]*>([^<]{6,150})</h1>", html)
    if not m:
        m = re.search(r"<title>([^<]{6,150})</title>", html)
    if m:
        t = _clean_text(m.group(1))
        t = re.sub(r"\s*[|｜–-]\s*鉅亨網.*$", "", t)
        t = re.sub(r"\s*[|｜–-]\s*anue.*$", "", t, flags=re.I)
        out["title"] = t.strip() or None

    # ---- 3) 內文 fallback ----
    m = re.search(r'class="main-content"(.*?)(?:class="_1mDE|$)', html, flags=re.S)
    if m:
        out["content"] = _clean_text(m.group(1))[:3000] or None
    if not out["content"] or len(out["content"]) < 80:
        m = re.search(r'property="og:description"[^>]*content="([^"]+)"', html)
        if m:
            out["content"] = _clean_text(m.group(1))[:3000] or None

    return out



# -------------------- 清單建置 --------------------
N_WAVES = 16   # 波次數: 每波均勻覆蓋 5 年, 波0 完成即有 1/16 密度 (~17篇/日)


def build_crawl_list(id_min=ID_MIN, min_len=MIN_LEN, wave=None):
    df = pd.read_parquet(CDX_LIST)
    old = df[df["kind"] == "old_article"].copy()
    old["aid"] = old["original"].map(
        lambda u: int(re.search(r"/news/id/(\d+)", u).group(1)))
    old["length"] = old["length"].astype(int)

    # 真實流水號範圍: 目前最新約 6.6M; 超過 10M 是錯誤網址
    old = old[(old["aid"] >= id_min) & (old["aid"] <= 10_000_000)]
    old = old[old["length"] >= min_len]

    # 每篇文章取最佳快照 (優先 print 版, 其次最大 length)
    old["is_print"] = old["original"].str.contains("/print")
    best = (old.sort_values(["is_print", "length"], ascending=[False, False])
               .groupby("aid", as_index=False)
               .first()[["aid", "timestamp", "original", "length"]])
    # 由舊到新排序 (訓練期優先), 波次取 stride 均勻切片
    best = best.sort_values("aid").reset_index(drop=True)
    if wave is not None:
        best = best.iloc[wave::N_WAVES].reset_index(drop=True)
    return best


# -------------------- 主流程 --------------------
def load_done() -> set:
    if not os.path.exists(OUT_PATH):
        return set()
    df = pd.read_parquet(OUT_PATH, columns=["aid"])
    return set(df["aid"].astype(int))


def crawl_one(row, matcher) -> dict | None:
    html = fetch_article(int(row["aid"]), row["timestamp"], row["original"])
    if html is None:
        return None
    parsed = parse_article(html)
    if not parsed["title"] and not parsed["content"]:
        return None

    text = (parsed["title"] or "") + "\n" + (parsed["content"] or "")
    codes = []
    code_pattern, name_pattern, code_to_name, name_to_code = matcher
    for m in code_pattern.findall(text):
        c = m if isinstance(m, str) else m[0]
        if c not in codes:
            codes.append(c)
    for m in name_pattern.findall(text):
        c = name_to_code.get(m if isinstance(m, str) else m[0])
        if c and c not in codes:
            codes.append(c)

    # 日期 fallback: 用快照日期近似
    date = parsed["date"]
    approx = False
    if not date:
        ts = row["timestamp"]
        date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        approx = True

    return {
        "aid": int(row["aid"]),
        "date": date,
        "date_approx": approx,
        "title": parsed["title"],
        "content": parsed["content"],
        "stock_codes": ",".join(codes),
        "snapshot": row["timestamp"],
        "source": "cnyes_wayback",
    }


def run(limit=None, workers=WORKERS, benchmark=False, waves=None):
    """依波次執行: wave0 完成後全 5 年均有 1/16 密度覆蓋, 依序加密至全量。"""
    n_target = BENCH_N if benchmark else limit
    wave_seq = waves if waves is not None else list(range(N_WAVES))

    matcher = build_news_matcher()
    t_all = time.time()

    for wave in wave_seq:
        crawl_list = build_crawl_list(wave=wave)
        done = set() if benchmark else load_done()
        todo = crawl_list[~crawl_list["aid"].isin(done)]
        if n_target:
            todo = todo.head(n_target)
        if len(todo) == 0:
            print(f"[wave {wave}] 已完成, 跳過")
            continue
        print(f"\n[wave {wave}/{N_WAVES}] 本波 {len(crawl_list)} 篇, "
              f"待抓 {len(todo)} (累計完成 {len(done)})")

        results = []
        t0 = time.time()
        done_count = 0

        def work(i_row):
            _, row = i_row
            try:
                return crawl_one(row, matcher)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(work, (i, row)): i for i, row in todo.iterrows()}
            for fut in as_completed(futures):
                r = fut.result()
                done_count += 1
                if r:
                    results.append(r)
                if done_count % 50 == 0:
                    el = time.time() - t0
                    speed = done_count / el
                    remain = (len(todo) - done_count) / max(speed, 1e-9) / 3600
                    print(f"  {done_count}/{len(todo)}, 成功 {len(results)}, "
                          f"{speed:.2f} 篇/s, 本波剩 {remain:.1f}h", flush=True)
                    _save(results, benchmark)

        _save(results, benchmark, final=True)
        el = time.time() - t0
        rate = done_count / max(el, 1e-9)
        print(f"[wave {wave}] 完成: +{len(results)} 篇 "
              f"({done_count} 嘗試, {el/60:.0f} 分, {rate:.2f} 篇/s)")

    if not benchmark:
        total_done = len(load_done())
        full = build_crawl_list()
        print(f"\n全部波次結束: 累計 {total_done} / {full['aid'].nunique()} 篇, "
              f"總耗時 {(time.time()-t_all)/3600:.2f} 小時")


def _save(results, benchmark, final=False):
    if not results:
        return
    df_new = pd.DataFrame(results)
    if benchmark:
        path = OUT_PATH.replace(".parquet", "_bench.parquet")
        df_new.to_parquet(path, index=False)
        return
    # 增量合併
    if os.path.exists(OUT_PATH):
        old = pd.read_parquet(OUT_PATH)
        df_all = pd.concat([old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["aid"]).reset_index(drop=True)
    else:
        df_all = df_new
    df_all.to_parquet(OUT_PATH, index=False)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "benchmark":
        run(benchmark=True)
    else:
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run(limit=limit)