import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""PTT 資料解析器：將 251k 篇 Stock 板 JSON 轉為結構化 parquet。

輸出:
  - ptt_articles.parquet : 文章層級 (title + body)
  - ptt_pushes.parquet   : 留言層級 (每則 push 一列)
"""
import os
import json
import datetime as dt
import multiprocessing as mp

import pandas as pd

import config

_TW = dt.timezone(dt.timedelta(hours=8))  # 台灣時區


def _parse_one(args) -> tuple:
    """解析單一 JSON 檔, 回傳 (article_dict or None, pushes_list)。"""
    path, batch = args
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, []

    ts = data.get("timestamp")
    if not ts:
        return None, []
    try:
        dt_obj = dt.datetime.fromtimestamp(int(ts), tz=_TW)
    except (ValueError, OSError, OverflowError):
        dt_obj = dt.datetime.fromtimestamp(int(ts), tz=_TW)
    date = dt_obj.strftime("%Y-%m-%d")
    ts_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

    article = {
        "article_id": data.get("article_id", ""),
        "timestamp": int(ts),
        "date": date,
        "datetime": ts_str,
        "title": data.get("title", "") or "",
        "body": data.get("body", "") or "",
        "n_pushes": len(data.get("pushes", [])),
        "batch": batch,
    }

    pushes = []
    for i, p in enumerate(data.get("pushes", [])):
        if not isinstance(p, str):
            continue
        # 弱訊號: 前綴 推/噓/→ 為 PTT 的讚/倒讚/箭頭標記 (pttweb 鏡像大多已移除)
        tag = ""
        text = p.strip()
        for t in ("推", "噓", "→"):
            if text.startswith(t):
                tag = t
                text = text[1:].strip()
                break
        if not text:
            continue
        pushes.append({
            "article_id": data.get("article_id", ""),
            "push_index": i,
            "push_tag": tag,
            "push_text": text,
            "timestamp": int(ts),
            "date": date,
            "datetime": ts_str,
            "batch": batch,
        })

    return article, pushes


def _list_files() -> list:
    out = []
    for batch in config.PTT_BATCHES:
        p = os.path.join(config.RAW_PTT_DIR, batch)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            out.append((os.path.join(p, fn), batch))
    return out


def parse_ptt(workers: int = None):
    files = _list_files()
    print(f"待解析 PTT 檔案數: {len(files)}")

    workers = workers or max(1, mp.cpu_count() - 1)
    articles = []
    pushes = []
    n_done = 0

    with mp.Pool(processes=workers) as pool:
        for art, ps in pool.imap_unordered(_parse_one, files, chunksize=200):
            if art:
                articles.append(art)
            pushes.extend(ps)
            n_done += 1
            if n_done % 50000 == 0:
                print(f"  ...已處理 {n_done}/{len(files)}")

    art_df = pd.DataFrame(articles)
    push_df = pd.DataFrame(pushes)

    art_path = os.path.join(config.PROCESSED_DIR, "ptt_articles.parquet")
    push_path = os.path.join(config.PROCESSED_DIR, "ptt_pushes.parquet")

    art_df.to_parquet(art_path, index=False)
    push_df.to_parquet(push_path, index=False)

    print(f"文章數: {len(art_df)} -> {art_path}")
    print(f"留言數: {len(push_df)} -> {push_path}")
    print(f"時間範圍: {art_df['date'].min()} ~ {art_df['date'].max()}")


if __name__ == "__main__":
    parse_ptt()