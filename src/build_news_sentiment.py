import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""新聞標題情緒評分 (現有新聞無時間戳, 產出「個股靜態新聞情緒」)。

論文式(3): News_score = Σ news_score_i。因無日期, 聚合為每股平均新聞情緒。
"""
import os
import glob

import pandas as pd

import config
from src.score_utils import score_column
from src.lexicon import build_lexicon


def build_news_sentiment():
    rows = []
    codes = [d for d in os.listdir(config.RAW_STOCKS_DIR)
             if os.path.isdir(os.path.join(config.RAW_STOCKS_DIR, d))]
    for code in sorted(codes):
        nf = os.path.join(config.RAW_STOCKS_DIR, code, "news")
        if not os.path.isdir(nf):
            continue
        for fn in os.listdir(nf):
            df = pd.read_csv(os.path.join(nf, fn), encoding="utf-8")
            tcol = [c for c in df.columns if c.lstrip("\ufeff").strip() == "title"]
            if not tcol:
                continue
            titles = df[tcol[0]].astype(str).tolist()
            for t in titles:
                rows.append((code, t))

    if not rows:
        print("無新聞資料")
        return None

    news = pd.DataFrame(rows, columns=["code", "title"])
    print(f"新聞標題數: {len(news)}")

    pos, neg = score_column(news["title"])
    news["news_pos"] = pos
    news["news_neg"] = neg

    # 個股靜態新聞情緒
    static = (
        news.groupby("code", as_index=False)
        .agg(news_pos=("news_pos", "mean"), news_neg=("news_neg", "mean"),
             news_count=("title", "count"))
    )
    static["news_net"] = static["news_pos"] - static["news_neg"]

    out = os.path.join(config.PROCESSED_DIR, "news_sentiment_static.parquet")
    static.to_parquet(out, index=False)
    print(f"個股新聞情緒: {len(static)} 檔 -> {out}")
    return static


if __name__ == "__main__":
    build_news_sentiment()