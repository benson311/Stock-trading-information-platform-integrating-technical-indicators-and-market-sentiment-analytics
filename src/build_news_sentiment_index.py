"""Phase 3+4: 純新聞情緒語料建置 + 情緒指數 (完全不用 PTT)。

資料來源:
  1. cnyes_news_raw.parquet   — 鉅亨網 5 年文章 (波次爬取中, 已有部分)
  2. yahoo_news_dated.parquet — Yahoo 新聞 (已補時間戳, 逐股)

流程:
  1. 語料合併: (date, code, text) — 鉅亨展開 stock_codes, Yahoo 逐股
  2. 加權詞典評分 (Aho-Corasick, 0.7ms/篇)
  3. 聚合為 (code, date) 日情緒指數: news_pos, news_neg, news_count
  4. 產業情緒: TWSE 產業別同業加總 (排除自身)

輸出:
  data/processed/news_sentiment_index.parquet  (自身日情緒)
  data/processed/news_sector_sentiment.parquet (產業日情緒)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys
import json
import time

import pandas as pd

import config
from src.score_utils import score_column

OUT_INDEX = os.path.join(config.PROCESSED_DIR, "news_sentiment_index.parquet")
OUT_SECTOR = os.path.join(config.PROCESSED_DIR, "news_sector_sentiment.parquet")


def build_corpus(cnyes_path: str = None) -> pd.DataFrame:
    """合併兩來源為 (date, code, title, content) 語料表。

    cnyes_path: 可指定凍結快照 (避免與背景爬蟲同時讀寫)。
    """
    rows = []

    # 1) 鉅亨 (stock_codes 逗號分隔, 展開)
    if cnyes_path is None:
        cnyes_path = os.path.join(config.PROCESSED_DIR, "cnyes_news_raw.parquet")
    if os.path.exists(cnyes_path):
        cnyes = pd.read_parquet(cnyes_path)
        cnyes = cnyes.dropna(subset=["date"])
        cnyes = cnyes[cnyes["stock_codes"].str.len() > 0]
        print(f"鉅亨文章 (含個股標記): {len(cnyes):,}")
        for _, r in cnyes.iterrows():
            for code in str(r["stock_codes"]).split(","):
                if not code.strip():
                    continue
                rows.append({
                    "date": r["date"], "code": code.strip(),
                    "text": (r["title"] or "") + " " + (r["content"] or ""),
                    "source": "cnyes",
                    "weight": 1.0,   # 專業新聞
                })

    # 2) Yahoo (逐股 CSV)
    yahoo_path = os.path.join(config.PROCESSED_DIR, "yahoo_news_dated.parquet")
    if os.path.exists(yahoo_path):
        yh = pd.read_parquet(yahoo_path)
        yh = yh.dropna(subset=["date"])
        print(f"Yahoo 新聞 (有日期): {len(yh):,}")
        for _, r in yh.iterrows():
            rows.append({
                "date": r["date"], "code": r["code"],
                "text": r["title"] or "",
                "source": "yahoo",
                "weight": 0.8,   # 標題 only, 權重略低
            })

    corpus = pd.DataFrame(rows)
    print(f"語料合計: {len(corpus):,} 列 (文章×個股), "
          f"{corpus['code'].nunique()} 檔, "
          f"{corpus['date'].min()} ~ {corpus['date'].max()}")
    return corpus


def score_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    """加權詞典評分。"""
    t0 = time.time()
    pos, neg = score_column(corpus["text"])
    corpus["pos"] = pos
    corpus["neg"] = neg
    corpus["wpos"] = corpus["pos"] * corpus["weight"]
    corpus["wneg"] = corpus["neg"] * corpus["weight"]
    print(f"評分完成: {len(corpus):,} 篇, 耗時 {(time.time()-t0)/60:.1f} 分")
    return corpus


def build_index(corpus: pd.DataFrame) -> pd.DataFrame:
    """聚合為 (code, date) 日情緒指數。"""
    idx = (corpus.groupby(["code", "date"], as_index=False)
           .agg(news_pos=("wpos", "sum"),
                news_neg=("wneg", "sum"),
                news_count=("pos", "count")))
    idx["net"] = (idx["news_pos"] - idx["news_neg"]) / (
        idx["news_pos"] + idx["news_neg"] + 1e-6)
    idx.to_parquet(OUT_INDEX, index=False)
    print(f"新聞情緒指數: {len(idx):,} 列 (個股×日) -> {OUT_INDEX}")
    return idx


def build_sector(idx: pd.DataFrame) -> pd.DataFrame:
    """產業情緒: 同 TWSE 產業別其他股票加總 (排除自身)。"""
    meta = json.load(open(os.path.join(config.PROCESSED_DIR, "stock_metadata.json"),
                          encoding="utf-8"))
    df = idx[["code", "date", "news_pos", "news_neg"]].copy()
    df["industry"] = df["code"].map(lambda c: meta.get(c, {}).get("industry", "00"))

    ind = df.groupby(["industry", "date"], as_index=False).agg(
        ind_pos=("news_pos", "sum"), ind_neg=("news_neg", "sum"))
    df = df.merge(ind, on=["industry", "date"], how="left")
    df["sector_pos"] = df["ind_pos"] - df["news_pos"]
    df["sector_neg"] = df["ind_neg"] - df["news_neg"]
    out = df[["code", "date", "sector_pos", "sector_neg"]]
    out.to_parquet(OUT_SECTOR, index=False)
    print(f"產業情緒: {len(out):,} 列 -> {OUT_SECTOR}")
    return out


def main(cnyes_path: str = None):
    corpus = build_corpus(cnyes_path=cnyes_path)
    if corpus.empty:
        print("語料為空, 請先完成爬取")
        return
    corpus = score_corpus(corpus)
    idx = build_index(corpus)
    build_sector(idx)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    main(cnyes_path=src)