"""建構鉅亨網新聞情緒 (靜態)，整合進情緒指數。

論文式: News_score_t = Σ news_score_i,t
新聞有時間戳，可做時間序列；這裡因範圍僅 60 天，先做逐日情緒。
"""
import os

import pandas as pd

import config
from src.score_utils import score_column
from src.sentiment import SentimentAnalyzer


def build_cnyes_news_sentiment():
    """對鉅亨網新聞評分，產生逐日個股新聞情緒指數"""
    news = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "cnyes_news.parquet"))
    print(f"新聞總數: {len(news)}")

    # 爆炸 stock_codes (一則新聞可能提及多檔)
    news_exp = news.explode("stock_codes").rename(columns={"stock_codes": "code"}).dropna(subset=["code"])
    print(f"展開後: {len(news_exp)} 筆")

    # 文字 = title + summary (內容較長，摘要較精簡)
    news_exp["text"] = news_exp["title"].fillna("") + " " + news_exp["summary"].fillna("")

    # 評分
    analyzer = SentimentAnalyzer(mode="lexicon")
    pos, neg = score_column(news_exp["text"])
    news_exp["news_pos"] = pos
    news_exp["news_neg"] = neg
    news_exp["news_net"] = pos - neg

    # 逐日、逐股加總
    idx = (
        news_exp.groupby(["code", "date"], as_index=False)
        .agg(
            news_pos=("news_pos", "sum"),
            news_neg=("news_neg", "sum"),
            news_count=("news_pos", "count"),
        )
    )
    idx["news_net"] = (idx["news_pos"] - idx["news_neg"]) / (idx["news_pos"] + idx["news_neg"] + 1e-6)

    out = os.path.join(config.PROCESSED_DIR, "cnyes_news_sentiment.parquet")
    idx.to_parquet(out, index=False)
    print(f"鉅亨網新聞情緒: {len(idx)} 列 -> {out}")
    return idx


if __name__ == "__main__":
    build_cnyes_news_sentiment()