"""重建情緒指數：結合 PTT 評論 + 鉅亨網新聞 (論文式 St = w1*News + w2*Review)"""
import os

import pandas as pd

import config


def build_combined_sentiment_index(w1=0.4, w2=0.6):
    """結合新聞情緒 (w1) + 評論情緒 (w2)"""
    # 1. PTT 評論情緒
    review = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "sentiment_index_review.parquet"))
    review = review.rename(columns={
        "review_pos": "review_pos_raw",
        "review_neg": "review_neg_raw",
        "review_net": "review_net_raw",
        "review_count": "review_count_raw",
    })
    
    # 2. 鉅亨網新聞情緒
    news = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "cnyes_news_sentiment.parquet"))
    news = news.rename(columns={
        "news_pos": "news_pos_raw",
        "news_neg": "news_neg_raw",
        "news_net": "news_net_raw",
        "news_count": "news_count_raw",
    })
    
    # 合併
    combined = pd.merge(review, news, on=["code", "date"], how="outer")
    
    # 填補缺失 (無新聞/無評論的日子 = 中性)
    for col in ["review_pos_raw", "review_neg_raw", "review_count_raw",
                "news_pos_raw", "news_neg_raw", "news_count_raw"]:
        combined[col] = combined[col].fillna(0.0)
    
    # 加權合成 (論文式 St = w1*News + w2*Review)
    combined["pos"] = w1 * combined["news_pos_raw"] + w2 * combined["review_pos_raw"]
    combined["neg"] = w1 * combined["news_neg_raw"] + w2 * combined["review_neg_raw"]
    combined["net"] = (combined["pos"] - combined["neg"]) / (combined["pos"] + combined["neg"] + 1e-6)
    combined["count"] = combined["review_count_raw"] + combined["news_count_raw"]
    
    # 只保留需要的欄位
    out_cols = ["code", "date", "pos", "neg", "net", "count",
                "news_pos_raw", "news_neg_raw", "news_count_raw",
                "review_pos_raw", "review_neg_raw", "review_count_raw"]
    combined = combined[out_cols]
    
    out = os.path.join(config.PROCESSED_DIR, "sentiment_index_combined.parquet")
    combined.to_parquet(out, index=False)
    print(f"結合情緒指數: {len(combined)} 列 -> {out}")
    print(f"涵蓋 {combined['code'].nunique()} 檔, 日期 {combined['date'].min()} ~ {combined['date'].max()}")
    return combined


if __name__ == "__main__":
    build_combined_sentiment_index()