import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""情緒指數時間序列建置 (論文 Section 2.1.2, 式(2)-(4))。

St = w1 * News_score_t + w2 * Review_score_t

由於現有新聞標題無時間戳, 本系統以 PTT 文章 + 留言作為「投資人評論」
(Review) 的主要時間序列來源; 新聞則產出一檔「靜態個股新聞情緒」供參考。

輸出 (全體存為單一 parquet):
  - code, date, review_pos, review_neg, review_count, review_net
"""
import os

import pandas as pd

import config


def build_review_index():
    """由已評分的 PTT 文章與留言, 聚合為 (code, date) 層級情緒指數。"""
    arts = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "ptt_articles_scored.parquet"))
    pushes = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "ptt_pushes_scored.parquet"))

    art_df = arts[["date", "stock_codes", "pos_score", "neg_score"]].copy().assign(weight=2.0)
    push_df = pushes[["date", "stock_codes", "pos_score", "neg_score"]].copy().assign(weight=1.0)

    art_df = _explode_stocks(art_df)
    push_df = _explode_stocks(push_df)

    all_review = pd.concat([art_df, push_df], ignore_index=True)
    # 加權情緒分
    all_review["wpos"] = all_review["pos_score"] * all_review["weight"]
    all_review["wneg"] = all_review["neg_score"] * all_review["weight"]

    idx = (
        all_review.groupby(["code", "date"], as_index=False)
        .agg(
            review_pos=("wpos", "sum"),
            review_neg=("wneg", "sum"),
            review_count=("pos_score", "count"),
        )
    )
    idx["review_net"] = (idx["review_pos"] - idx["review_neg"]) / (idx["review_pos"] + idx["review_neg"] + 1e-6)
    idx = idx[["code", "date", "review_pos", "review_neg", "review_count", "review_net"]]

    out = os.path.join(config.PROCESSED_DIR, "sentiment_index_review.parquet")
    idx.to_parquet(out, index=False)
    print(f"個股日層級情緒指數: {len(idx)} 列 -> {out}")
    print(f"涵蓋 {idx['code'].nunique()} 檔, 日期 {idx['date'].min()} ~ {idx['date'].max()}")
    return idx


def _explode_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """將 stock_codes (list) 展開為每列一個 code。"""
    df = df.explode("stock_codes").rename(columns={"stock_codes": "code"})
    df = df.dropna(subset=["code"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    build_review_index()