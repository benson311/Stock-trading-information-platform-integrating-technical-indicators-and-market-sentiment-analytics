import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""情緒法對照: 詞典法 vs FinBERT (論文原始方法)。

抽樣 N 則已評分的 PTT 文本, 以 FinBERT 重評, 計算一致性與相關係數。
"""
import os
import time

import pandas as pd

import config
from src.sentiment_finbert import FinBertSentimentAnalyzer


def compare(n: int = 1000, seed: int = 42):
    arts = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "ptt_articles_scored.parquet"))
    sample = arts[["title", "body", "pos_score", "neg_score", "sentiment"]].sample(n, random_state=seed)
    texts = (sample["title"].fillna("") + " " + sample["body"].fillna("").str[:200]).tolist()

    fb = FinBertSentimentAnalyzer()
    t0 = time.time()
    fb_scores = fb.score_batch(texts, batch_size=16)
    print(f"FinBERT 評分 {n} 則, 耗時 {(time.time()-t0)/60:.1f} 分鐘")

    fb_sent = [p - n for p, n in fb_scores]
    sample["finbert_sent"] = fb_sent
    sample["finbert_pos"] = [p for p, _ in fb_scores]
    sample["finbert_neg"] = [n for _, n in fb_scores]

    corr = sample["sentiment"].corr(sample["finbert_sent"])
    print(f"\n詞典 sentiment vs FinBERT sentiment 相關係數: {corr:.4f}")

    # 方向一致性 (正/負/中性)
    def sign(x, thr=0.1):
        return 1 if x > thr else (-1 if x < -thr else 0)
    agree = (sample["sentiment"].map(sign) == sample["finbert_sent"].map(sign)).mean()
    print(f"方向一致率 (正/負/中性): {agree:.3f}")

    sample.to_csv(os.path.join(config.RESULTS_DIR, "sentiment_method_comparison.csv"),
                  index=False, encoding="utf-8-sig")
    print(f"存 to results/sentiment_method_comparison.csv")
    return corr, agree


if __name__ == "__main__":
    compare()