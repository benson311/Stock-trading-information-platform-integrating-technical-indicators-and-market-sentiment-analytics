import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""對 PTT 文章評分 (title + body), 存出 ptt_articles_scored.parquet。"""
import os
import time

import pandas as pd

import config
from src.score_utils import score_column


def main():
    arts = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "ptt_articles_mapped.parquet"))
    print(f"文章數: {len(arts)}")

    text = arts["title"].fillna("") + "\n" + arts["body"].fillna("")
    t0 = time.time()
    pos, neg = score_column(text)
    print(f"評分完成, 耗時 {(time.time()-t0)/60:.1f} 分鐘")

    arts["pos_score"] = pos
    arts["neg_score"] = neg
    arts["sentiment"] = pos - neg  # 情緒極性 [-1,1]
    out = os.path.join(config.PROCESSED_DIR, "ptt_articles_scored.parquet")
    arts.to_parquet(out, index=False)
    print(f"已存 {out}, 平均 pos={pos.mean():.3f} neg={neg.mean():.3f}")


if __name__ == "__main__":
    main()