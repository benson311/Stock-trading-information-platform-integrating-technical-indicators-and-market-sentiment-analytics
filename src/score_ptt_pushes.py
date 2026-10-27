import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""對 PTT 留言評分 (push_text), 存出 ptt_pushes_scored.parquet。"""
import os
import time

import pandas as pd

import config
from src.score_utils import score_column


def main():
    pushes = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "ptt_pushes_mapped.parquet"))
    print(f"留言數: {len(pushes)}")

    t0 = time.time()
    pos, neg = score_column(pushes["push_text"])
    print(f"評分完成, 耗時 {(time.time()-t0)/60:.1f} 分鐘")

    pushes["pos_score"] = pos
    pushes["neg_score"] = neg
    pushes["sentiment"] = pos - neg
    out = os.path.join(config.PROCESSED_DIR, "ptt_pushes_scored.parquet")
    pushes.to_parquet(out, index=False)
    print(f"已存 {out}, 平均 pos={pos.mean():.3f} neg={neg.mean():.3f}")


if __name__ == "__main__":
    main()