import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""批次情緒評分工具: 以多行程平行對文字欄位評分。"""
import multiprocessing as mp

import pandas as pd

from src.sentiment import SentimentAnalyzer


def _init_worker():
    global _ANALYZER
    _ANALYZER = SentimentAnalyzer(mode="lexicon")


def _score_batch(texts):
    global _ANALYZER
    pos_list, neg_list = [], []
    for t in texts:
        p, n = _ANALYZER.score(str(t))
        pos_list.append(p)
        neg_list.append(n)
    return pos_list, neg_list


def score_column(series: pd.Series, workers: int = None, chunksize: int = 2000):
    """對文字 Series 平行評分, 回傳 (pos_series, neg_series)。"""
    workers = workers or max(1, mp.cpu_count() - 2)
    texts = series.fillna("").tolist()
    pos_out = [0.0] * len(texts)
    neg_out = [0.0] * len(texts)

    chunks = [texts[i:i + chunksize] for i in range(0, len(texts), chunksize)]
    with mp.Pool(processes=workers, initializer=_init_worker) as pool:
        for ci, (pl, nl) in enumerate(pool.imap(_score_batch, chunks, chunksize=1)):
            off = ci * chunksize
            pos_out[off:off + len(pl)] = pl
            neg_out[off:off + len(nl)] = nl
    return pd.Series(pos_out, index=series.index), pd.Series(neg_out, index=series.index)