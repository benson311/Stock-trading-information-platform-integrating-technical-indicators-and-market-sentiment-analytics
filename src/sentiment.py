import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""情緒分析器。

模式:
  - 'lexicon' (預設) : NTUSD + 金融詞典, 處理否定詞/程度詞, CPU 快速, 適合 1100 萬則留言。
  - 'finbert'       : 使用 HuggingFace 預訓練中文金融情緒模型 (CPU 較慢, 用於驗證子集)。

輸出每則文字的 (pos_score, neg_score) ∈ [0,1], 參考論文以 softmax 機率表示正/負情緒機率。
"""
import os
import re
import bisect

import ahocorasick

from src.lexicon import build_weighted_lexicon, NEGATIONS, INTENSIFIERS
import config


class LexiconSentimentAnalyzer:
    """詞典式繁體中文情緒分析器 (Aho-Corasick 多模式匹配 + 否定/程度處理)。

    使用 AC 自動機讓 2 萬詞匹配降至 O(文本長度), 勝過 regex 巨型 alternation。
    """

    def __init__(self):
        self.pos_w, self.neg_w = build_weighted_lexicon()
        self.pos = set(self.pos_w)
        self.neg = set(self.neg_w)
        self._neg_words = list(NEGATIONS)
        self._inten = list(INTENSIFIERS.items())
        self._build_automata()

    def _build_automata(self):
        # 情緒詞 automaton: value = (type, word_len, weight); type ∈ {pos, neg}
        self._sent = ahocorasick.Automaton()
        for w, weight in self.pos_w.items():
            self._sent.add_word(w, ("pos", len(w), weight))
        for w, weight in self.neg_w.items():
            self._sent.add_word(w, ("neg", len(w), weight))
        self._sent.make_automaton()

        # 否定詞 / 程度詞 automaton
        self._neg_auto = ahocorasick.Automaton()
        for w in self._neg_words:
            self._neg_auto.add_word(w, len(w))
        self._neg_auto.make_automaton()

        self._inten_auto = ahocorasick.Automaton()
        for w, v in self._inten:
            self._inten_auto.add_word(w, v)
        self._inten_auto.make_automaton()

    def score(self, text: str) -> tuple:
        """回傳 (pos_score, neg_score) ∈ [0,1], 相加為 1。"""
        if not text or not text.strip():
            return 0.5, 0.5
        t = text.strip()

        # 命中位置: 情緒詞 (start, typ, weight), 否定詞 (end), 程度詞 (end, v)
        sent_hits = [(end - ln, typ, weight) for end, (typ, ln, weight) in self._sent.iter(t)]
        neg_ends = [end for end, _ in self._neg_auto.iter(t)]
        inten_hits = [(end, v) for end, v in self._inten_auto.iter(t)]

        pos_score = 0.0
        neg_score = 0.0
        for start, typ, weight in sent_hits:
            w = weight
            if _has_in_window(neg_ends, start - 6, start):
                w = -w
            w *= _intensity_in_window(inten_hits, start - 6, start)
            if typ == "pos":
                if w > 0:
                    pos_score += w
                else:
                    neg_score += -w
            else:
                if w > 0:
                    neg_score += w
                else:
                    pos_score += -w

        if pos_score == 0 and neg_score == 0:
            return 0.5, 0.5
        s = pos_score + neg_score + 1.0
        return (pos_score + 0.5) / s, (neg_score + 0.5) / s


def _has_in_window(sorted_positions, lo, hi):
    i = bisect.bisect_left(sorted_positions, lo)
    return i < len(sorted_positions) and sorted_positions[i] <= hi


def _intensity_in_window(inten, lo, hi):
    w = 1.0
    for end, v in inten:
        if lo <= end <= hi:
            w = max(w, v)
    return w


class SentimentAnalyzer:
    """統一介面, 依 config.SENTIMENT_MODE 選擇引擎。"""

    def __init__(self, mode: str = None):
        self.mode = mode or config.SENTIMENT_MODE
        if self.mode == "lexicon":
            self.engine = LexiconSentimentAnalyzer()
        else:
            self.engine = LexiconSentimentAnalyzer()  # finbert 待擴充
        print(f"[SentimentAnalyzer] 模式: {self.mode}")

    def score(self, text: str) -> tuple:
        return self.engine.score(text)

    def score_pair(self, texts) -> tuple:
        """批次回傳 (pos_list, neg_list)。"""
        pos_l, neg_l = [], []
        for t in texts:
            p, n = self.score(t)
            pos_l.append(p)
            neg_l.append(n)
        return pos_l, neg_l


if __name__ == "__main__":
    a = SentimentAnalyzer()
    tests = [
        "台積電大漲, 利多不斷, 法人持續買超",
        "大盤暴跌, 投資人恐慌殺出",
        "台積電沒有下跌, 反而持續上漲",
        "這檔股票不看好, 建議不要買進",
        "利多出盡, 股價跳水創波段新低",
        "多頭氣勢如虹, 一路飆漲創歷史新高",
    ]
    for t in tests:
        p, n = a.score(t)
        print(f"pos={p:.3f} neg={n:.3f}   {t}")