import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""FinBERT (中文金融情緒 BERT) 情緒評分 — 論文原始方法對照。

使用 yiyanghkust/finbert-tone-chinese (3 類: Positive/Neutral/Negative)。
因模型為簡體訓練, 先以 opencc 將繁中轉簡中。
CPU 環境下建議對「驗證子集」使用, 並與詞典法比較。
"""
import warnings
import torch
from opencc import OpenCC

import config

_LABELS = {0: "Neutral", 1: "Positive", 2: "Negative"}


class FinBertSentimentAnalyzer:
    def __init__(self, model_id: str = None):
        self.model_id = model_id or "yiyanghkust/finbert-tone-chinese"
        self._cc = OpenCC("t2s")
        self._tok = None
        self._model = None
        self._device = torch.device("cpu")

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"[FinBERT] 載入 {self.model_id} ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.to(self._device)
        self._model.eval()
        print("[FinBERT] 載入完成")

    def _to_simp(self, text: str) -> str:
        return self._cc.convert(text)

    def score(self, text: str) -> tuple:
        """回傳 (pos_score, neg_score) (neutral 依比例分配)。"""
        self._load()
        if not text or not text.strip():
            return 0.5, 0.5
        simp = self._to_simp(text)
        enc = self._tok(simp, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        p_pos = probs[1].item()
        p_neg = probs[2].item()
        p_neu = probs[0].item()
        # 中性按比例融入正負, 維持 pos+neg=1
        s = p_pos + p_neg + 1e-9
        return p_pos / (s) * (1 - p_neu) + p_neu / 2, p_neg / s * (1 - p_neu) + p_neu / 2

    def score_batch(self, texts, batch_size: int = 32) -> list:
        """批次評分, 回傳 list[(pos, neg)]。"""
        self._load()
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            simp = [self._to_simp(t) if t else "" for t in batch]
            enc = self._tok(simp, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            for row in probs:
                p_pos = row[1].item()
                p_neg = row[2].item()
                p_neu = row[0].item()
                s = p_pos + p_neg + 1e-9
                out.append(((p_pos / s) * (1 - p_neu) + p_neu / 2,
                            (p_neg / s) * (1 - p_neu) + p_neu / 2))
        return out


if __name__ == "__main__":
    a = FinBertSentimentAnalyzer()
    tests = [
        "台積電大漲，利多不斷，法人持續買超",
        "大盤暴跌，投資人恐慌殺出",
        "這檔股票不看好，建議不要買進",
    ]
    for t in tests:
        print(a.score(t), t)