"""台灣股市情緒分析與預測系統 — 全域設定。

參考論文: Fang, K. & Zhang, Y. (2024). "A BERT-Based Multi-Source Sentiment Analysis and
Technical Analysis Model for Stock Price Prediction" (BERT-LLA), Mathematics 12, 1572.

台灣股市特性調整:
  - 交易時間 09:00–13:30 (UTC+8, 台灣時區)
  - 單日漲跌幅限制 ±10% (上市/上櫃普通股)
  - T+2 交割 (今日成交, 後二營業日交割)
  - 證券交易稅: 賣出 0.3%; 手續費: 買賣各 0.1425% (券商可談, 預設值)
  - 民國年 (ROC) 日曆 → 西元年轉換
"""
import os

# 專案根目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始資料路徑 (使用者提供的資料)
RAW_DIR = os.path.dirname(BASE_DIR)  # 新資料試跑/
RAW_STOCKS_DIR = os.path.join(RAW_DIR, "stocks")
RAW_TWSE_DIR = os.path.join(RAW_DIR, "twse", "stock_day")
RAW_T86_DIR = os.path.join(RAW_DIR, "twse", "t86")          # 三大法人買賣超日報 (T86)
RAW_MARGIN_DIR = os.path.join(RAW_DIR, "twse", "mi_margn")  # 融資融券餘額 (MI_MARGN)
RAW_PTT_DIR = os.path.join(RAW_DIR, "ptt raw")

# 處理後資料路徑
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LEXICON_DIR = os.path.join(DATA_DIR, "lexicon")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for _d in (PROCESSED_DIR, LEXICON_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# -------------------- 台灣股市特性 --------------------
TZ_TW = "Asia/Taipei"
SETTLEMENT_DAYS = 2            # T+2 交割
PRICE_LIMIT = 0.10             # 單日漲跌停 ±10%
TAX_RATE = 0.003               # 證券交易稅 (賣出) 0.3%
FEE_RATE = 0.001425            # 手續費 買賣各 0.1425%
INITIAL_CAPITAL = 1_000_000.0  # 初始資金 (與論文一致, 新台幣)
ROUND_LOT = 1000               # 台灣整股一張 1000 股

# -------------------- PTT 資料 --------------------
# 10 個 batch 目錄 (位於 ptt raw/ 下)
PTT_BATCHES = [
    "pttweb/batch-001",
    "pttweb 2/batch-002",
    "pttweb 3/batch-003",
    "pttweb 4/batch-004",
    "pttweb 5/batch-005",
    "pttweb 6/batch-006",
    "pttweb 7/batch-007",
    "pttweb 8/batch-008",
    "pttweb 9/batch-009",
    "pttweb 10/batch-010",
]

# -------------------- 情緒指數 (論文式(2): St = w1*News + w2*Review) --------------------
W_NEWS = 0.5      # 新聞權重 w1
W_REVIEW = 0.5    # 評論權重 w2 (論文最佳約 w1=0.4, w2=0.1, 此為預設)

# -------------------- 技術指標 (論文 Table 1, 12 個) --------------------
TECH_INDICATORS = {
    "trend": ["MA10", "MA20", "EMA10", "EMA20", "MACD_6_15_6", "MACD_12_26_9"],
    "momentum": ["RSI6", "RSI12", "WILLR14", "MOM14"],
    "volume": ["OBV", "ADOSC_3_10"],
}

# -------------------- 預測模型 (論文 Section 3.2.1) --------------------
LSTM1_UNITS = 64
LSTM2_UNITS = 32
DROPOUT = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 200            # 論文設 1000; CPU 環境下預設降低
SENTIMENT_LR = 2e-5     # BERT 微調學習率
WINDOW_SIZE = 20        # 滑動窗長度 h (交易日)
FORECAST_HORIZON = 1    # 預測未來 1 日收盤價

# -------------------- 訓練/測試切分 --------------------
TRAIN_RATIO = 0.75
STUDY_PERIOD = 100      # 論文 study period = 100 個交易日 (75 train / 25 test)

# -------------------- 回測 --------------------
BACKTEST_START = "2019-01-01"
BACKTEST_END = "2025-06-30"
FAST_MA = 5             # 快線
SLOW_MA = 10            # 慢線
PRED_THRESHOLD = 0.05   # 預測價超出/低於 5 日均線 5% 門檻 (買賣訊號 1)

# -------------------- 情緒分析器 --------------------
# 模式: 'lexicon' (詞典, CPU 快速, 預設) 或 'finbert' (BERT, 較慢)
SENTIMENT_MODE = "lexicon"
SENTIMENT_MODEL_ID = ""  # 若使用 finbert, 填入 HuggingFace model id