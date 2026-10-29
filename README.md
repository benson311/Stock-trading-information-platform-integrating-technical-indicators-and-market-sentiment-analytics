# 台股情緒分析與預測系統 Taiwan Stock Sentiment & Prediction System

> **A Taiwan-stock replication & extension of the BERT-LLA paper — with rigorous, honest ablation studies.**
> 以 LSTM+Attention 預測台股 302 檔個股次日報酬，整合技術指標、新聞/PTT 情緒、三大法人買賣超與融資融券共 29 維特徵；以同 seed A/B 消融、統計檢定與天真基準檢驗，量測「什麼訊號真的有用」。

參考論文：**Fang & Zhang (2024), "A BERT-Based Multi-Source Sentiment Analysis and Technical Analysis Model for Stock Price Prediction" (BERT-LLA), Mathematics 12(9), 1572** —— 將方法完整移植到台灣股市，並納入市場特性（漲跌停、T+2、稅費、產業別）。

## 亮點 Highlights

- **完整可重現管線**：自建 TWSE 官方爬蟲（三大法人 2015–、融資融券 2011–，斷點續爬＋節流退避）→ parquet 快取 → 29 維特徵 → 6 模型比較 → 台股規則回測 → 虛擬交易前端
- **嚴謹實證**：同 seed A/B 消融、逐股配對比較、統計檢定（t-test）、predict-0 天真基準檢驗——含誠實的負結果
- **三種獨立方法交叉驗證**特徵價值：統計檢定 / XGBoost 重要度 / A/B 消融
- **跨股池化訓練**（股票/產業 embedding，樣本 ×300）：MSE 相對逐股再降 13%、90.7% 個股配對勝出
- **零依賴虛擬交易 webapp**：模型導向模擬市場、多來源即時報價自動容錯、台股交易規則完整實作

## 核心成果 Results at a Glance

### 訊號預測力階梯（特徵 vs 次日報酬，n≈295 檔）

| 訊號 | corr | t | 結論 |
|---|---|---|---|
| PTT 散戶情緒 | +0.002 | 0.6 | 虛無——「說的」沒資訊 |
| 新聞情緒（詞典法） | +0.042 | 4.5 | 顯著但弱 |
| **外資買超 60日 z-score** | **+0.043** | **14.9** | **極顯著（82% 個股為正）——法人「做的」有資訊** |
| 融資餘額 z | ≈0 | 0.1 | 不顯著（僅 2011–12 樣本顯著 → 時期依賴） |

### 預測準確度（302 檔，測試集 2023–2026，輸出皆 ±10% 裁切）

| 模型 | MSE 中位 | 贏過「不預測(predict-0)」的個股比例 |
|---|---|---|
| 天真基準（永遠預測 0） | 0.000404 | — |
| 逐股 LSTM+Attention，17 維 | 0.000488 | 4.0% |
| 逐股 LSTM+Attention，29 維 | 0.000469 | 6.6% |
| **Pool B（池化＋股票/產業 embedding）** | **0.000408** | 35.8% |

### 四層誠實結論

1. **訊號層 ✅**：法人實際資金流向是全資料集最強預測訊號（t≈15）；散戶言論無資訊（t=0.6）
2. **相對層 ✅**：同 seed A/B 下，法人特徵使 MSE 相對改善 3.9%、54.3% 個股受益；池化再降 13%、90.7% 個股勝出——特徵與架構與樣本量的排名全部有效
3. **絕對層 ❌**：所有模型皆無法穩定擊敗「不預測」的天真基準——訊號真實存在，但小於模型估計雜訊（與日頻報酬近隨機的市場效率性一致）
4. **交易層 ❌**：方向準確率 ≈50%（池化甚至更低）、DMA+預測回測不穩定勝過單純 DMA——**統計顯著 ≠ 可交易**

> 這個 repo 的主要貢獻：一個可重複的完整框架，以及把「訊號存在」與「模型可用」之間的每一層耗損都量化出來。

## 台股化改動（相對原論文）

| 項目 | 論文（美股） | 本系統（台灣） |
|---|---|---|
| 散戶評論 | Eastmoney 股吧 | PTT Stock 板（25 萬文 / 1,100 萬留言） |
| 漲跌幅 | 無限制 | **±10% 漲跌停**（預測輸出同步裁切） |
| 交割/稅費 | — | **T+2、賣出稅 0.3%、手續費 0.1425%** |
| 預測目標 | 股價絕對值 | **次日報酬率**（±10% 天然有界） |
| 產業 | 新能源車產業鏈 | **TWSE 31 類產業別同業情緒** |
| 新增訊號 | — | **三大法人買賣超、融資融券餘額**（12 維） |
| 語言 | 英文 | 繁體中文（NTUSD＋金融詞典，FinBERT 對照） |

## 系統架構

```
原始資料層            →  處理層                →  特徵層           →  模型層
─────────────────────────────────────────────────────────────────────
TWSE T86 (法人)          parse_ptt / 情緒詞典      TECH 12 技術指標    逐股 LSTM+Attention
TWSE MI_MARGN (融資融券)  news sentiment index    SENT 5 情緒         LSTM/RNN/CNN/RF/SVM
Yahoo/TWSE 股價          institutional.parquet    INST 12 法人/信用    Pool A/B (embedding)
PTT 25萬文 / 新聞 5.6萬篇  margin.parquet          (29 維)            HistGBM 分類器
                                                                  →  回測層 → webapp
```

## 快速開始

```bash
pip install -r requirements.txt      # Python 3.12，CPU 即可（無需 GPU）
python3 -m src.crawl_twse_extra --report both   # 爬法人群/融資融券（斷點續爬）
python3 -m src.build_features                   # 29 維特徵建置
python3 -m src.run_pipeline lstm_attention 60   # 逐股訓練+回測（自動存權重）
python3 -m src.pooled_model 10 A,B               # 池化訓練（主成果模型）
python3 -m src.export_webapp_data               # 匯出 webapp 資料（302 檔）
open webapp/index.html                          # 開啟虛擬交易前端
```

權重快速路徑：`run_pipeline ... --load-weights`（載入已存權重，重評估從 3 小時 → 10 分鐘）；
`pooled_model ... --reuse`；補齊逐股權重：`save_all_weights`。其他模組見 §模組架構。
原始語料不隨 repo 附帶（版權），取得方式見 **[DATA_SOURCES.md](DATA_SOURCES.md)**。

## 虛擬交易前端（webapp/）

單機網頁、**零外部依賴**（離線可開）：`open webapp/index.html`

- **虛擬交易**：自訂初始資金、張數下單、台股規則（手續費 0.1425% 最低 20 元、證交稅 0.3%、±10% 漲跌停擋單）、localStorage 帳戶持久化
- **行情 × 模型預測**：302 檔真實模型輸出（預測次日報酬 / BUY-SELL-HOLD）、走勢圖＋MA＋預測價虛線
- **模擬盤＝模型導向虛擬市場**：個股自昨收出發，報酬 = 市場因子＋產業連動＋個股歷史波動雜訊＋收斂引力，**虛擬收盤即模型預測價**；虛擬日長度、模型導向 0–100%、震盪強度皆可調
- **真實報價**：多來源自動 failover（Yahoo 直連 → TWSE 官方即時 MIS 中繼（單請求 150 檔）→ Yahoo 批次中繼）；AIMD 自適應速率、失敗退避與熔斷冷卻、重點股優先輪動、分頁隱藏暫停
- **投組建議**：等權 / 預測加權 / 產業分散（≤25%）三模式資金分配，一鍵採用建立持倉
- **後端邊界**：接後端時僅需將 `app.js` 頂部 `Api` 物件改為 `fetch()`，前端零改動

## 模組架構

```
src/
  price_loader.py         # 股價合併 (Yahoo+TWSE)
  crawl_twse_extra.py     # TWSE 官方爬蟲 (T86/MI_MARGN, 斷點續爬+退避)
  load_institutional.py   # 跨年度欄位版型解析 → parquet 快取
  parse_ptt.py / map_ptt_to_stocks.py          # PTT 解析與個股標記
  lexicon.py / sentiment.py / sentiment_finbert.py  # 情緒評分 (詞典 vs FinBERT)
  build_sentiment_index.py / build_news_sentiment*.py  # 情緒指數
  technical_indicators.py # 12 技術指標
  build_features.py       # 29 維特徵 (技術+情緒+法人+融資融券)
  dataset.py              # 滑動窗 / 正規化 / 切分
  models.py               # LSTM+Attention / LSTM / RNN / CNN
  train.py / run_pipeline.py / save_all_weights.py    # 訓練與權重管理
  pooled_model.py         # 跨股池化訓練 (+股票/產業 embedding)
  compare_models.py / ablation_institutional.py       # 基線與 A/B 消融
  test_*_predictive_power.py  # 訊號預測力統計檢定
  clip_predictions.py     # ±10% 裁切後處理 (免重訓重算)
  run_xgboost.py          # 橫斷面方向分類 + 特徵重要度
  backtest.py / autotrade*.py # 台股規則回測 / 交易訊號
  export_webapp_data.py   # webapp 資料匯出
```

完整執行順序見 `run.sh`（一鍵、可單步、已完成自動跳過）。

## 結果檔（隨 repo 附帶，皆為聚合統計）

| 檔 | 內容 |
|---|---|
| `results/accuracy_lstm_attention.csv` | 29 維逐股 MSE/RMSE/MAE/方向準確率 |
| `results/accuracy_lstm_attention_17feat_clipped.csv` | 17 維對照組（A/B 另一臂） |
| `results/backtest_lstm_attention*.csv` | DMA+預測 vs DMA 回測 |
| `results/model_comparison.csv` | 六模型比較 |
| `results/pooled_model_results.csv` | 池化 vs 逐股（302 檔配對） |
| `results/inst_margin_predictive_power.csv` | 法人/融資融券訊號預測力檢定 |
| `results/news_vs_ptt_predictive_power.csv` | 新聞 vs PTT 訊號檢定 |
| `results/ablation_institutional.csv` | 法人特徵 A/B 消融（20 檔） |
| `results/xgboost_inst_importance.csv` | 特徵重要度 |
| `results/old_vs_new_29feat_summary.txt` | 17→29 維全量對照摘要 |

## 測試與 CI

- `tests/test_core.py`：以**合成資料**鎖定三個核心函式——滑動窗對位（`make_sequences_for_stock`）、切分協議（`split_train_test`）、台股費用回測（`run_backtest` 的手續費/證交稅/張數取整/T+2 交割時序）——不需真實 `data/` 即可執行
- GitHub Actions（`.github/workflows/ci.yml`）：每次 push/PR 自動跑 `ruff`（僅檢查執行期錯誤）＋ `pytest`
- 本機：`pytest tests -q`

## 因子評估與實驗紀錄

```bash
python3 -m src.factor_eval        # Rank IC + 產業中性 IC + 五分位價差 (29 維全掃)
python3 -m src.experiment_log     # 檢視 trials.jsonl (每次訓練/消融/評估自動記錄)
```

- `factor_eval`：MSE 之外的標準量化評估——**rank IC**（每日橫斷面特徵排序 vs 次日報酬排序）＋
  **五分位價差**（Q5−Q1 等權日報酬差，年化＋t 檢定）＋**產業中性 IC**，
  輸出 `results/factor_eval.csv`。實測亮點：RSI6 呈顯著**負** IC（日頻反轉，t≈−24）；
  `trust_net_vol` 五分位價差年化 +71%（t≈15.6）——法人特徵在標準量化框架下同樣成立
- `trials.jsonl`：每次實驗自動追加一行（時間/參數/指標/git sha），誠實累計「試過幾個設定」

## 引用

- 方法參考：Fang, K. & Zhang, Y. (2024). *A BERT-Based Multi-Source Sentiment Analysis and
  Technical Analysis Model for Stock Price Prediction*. Mathematics, 12(9), 1572.
- 情緒詞典：NTUSD（台大）；個股產業別：TWSE。

## 免責聲明

本專案為**畢業專題之學術研究**，所有預測與訊號基於歷史資料，**不構成任何投資建議**；
回測結果含模型風險與過擬合風險。新聞與 PTT 內容版權屬原作者/媒體，本 repo 不散布任何原文語料
（見 [DATA_SOURCES.md](DATA_SOURCES.md)）。

## License

MIT License（見 [LICENSE](LICENSE)）。若本專案對你的研究有幫助，歡迎引用：
`Laishangqi (2026). Taiwan Stock Sentiment & Prediction System. GitHub repository.`
