# 資料來源與版權說明（DATA_SOURCES.md）

本 repo **不附帶任何原始語料**（版權與體積考量），僅附模型權重（pooled）與聚合統計結果。
所有原始資料皆可由下方公開管道自行取得後，依 README 的管線重現全部結果。

## 1. 資料清單與取得方式

| 資料 | 涵蓋期間 | 公開取得方式 | 隨 repo 附檔 |
|---|---|---|---|
| 三大法人買賣超（T86） | 2015–2026 | TWSE 官方端點：`https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&response=json`（[公開資訊觀測站](https://mops.twse.com.tw) 同源） | ❌ 不附（管線內建爬蟲 `src/crawl_twse_extra.py`） |
| 融資融券餘額（MI_MARGN） | 2011–2026 | TWSE 官方端點：`https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date=YYYYMMDD&response=json&selectType=STOCK` | ❌ 不附（同上） |
| 個股日線（官方） | 2015–2024 | TWSE [STOCK_DAY](https://www.twse.com.tw/zh/page/trading/exchange/STOCK_DAY.html) | ❌ 不附 |
| 個股日線（還原） | 2011–2026 | Yahoo Finance（`XXXX.TW` / [yfinance](https://pypi.org/project/yfinance/)） | ❌ 不附 |
| PTT Stock 板文章/留言 | 2015–2025 | [PTT web 公開 JSON](https://www.ptt.cc/bbs/Stock/index.html)（版權屬各原作者，僅供學術分析） | ❌ 不附 |
| 鉅亨網新聞 | 2021–2026 | news.cnyes.com（版權屬鉅亨網，**僅供學術，不得公開轉載**） | ❌ 不附（格式見 `README_news.txt`） |
| Yahoo 股市新聞 | 2025–2026 | tw.stock.yahoo.com（版權屬各原始媒體） | ❌ 不附 |

## 2. 重現步驟

```bash
pip install -r requirements.txt
python3 -m src.crawl_twse_extra --report both   # 法人 + 融資融券（可中斷續爬）
#    + 自行取得股價/PTT/新聞語料放入上層對應資料夾（見 config.py RAW_DIR）
python3 -m src.build_features                    # 特徵建置（29 維）
python3 -m src.run_pipeline lstm_attention 60    # 逐股訓練 + 回測
python3 -m src.pooled_model 10 A,B               # 池化訓練（樣本 ×300）
python3 -m src.export_webapp_data                # 匯出 webapp 資料檔
```

## 3. 隨 repo 附帶的產物

- `models/pooled_A.pt` / `models/pooled_B.pt`：池化模型權重（樣本 ×300 之主成果模型）
- `results/*.csv`：**聚合統計**（準確度、回測、預測力檢定、消融、重要度）——不含任何原文內容
- `webapp/mock_data.js`：模型輸出與歷史收盤價（事實數據）

## 4. 授權與引用

- 程式碼：MIT License（見 `LICENSE`）
- 方法參考：Fang, K. & Zhang, Y. (2024). *A BERT-Based Multi-Source Sentiment Analysis and Technical Analysis Model for Stock Price Prediction*. Mathematics, 12(9), 1572.
- 資料版權：TWSE 官方資料依其使用規範引用；新聞與 PTT 內容版權屬原作者/媒體，
  本專案僅做學術分析，不隨 repo 散布原文。
