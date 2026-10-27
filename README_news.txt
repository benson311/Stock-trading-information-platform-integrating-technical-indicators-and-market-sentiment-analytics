台股新聞語料快照 (畢業專案: 台灣股市情緒分析與預測)
匯出日期: 2026-09-24 13:29

=== 一、內容概述 ===
本資料集包含兩大台股新聞來源, 皆已附「發布日期」與「個股代碼標記」,
可直接用於情緒分析、事件研究、股市預測等學術研究。

1. cnyes_news_snapshot_20260924.csv — 鉅亨網 (news.cnyes.com) 新聞
   - 43,707 篇
   - 日期範圍: 2021-03-07 ~ 2026-09-21
   - 覆蓋股票: 299 檔

2. yahoo_news_dated.csv — Yahoo 奇摩股市 (tw.stock.yahoo.com) 新聞
   - 11,981 篇
   - 日期範圍: 2025-09-03 ~ 2026-09-03
   - 覆蓋股票: 267 檔

=== 二、欄位定義 ===

[cnyes_news_snapshot_20260924.csv]
  aid          鉅亨網文章 ID (流水號, 可組出原文網址:
               https://news.cnyes.com/news/id/{aid})
  date         發布日期 (YYYY-MM-DD, 已轉台灣時區)
  date_approx  日期是否為近似值 (True=取自網頁存檔時間, 僅極少數)
  title        標題
  content      內文 (JSON-LD articleBody, 已去除網頁雜訊, 部分快訊類較短)
  stock_codes  文中提及的台股代碼 (逗號分隔; 空=未提及具體個股)
  source       來源標記 (cnyes_wayback / cnyes_live)

[yahoo_news_dated.csv]
  code   股票代碼 (該新聞所屬個股頁面)
  date   發布日期 (取自文章頁 JSON-LD datePublished)
  title  標題
  url    原文網址

=== 三、使用方式 (Python) ===
  import pandas as pd
  df = pd.read_csv("cnyes_news_snapshot_20260924.csv")
  # 依個股過濾
  tsmc = df[df["stock_codes"].str.contains("2330", na=False)]

=== 四、資料來源與授權聲明 ===
- 鉅亨網新聞內容版權屬 cnyes.com 所有
- Yahoo 新聞內容版權屬各原始媒體所有
- 本資料集僅供「學術研究/課堂專題」使用, 請勿公開轉載或商業利用
- 引用建議格式:
  資料來源: 鉅亨網 (https://news.cnyes.com) 與 Yahoo奇摩股市
  (https://tw.stock.yahoo.com), 經網頁爬取整理, 2026。

=== 五、備註 ===
- 鉅亨網語料仍在背景擴充中 (波次式爬取), 此為當日快照;
  最終完整版預計涵蓋 2021-2026 約 50 萬篇。
- 文章依「鉅亨文章 ID」去重; 個股標記採「代碼+公司簡稱」雙重比對,
  並已排除易誤判之地名/通用詞 (如 南港區/統一發票)。
