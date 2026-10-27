"""快照匯出: 把已爬新聞語料匯出成 CSV + 說明書 + zip, 方便分享。

輸出 (專案根目錄):
  cnyes_news_snapshot_{date}.csv   — 鉅亨網文章 (Excel 可開, UTF-8 BOM)
  yahoo_news_dated.csv            — Yahoo 新聞 (含時間戳)
  資料說明.txt                      — 欄位定義/來源/引用方式
  news_snapshot_{date}.zip        — 上述檔案打包
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import zipfile
from datetime import datetime

import pandas as pd

import config

EXPORT_COLS_CNYES = ["aid", "date", "date_approx", "title", "content",
                     "stock_codes", "source"]
EXPORT_COLS_YAHOO = ["code", "date", "title", "url"]


def main():
    today = datetime.now().strftime("%Y%m%d")
    root = config.BASE_DIR

    cnyes_path = os.path.join(config.PROCESSED_DIR, "cnyes_news_raw.parquet")
    yahoo_path = os.path.join(config.PROCESSED_DIR, "yahoo_news_dated.parquet")

    # --- 1) 鉅亨 CSV ---
    cnyes = pd.read_parquet(cnyes_path)
    cnyes_csv = os.path.join(root, f"cnyes_news_snapshot_{today}.csv")
    cnyes[EXPORT_COLS_CNYES].to_csv(cnyes_csv, index=False, encoding="utf-8-sig")

    # --- 2) Yahoo CSV ---
    yh = pd.read_parquet(yahoo_path)
    yahoo_csv = os.path.join(root, "yahoo_news_dated.csv")
    yh[EXPORT_COLS_YAHOO].to_csv(yahoo_csv, index=False, encoding="utf-8-sig")

    # --- 3) 說明書 ---
    readme = f"""台股新聞語料快照 (畢業專案: 台灣股市情緒分析與預測)
匯出日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}

=== 一、內容概述 ===
本資料集包含兩大台股新聞來源, 皆已附「發布日期」與「個股代碼標記」,
可直接用於情緒分析、事件研究、股市預測等學術研究。

1. cnyes_news_snapshot_{today}.csv — 鉅亨網 (news.cnyes.com) 新聞
   - {len(cnyes):,} 篇
   - 日期範圍: {cnyes['date'].min()} ~ {cnyes['date'].max()}
   - 覆蓋股票: {cnyes['stock_codes'].str.split(',').explode().dropna().nunique()} 檔

2. yahoo_news_dated.csv — Yahoo 奇摩股市 (tw.stock.yahoo.com) 新聞
   - {len(yh):,} 篇
   - 日期範圍: {yh['date'].dropna().min()} ~ {yh['date'].dropna().max()}
   - 覆蓋股票: {yh['code'].nunique()} 檔

=== 二、欄位定義 ===

[cnyes_news_snapshot_{today}.csv]
  aid          鉅亨網文章 ID (流水號, 可組出原文網址:
               https://news.cnyes.com/news/id/{{aid}})
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
  df = pd.read_csv("cnyes_news_snapshot_{today}.csv")
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
"""
    readme_path = os.path.join(root, "README_news.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    # --- 4) 打包 ---
    zip_path = os.path.join(root, f"news_snapshot_{today}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(cnyes_csv, arcname=os.path.basename(cnyes_csv))
        z.write(yahoo_csv, arcname=os.path.basename(yahoo_csv))
        z.write(readme_path, arcname=os.path.basename(readme_path))

    # 檔案大小
    sz_csv = os.path.getsize(cnyes_csv) / 1e6
    sz_zip = os.path.getsize(zip_path) / 1e6
    print(f"匯出完成:")
    print(f"  {cnyes_csv} ({sz_csv:.1f} MB)")
    print(f"  {yahoo_csv} ({os.path.getsize(yahoo_csv)/1e6:.1f} MB)")
    print(f"  {readme_path}")
    print(f"  {zip_path} ({sz_zip:.1f} MB) <- 分享用這個")
    print(f"\n內容: 鉅亨 {len(cnyes):,} 篇 + Yahoo {len(yh):,} 篇")


if __name__ == "__main__":
    main()