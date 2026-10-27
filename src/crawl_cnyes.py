"""鉅亨網 財經新聞爬蟲 (API 版)

API 端點: https://news.cnyes.com/api/v3/news/category/headline
支援參數: page, limit, category, startAt, endAt
輸出: data/processed/cnyes_news.parquet (標題、內容、發布時間、相關股票代碼)
"""
import os
import time
import json
import re
from datetime import datetime, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup

import config


class CnyesCrawler:
    BASE_URL = "https://news.cnyes.com/api/v3/news/category/headline"
    
    # 台股相關分類 ID (從觀察得到)
    TAIWAN_CATEGORIES = {
        827: "台股新聞",
        828: "美股新聞", 
        829: "期貨選擇權",
        830: "基金新聞",
        831: "債券新聞",
        832: "外匯新聞",
        833: "產業新聞",
        834: "法說會",
        835: "大陸新聞",
        836: "國際新聞",
        837: "上櫃新聞",
        838: "興櫃新聞",
    }
    
    def __init__(self, delay=0.5, timeout=15):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://news.cnyes.com/",
        })
    
    def _clean_html(self, html: str) -> str:
        """移除 HTML 標籤，保留純文字"""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        # 清理多餘空白
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _extract_stock_codes(self, item: dict) -> list:
        """從 market、keyword、category 提取相關股票代碼"""
        codes = set()
        
        # 1. 從 market 陣列取得 (格式: {"symbol": "2330.TW", "name": "台積電"})
        for m in item.get("market", []):
            sym = m.get("symbol", "")
            if sym:
                # 移除 .TW 後綴
                code = sym.replace(".TW", "").replace(".TWO", "")
                if code.isdigit() and len(code) == 4:
                    codes.add(code)
        
        # 2. 從 keyword 提取 (如 "2330", "台積電")
        for kw in item.get("keyword", []):
            # 直接是代碼
            if kw.isdigit() and len(kw) == 4:
                codes.add(kw)
            # 公司簡稱對照
            else:
                # 這裡可以加入公司簡稱對照表，先簡單處理
                pass
        
        # 3. 從標題/摘要正則提取 4 位數字 (但排除年份)
        title = item.get("title", "") or ""
        summary = item.get("summary", "") or ""
        text = title + " " + summary
        # 匹配 4 位數字，但排除 19xx, 20xx 年份
        for m in re.finditer(r"(?<!\d)(\d{4})(?!\d)", text):
            code = m.group(1)
            if not (code.startswith("19") or code.startswith("20")):
                # 進一步驗證是否為有效股票代碼 (在我們的 302 清單中)
                codes.add(code)
        
        return list(codes)
    
    def fetch_page(self, page=1, limit=30, category=None, start_at=None, end_at=None) -> dict:
        """抓取單頁 API (API 固定每頁 30 筆)"""
        params = {
            "page": page,
            "limit": limit,
        }
        if category:
            params["category"] = category
        if start_at:
            params["startAt"] = int(start_at)
        if end_at:
            params["end_at"] = int(end_at)
        
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"請求失敗 page={page}: {e}")
            return {}
    
    def fetch_range(self, start_date: str, end_date: str, categories=None, max_pages=1000) -> list:
        """抓取日期範圍內的新聞
        
        Args:
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"
            categories: 要抓取的分類 ID 列表，預設台股新聞(827)
            max_pages: 最大翻頁數
        """
        if categories is None:
            categories = [827]  # 台股新聞
        
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) + 86399  # 當天結束
        
        all_items = []
        
        for cat in categories:
            print(f"抓取分類 {cat} ({self.TAIWAN_CATEGORIES.get(cat, '未知')})...")
            for page in range(1, max_pages + 1):
                data = self.fetch_page(
                    page=page, 
                    limit=100, 
                    category=cat,
                    start_at=start_ts,
                    end_at=end_ts
                )
                
                if not data or "items" not in data:
                    break
                
                items = data["items"].get("data", [])
                if not items:
                    break
                
                # 過濾日期範圍內的
                filtered = []
                for item in items:
                    pub_ts = item.get("publishAt", 0)
                    if start_ts <= pub_ts <= end_ts:
                        filtered.append(item)
                
                all_items.extend(filtered)
                print(f"  第 {page} 頁: {len(filtered)} 篇 (累計 {len(all_items)})")
                
                # 如果 API 回傳的最新文章已早於 start_date，可提早結束
                if items:
                    last_pub = items[-1].get("publishAt", 0)
                    if last_pub < start_ts:
                        break
                
                # API 固定每頁 30 筆，若不足 30 筆表示已到最後一頁
                if len(items) < 30:
                    break
                
                time.sleep(self.delay)
        
        return all_items
    
    def parse_items(self, items: list) -> pd.DataFrame:
        """解析原始項目為 DataFrame"""
        rows = []
        for item in items:
            pub_ts = item.get("publishAt", 0)
            pub_dt = datetime.fromtimestamp(pub_ts)
            
            stock_codes = self._extract_stock_codes(item)
            
            row = {
                "news_id": item.get("newsId"),
                "title": item.get("title", "").strip(),
                "content": self._clean_html(item.get("content", "")),
                "summary": (item.get("summary") or "").strip(),
                "publish_at": pub_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date": pub_dt.strftime("%Y-%m-%d"),
                "category_id": item.get("categoryId"),
                "category_name": self.TAIWAN_CATEGORIES.get(item.get("categoryId", 0), "未知"),
                "keywords": ";".join(item.get("keyword", [])),
                "stock_codes": ",".join(stock_codes),
                "source": "cnyes",
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save(self, df: pd.DataFrame, path=None):
        if path is None:
            path = os.path.join(config.PROCESSED_DIR, "cnyes_news.parquet")
        df.to_parquet(path, index=False)
        print(f"已儲存 {len(df)} 筆新聞到 {path}")


def crawl_cnyes_5years():
    """抓取近 5 年鉅亨網台股新聞"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    
    print(f"抓取範圍: {start_date} ~ {end_date}")
    
    crawler = CnyesCrawler(delay=0.3)
    items = crawler.fetch_range(start_date, end_date, categories=[827], max_pages=500)
    print(f"共抓取 {len(items)} 筆原始新聞")
    
    df = crawler.parse_items(items)
    # 只保留有股票代碼的新聞
    df = df[df["stock_codes"].str.len() > 0].reset_index(drop=True)
    print(f"含股票代碼: {len(df)} 筆")
    
    crawler.save(df)
    return df


if __name__ == "__main__":
    crawl_cnyes_5years()