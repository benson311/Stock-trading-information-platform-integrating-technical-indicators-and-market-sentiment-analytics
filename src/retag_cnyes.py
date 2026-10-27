"""重標鉅亨語料: 用 v2 匹配器 (全名+慣用簡稱+英文) 重跑所有已存文章。

不重爬 — 標題+內文已存在 parquet, 純文字重算 stock_codes。
未來 wave 爬取也會自動套用 (crawl_cnyes_wayback 已改用 news_matcher)。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import time

import pandas as pd

import config
from src.news_matcher import build_news_matcher

RAW_PATH = os.path.join(config.PROCESSED_DIR, "cnyes_news_raw.parquet")


def match_text(text: str, matcher) -> list:
    code_pattern, name_pattern, code_to_name, name_to_code = matcher
    codes = []
    for m in code_pattern.findall(text):
        c = m if isinstance(m, str) else m[0]
        if c not in codes:
            codes.append(c)
    for m in name_pattern.findall(text):
        c = name_to_code.get(m if isinstance(m, str) else m[0])
        if c and c not in codes:
            codes.append(c)
    return codes


def main():
    df = pd.read_parquet(RAW_PATH)
    old_rate = (df["stock_codes"].str.len() > 0).mean()
    print(f"重標前: {len(df):,} 篇, 標記率 {old_rate:.1%}")

    matcher = build_news_matcher()
    t0 = time.time()
    new_tags = []
    for _, r in df.iterrows():
        text = (r["title"] or "") + "\n" + (r["content"] or "")
        new_tags.append(",".join(match_text(text, matcher)))
    df["stock_codes"] = new_tags
    el = time.time() - t0
    print(f"重標完成: {el/60:.1f} 分 ({len(df)/el:.0f} 篇/秒)")

    df.to_parquet(RAW_PATH, index=False)
    new_rate = (df["stock_codes"].str.len() > 0).mean()
    print(f"重標後: 標記率 {old_rate:.1%} -> {new_rate:.1%}")

    # 驗證: 之前漏掉的「台積」文章
    tz = df[df["title"].str.contains("台積法說|台積帶頭|台積秒填息|台積升級", na=False)]
    print(f"\n驗證「台積」文章 ({len(tz)} 篇):")
    for _, r in tz.head(4).iterrows():
        print(f"  {r['title'][:35]} -> {r['stock_codes']}")

    # 新命中分布
    hits = (df[df["stock_codes"].str.len() > 0]
            .stock_codes.str.split(",").explode().value_counts())
    print(f"\n新命中 Top15:")
    print(hits.head(15).to_string())


if __name__ == "__main__":
    main()