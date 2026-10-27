import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""將 PTT 文字對應到個股 (股票代碼/公司簡稱標記)。

策略: 對 302 檔股票建立代碼 + 簡稱的 regex, 用 .str 向量化抽取。
留言量達 1100 萬則, 需向量化 + 分塊處理。
"""
import os
import json
import re

import pandas as pd

import config


def load_stock_meta() -> dict:
    with open(os.path.join(config.PROCESSED_DIR, "stock_metadata.json"), encoding="utf-8") as f:
        return json.load(f)


def build_matchers(stock_meta: dict):
    """建立代碼 regex 與名稱 regex, 與 code/name 對照表。

    回傳 (code_pattern, name_pattern, code_to_name, name_to_code)。
    """
    # 排除易與年份混淆的代碼 (19xx/20xx 開頭), 這些以「名稱」匹配為主。
    all_codes = sorted(stock_meta.keys(), key=lambda x: -len(x))
    codes = [c for c in all_codes if not (c.startswith("19") or c.startswith("20"))]
    names = sorted({m["name"] for m in stock_meta.values()}, key=len, reverse=True)

    code_pattern = re.compile(r"(?<!\d)(" + "|".join(codes) + r")(?!\d)")
    name_pattern = re.compile("(" + "|".join(re.escape(n) for n in names) + ")")

    code_to_name = {m["code"]: m["name"] for m in stock_meta.values()}
    name_to_code = {m["name"]: m["code"] for m in stock_meta.values()}
    return code_pattern, name_pattern, code_to_name, name_to_code


def extract_codes(series: pd.Series, code_pattern, name_pattern,
                  code_to_name, name_to_code, chunksize: int = 500_000):
    """回傳每一列文字所提及的股票代碼 (list, 依出現先後, 去重)。"""
    code_pat_str = code_pattern.pattern
    name_pat_str = name_pattern.pattern
    results = []
    for i in range(0, len(series), chunksize):
        chunk = series.iloc[i:i + chunksize].astype(str)
        code_hits = chunk.str.findall(code_pat_str)
        name_hits = chunk.str.findall(name_pat_str)

        for codes, names in zip(code_hits, name_hits):
            seen = []
            for c in codes:
                if c not in seen:
                    seen.append(c)
            for n in names:
                code = name_to_code.get(n)
                if code and code not in seen:
                    seen.append(code)
            results.append(seen)
    return results


def _add_codes(df: pd.DataFrame, text_col: str, code_pattern, name_pattern,
               code_to_name, name_to_code, chunksize: int = 500_000):
    codes = extract_codes(df[text_col], code_pattern, name_pattern,
                          code_to_name, name_to_code, chunksize)
    df["stock_codes"] = codes
    return df


def map_articles_and_pushes():
    stock_meta = load_stock_meta()
    code_pattern, name_pattern, code_to_name, name_to_code = build_matchers(stock_meta)

    # 1) 文章 (title + body 合併提及)
    arts = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "ptt_articles.parquet"))
    arts["full_text"] = arts["title"].fillna("") + "\n" + arts["body"].fillna("")
    print("文章數:", len(arts), "開始標記...")
    arts = _add_codes(arts, "full_text", code_pattern, name_pattern,
                      code_to_name, name_to_code)
    arts = arts.drop(columns=["full_text"])
    arts = arts[arts["stock_codes"].map(len) > 0].reset_index(drop=True)
    arts_path = os.path.join(config.PROCESSED_DIR, "ptt_articles_mapped.parquet")
    arts.to_parquet(arts_path, index=False)
    print(f"提及其一股票的文章: {len(arts)} -> {arts_path}")

    # 2) 留言 (push_text)
    pushes = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "ptt_pushes.parquet"))
    print("留言數:", len(pushes), "開始標記 (分塊)...")
    pushes = _add_codes(pushes, "push_text", code_pattern, name_pattern,
                        code_to_name, name_to_code)
    pushes = pushes[pushes["stock_codes"].map(len) > 0].reset_index(drop=True)
    push_path = os.path.join(config.PROCESSED_DIR, "ptt_pushes_mapped.parquet")
    pushes.to_parquet(push_path, index=False)
    print(f"提及其一股票的留言: {len(pushes)} -> {push_path}")


if __name__ == "__main__":
    map_articles_and_pushes()