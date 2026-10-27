"""新聞個股匹配器 v2 — 更強的中文推斷。

匹配層次:
  1. 公司簡稱 (TWSE 官方 302 檔)
  2. 記者慣用簡稱 (台積、中華電信、日月光、國泰人壽...)
  3. 官方全名 (台灣積體電路製造、聯華電子...) — 自動由 metadata 展開
  4. 英文名/代號 (TSMC, UMC, MediaTek, Foxconn, Largan...)
  5. 數字代碼 (排除 19xx/20xx 年份混淆)

安全設計:
  - 黑名單簡稱 (南港/統一/全國/南亞... 地名與通用詞) 僅靠代碼匹配
  - 長詞優先 (中鋼構 > 中鋼、台積電 > 台積、南亞科 > 南亞)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import re

import config
from src.map_ptt_to_stocks import build_matchers, load_stock_meta

# 高風險簡稱: 地名/通用詞, 不做名稱匹配 (僅保留代碼匹配)
NAME_BLACKLIST = {
    "統一", "南港", "全國", "欣欣", "新興", "冠軍", "泰山", "櫻花", "傳奇",
    "華夏", "農林", "大成", "中台", "中環", "統領", "華城", "東華", "幸福",
    "和大", "大山", "春雨", "南亞",  # 南亞 = 地區名 (South Asia)
}

# 記者慣用簡稱 / 子公司 / 英文名 → 股票代碼
# (代碼不在 302 universe 內者自動略過)
ALIASES = {
    # 半導體
    "2330": ["台積", "台灣積體電路", "TSMC", "TSM"],
    "2303": ["聯華電子", "UMC"],
    "2454": ["聯發科技", "MediaTek"],
    "3034": ["聯詠科技", "Novatek"],
    "3711": ["日月光", "日月光半導體", "日月光投資控股", "ASE"],
    "2408": ["南亞科技", "Nanya Technology"],
    "3037": ["欣興電子"],
    # 電子代工/零組件
    "2317": ["鴻海精密", "Foxconn", "Hon Hai"],
    "4938": ["和碩聯合", "Pegatron"],
    "3231": ["緯創資通", "Wistron"],
    "2382": ["廣達電腦", "Quanta"],
    "2395": ["Advantech"],
    "2301": ["光寶科技", "LiteOn", "Lite-On"],
    "3008": ["大立光電", "Largan"],
    "2308": ["台達電子", "Delta Electronics"],
    # 通信/電信
    "2412": ["中華電信", "Chunghwa Telecom"],
    "3045": ["台灣大哥大", "台哥大", "Taiwan Mobile"],
    # 金融控股 (含子公司 → 母公司)
    "2881": ["富邦金融", "台北富邦", "富邦人壽", "富邦證券", "富邦產險", "Fubon"],
    "2882": ["國泰金融", "國泰人壽", "國泰世華"],
    "2892": ["第一金融", "第一銀行"],
    "2886": ["兆豐金融", "兆豐銀行"],
    "2884": ["玉山金融", "玉山銀行"],
    "5871": ["中租控股", "Chailease"],
    # 航運
    "2603": ["長榮海運", "Evergreen"],
    "2610": ["中華航空", "China Airlines"],
    # 傳產
    "6505": ["台塑石化"],
    "1301": ["台灣塑膠", "Formosa Plastics"],
    "1101": ["台灣水泥", "臺灣水泥"],
    "2002": ["中國鋼鐵"],
    "2207": ["和泰汽車"],
    "1590": ["亞德客"],
}


# 名稱 → 特殊 regex (修正子字串誤配:
#   中電: 「盤中電子/台中電廠/股中電信」誤配 → 環視排除
#   資通: 「資通訊(ICT)/資通安全(網安)」誤配
#   陽明: 「陽明交大/陽明山」誤配)
OVERRIDES = {
    "中電": r"(?<!盤)(?<!台)(?<!股)中電(?!信)",
    "資通": r"資通(?!訊|安)",
    "陽明": r"陽明(?!交大|山)",
}


def _strip_company_suffix(full_name: str) -> str:
    """官方全名去除「股份有限公司」等後綴, 產生可匹配全名。"""
    s = full_name.strip()
    for suf in ("股份有限公司", "控股有限公司", "有限公司", "國際集團"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def build_news_matcher():
    """建置匹配器。回傳 (code_pattern, name_pattern, code_to_name, name_to_code)。

    name_to_code 涵蓋: 官方簡稱(去黑名單) + 慣用簡稱 + 官方全名 + 英文名。
    """
    meta = load_stock_meta()
    code_pattern, _, code_to_name, name_to_code = build_matchers(meta)

    name_to_code = {k: v for k, v in name_to_code.items() if k not in NAME_BLACKLIST}

    # 加入慣用簡稱
    for code, aliases in ALIASES.items():
        if code not in meta:
            continue  # 不在 302 universe
        for a in aliases:
            if a and a not in NAME_BLACKLIST:
                name_to_code[a] = code

    # 加入官方全名 (自動展開)
    for code, m in meta.items():
        full = _strip_company_suffix(m.get("full_name", ""))
        if full and len(full) >= 3 and full not in NAME_BLACKLIST:
            name_to_code.setdefault(full, code)

    # 長詞優先, 避免前綴誤配 (中鋼構>中鋼、台積電>台積)
    names = sorted(name_to_code.keys(), key=len, reverse=True)
    parts = [OVERRIDES.get(n, re.escape(n)) for n in names]
    name_pattern = re.compile("(" + "|".join(parts) + ")")
    return code_pattern, name_pattern, code_to_name, name_to_code


if __name__ == "__main__":
    code_pattern, name_pattern, code_to_name, name_to_code = build_news_matcher()
    print(f"名稱詞典: {len(name_to_code):,} 條 (簡稱+全名+慣用+英文)")

    tests = [
        ("市場靜待台積法說 量縮上漲66點", ["2330"]),
        ("光洋科靶材供應台積升級 3奈米也送認證", ["2330"]),
        ("台積電 ADR 上漲", ["2330"]),
        ("TSMC 法說登場 UMC 跟漲", ["2330", "2303"]),
        ("中華電信 5G 用戶數成長", ["2412"]),
        ("國泰人壽標下台北市精華區土地", ["2882"]),
        ("長榮海運貨櫃三雄齊揚", ["2603"]),
        ("南亞國家經濟成長放緩", []),          # 應不誤配 1303
        ("統一發票 11 12 月開獎", []),          # 應不誤配 1216
        ("南港展覽館舉辦科技展", []),            # 應不誤配 2101
        ("鴻海精密工業 Q2 財報亮眼", ["2317"]),
        ("日月光投控併購矽品", ["3711"]),
        ("中鋼構接單暢旺 鋼價上揚", ["2013"]),        # 2013 在 universe, 且不誤配 2002
        ("南亞科 DDR5 報價上漲", ["2408"]),
    ]
    ok = 0
    for text, expect in tests:
        codes = []
        for m in code_pattern.findall(text):
            c = m if isinstance(m, str) else m[0]
            if c not in codes:
                codes.append(c)
        for m in name_pattern.findall(text):
            c = name_to_code.get(m if isinstance(m, str) else m[0])
            if c and c not in codes:
                codes.append(c)
        passed = set(expect).issubset(set(codes)) and not any(
            c not in expect for c in codes)
        mark = "✓" if passed else "✗"
        print(f"  {mark} {text[:24]:<26} -> {codes}")
        ok += passed
    print(f"\n通過 {ok}/{len(tests)}")