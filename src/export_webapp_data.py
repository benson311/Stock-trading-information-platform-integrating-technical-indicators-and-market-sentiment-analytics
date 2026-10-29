import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""匯出虛擬交易前端 (webapp/) 所需資料 -> webapp/mock_data.js

來源皆為本專案真實輸出:
  - features.parquet            最新收盤 + 近 120 交易日走勢 (29 維特徵模型實際吃的資料)
  - auto_trade_signals.csv      LSTM+Attention 預測次日報酬 + BUY/SELL/HOLD 訊號 (29 維版)
  - twse/t86 最新一檔           證券名稱
  - stock_metadata.json         產業別

輸出: webapp/mock_data.js (window.MOCK_DATA = {...})
之後接後端時, 以相同 JSON 結構提供 API 即可, 前端無需修改。
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import config

OUT = os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "webapp", "mock_data.js")
HISTORY_DAYS = 120
MAX_HISTORY_STEPS = 120


def load_names():
    """跨 t86 檔案合併 code -> 股名 (新檔優先; 單日檔可能不完整, 如 2026 年僅 7 檔)。"""
    files = sorted(glob.glob(os.path.join(config.RAW_T86_DIR, "t86_*.json")))
    names, date = {}, ""
    for f in reversed(files):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("stat") == "OK" and d.get("data"):
            fields = d["fields"]
            ic = next(i for i, x in enumerate(fields) if "證券代號" in x)
            iN = next(i for i, x in enumerate(fields) if "證券名稱" in x)
            for r in d["data"]:
                names.setdefault(str(r[ic]).strip(), str(r[iN]).strip())
            if not date:
                date = os.path.basename(f)[4:12]
            if len(names) >= 500:  # 已含完整上市檔數, 停止回溯
                break
    return names, date


def main():
    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = 全部
    feats = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "features.parquet"))
    feats["date"] = feats["date"].astype(str)
    sig = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_trade_signals.csv"),
                      dtype={"code": str})
    meta = json.load(open(os.path.join(config.PROCESSED_DIR, "stock_metadata.json"),
                          encoding="utf-8"))
    names, t86_date = load_names()

    # 全部個股依預測報酬排序 (可選前 N 檔)
    sig = sig.sort_values("pred_return", ascending=False)
    if n_limit > 0:
        keep = set(sig.head(n_limit)["code"]) | {"2330"}
        sig = sig[sig["code"].isin(keep)]

    stocks = []
    for _, row in sig.iterrows():
        code = row["code"]
        g = feats[feats["code"] == code].sort_values("date")
        if len(g) < 60:
            continue
        g = g.tail(HISTORY_DAYS + 1)
        history = [[d, round(float(c), 2)] for d, c in zip(g["date"], g["close"])]
        last_close = history[-1][1]  # 走勢最後一日收盤 (訊號檔無 last_close 欄)

        # K 線 (近 60 日 OHLC, 供形態比對與蠟燭圖)
        ohlc = []
        for _, r2 in g.tail(60).iterrows():
            if pd.notna(r2["open"]) and pd.notna(r2["high"]) and pd.notna(r2["low"]):
                ohlc.append([r2["date"], round(float(r2["open"]), 2), round(float(r2["high"]), 2),
                             round(float(r2["low"]), 2), round(float(r2["close"]), 2)])

        # 個股新聞情緒 (近 60 日有新聞之日: [date, 淨情緒]) + 20 日均值
        news = []
        for _, r2 in g.tail(60).iterrows():
            net = float(r2["sent_pos"]) - float(r2["sent_neg"])
            if abs(net) > 1e-9:
                news.append([r2["date"], round(net, 3)])
        news_avg20 = round(float(g["sent_pos"].tail(20).mean() - g["sent_neg"].tail(20).mean()), 4)

        stocks.append({
            "code": code,
            "name": (meta.get(code, {}).get("name") or names.get(code) or code),
            "industry": meta.get(code, {}).get("industry", ""),
            "signal": row["signal"],
            "reason": row.get("reason", ""),
            "predReturn": round(float(row["pred_return"]), 6),
            "predPrice": round(float(row["pred_price"]), 2),
            "lastClose": last_close,
            "ohlc": ohlc,
            "news": news,
            "newsAvg20": news_avg20,
            "ma5": round(float(row["ma5"]), 2),
            "ma10": round(float(row["ma10"]), 2),
            "history": history,
        })

    stocks.sort(key=lambda s: -s["predReturn"])
    data = {
        "generatedAt": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "priceAsOf": stocks[0]["history"][-1][0] if stocks else "",
        "t86NameDate": t86_date,
        "modelInfo": {
            "name": "LSTM+Attention (29 維: 技術 12 + 情緒 5 + 法人/融資融券 12)",
            "signalRule": "預測價 > MA5×1.05 → BUY; < MA5×0.95 → SELL; 否則 HOLD",
            "note": "預測次日報酬已 ±10% 裁切 (台股漲跌停)",
        },
        "fees": {"commission": 0.001425, "minFee": 20, "tax": 0.003},
        "stocks": stocks,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("// 自動產生: python3 -m src.export_webapp_data\n")
        f.write("// 資料為本專案真實模型輸出; 接後端時以相同結構提供 API 即可\n")
        f.write("window.MOCK_DATA = ")
        json.dump(data, f, ensure_ascii=False)
        f.write(";\n")
    print(f"匯出 {len(stocks)} 檔 -> {OUT}")
    print(f"價格截至 {data['priceAsOf']}, 預測報酬最大 {data['stocks'][0]['predReturn']:.4f}")


if __name__ == "__main__":
    main()
