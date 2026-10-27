import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""模型比較與特徵消融 (論文 Table 6 / Table 7)。

模型比較: 對子集股票跑 lstm_attention vs lstm/rnn/cnn/rf/svm。
特徵消融: 對一檔股票比較 (1)全特徵 (2)無產業情緒 (3)僅技術指標。
"""
import os
import sys
import json

import numpy as np
import pandas as pd

import config
from src.price_loader import list_all_stock_codes
from src.train import run_stock
from src.dataset import prepare_stock_data, FEATURE_COLS


def compare_models(codes, epochs=60):
    models = ["lstm_attention", "lstm", "rnn", "cnn", "rf", "svm"]
    all_res = []
    for model in models:
        print(f"\n=== {model} ===")
        rows = []
        for code in codes:
            r = run_stock(code, model_name=model, epochs=epochs)
            if r:
                rows.append({k: r[k] for k in ("code", "model", "MSE", "RMSE", "MAE")})
        df = pd.DataFrame(rows)
        all_res.append(df)
        print(df.groupby("model")[["MSE", "RMSE", "MAE"]].mean())
    summary = pd.concat(all_res, ignore_index=True)
    out = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
    summary.to_csv(out, index=False)
    print(f"\n模型比較 -> {out}")


def feature_ablation(code, epochs=60, model_name="lstm_attention"):
    """三種輸入方案比較 (論文 Table 4, 7)。

    欄位順序: 0-11 技術指標, 12-13 自身情緒(sent_pos,sent_neg),
              14-15 產業情緒(sector_pos,sector_neg), 16 新聞(news_net)。
    方案1: 全部 (tech + sent + sector + news)
    方案2: tech + 自身情緒 (去產業情緒 + news)
    方案3: 僅技術指標
    """
    import torch
    from src.models import build_model
    from src.train import train_torch_model, predict_torch_model, evaluate

    data = prepare_stock_data(code, target="return")
    n_feat = data["X_train"].shape[2]

    def fit_mse(cols):
        cols = sorted(set(cols))
        Xtr = data["X_train"][:, :, cols]
        Xte = data["X_test"][:, :, cols]
        torch.manual_seed(0)
        m = build_model(model_name, len(cols), Xtr.shape[1])
        m = train_torch_model(m, Xtr, data["y_train"], epochs=epochs)
        pred = predict_torch_model(m, Xte)
        return evaluate(data["y_test"], pred)["MSE"]

    mse1 = fit_mse(range(n_feat))                 # 全部
    mse2 = fit_mse(range(14))                     # tech + 自身情緒 (0-13)
    mse3 = fit_mse(range(12))                     # 僅 tech (0-11)
    return {
        "code": code,
        "scheme1_full": mse1,
        "scheme2_own_sentiment": mse2,
        "scheme3_tech_only": mse3,
    }


if __name__ == "__main__":
    import pandas as pd
    codes = list_all_stock_codes()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode == "ablation":
        for c in ["2330", "2317", "2002"]:
            print(feature_ablation(c, epochs=60))
    elif mode == "models":
        subset = ["2330", "2317", "2454", "2603", "2881", "2002", "1101", "1301",
                  "2412", "2892", "2303", "3034", "3008", "3711", "1590", "4938",
                  "5871", "2498", "6505", "1326"]
        compare_models([c for c in subset if c in codes], epochs=60)
    else:
        # 預設: 兩者都跑 (基線比較 + 特徵消融), 各存 CSV
        subset = ["2330", "2317", "2454", "2603", "2881", "2002", "1101", "1301",
                  "2412", "2892", "2303", "3034", "3008", "3711", "1590", "4938",
                  "5871", "2498", "6505", "1326"]
        compare_models([c for c in subset if c in codes], epochs=60)
        rows = [feature_ablation(c, epochs=60) for c in ["2330", "2317", "2002"]]
        fab = pd.DataFrame(rows)
        fab.to_csv(os.path.join(config.RESULTS_DIR, "feature_ablation.csv"), index=False)
        print("\n特徵消融 ->", os.path.join(config.RESULTS_DIR, "feature_ablation.csv"))
        print(fab.to_string(index=False))