import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""法人買賣超 + 融資融券特徵消融實驗 (20 檔代表股, LSTM+Attention)。

方案 A (with_inst) : 全特徵 = tech12 + sent2 + sector2 + news1 + inst12 (29 維)
方案 B (no_inst)  : 去除法人/融資融券 = 前 17 維 (與新增前完全一致)
兩方案同 seed、同資料切分 → 差異即新特徵的貢獻。

輸出: results/ablation_institutional.csv (逐股兩方案 MSE/RMSE/MAE/方向準確率 + 平均)
"""
import os

import numpy as np
import pandas as pd
import torch

import config
from src.dataset import prepare_stock_data
from src.build_features import INST_COLS
from src.models import build_model
from src.train import train_torch_model, predict_torch_model, evaluate

REPRESENTATIVE = ["2330", "2317", "2454", "2603", "2881", "2002", "1101", "1301",
                  "2412", "2892", "2303", "3034", "3008", "3711", "1590", "4938",
                  "5871", "2498", "6505", "1326"]


def run_scheme(data, cols, epochs=60):
    """cols: 要用的特徵欄位索引 (在已正規化的 X 上切)。data 由 prepare_stock_data 產生。"""
    torch.manual_seed(0)
    np.random.seed(0)
    Xtr, Xte = data["X_train"][:, :, cols], data["X_test"][:, :, cols]
    m = build_model("lstm_attention", len(cols), Xtr.shape[1])
    m = train_torch_model(m, Xtr, data["y_train"], epochs=epochs)
    pred = predict_torch_model(m, Xte)
    y_true = data["y_test"]
    met = evaluate(y_true, pred)
    met["direction_acc"] = float(np.mean((pred > 0) == (y_true > 0)))
    met["n_features"] = len(cols)
    return met


def main(epochs=60):
    rows = []
    for i, code in enumerate(REPRESENTATIVE, 1):
        data = prepare_stock_data(code, target="return")
        if data is None or len(data["X_train"]) < 100:
            print(f"  {code}: 樣本不足, 跳過")
            continue
        n_feat = data["X_train"].shape[2]
        cols_all = list(range(n_feat))
        cols_no_inst = list(range(n_feat - len(INST_COLS)))  # inst 排在最尾
        a = run_scheme(data, cols_all, epochs)
        b = run_scheme(data, cols_no_inst, epochs)
        rows.append({"code": code,
                     "MSE_with_inst": a["MSE"], "MSE_no_inst": b["MSE"],
                     "RMSE_with_inst": a["RMSE"], "RMSE_no_inst": b["RMSE"],
                     "MAE_with_inst": a["MAE"], "MAE_no_inst": b["MAE"],
                     "dir_with_inst": a["direction_acc"], "dir_no_inst": b["direction_acc"],
                     "improve": b["MSE"] - a["MSE"]})
        print(f"  {i}/{len(REPRESENTATIVE)} {code}: MSE {a['MSE']:.6f} vs {b['MSE']:.6f} "
              f"({'改善' if a['MSE'] < b['MSE'] else '變差'} {abs(b['MSE']-a['MSE']):.2e})",
              flush=True)

    df = pd.DataFrame(rows)
    n_better = (df["improve"] > 0).sum()
    summary = pd.DataFrame([{
        "code": f"平均 (n={len(df)}, 法人特徵較佳 {n_better} 檔)",
        "MSE_with_inst": df["MSE_with_inst"].mean(), "MSE_no_inst": df["MSE_no_inst"].mean(),
        "RMSE_with_inst": df["RMSE_with_inst"].mean(), "RMSE_no_inst": df["RMSE_no_inst"].mean(),
        "MAE_with_inst": df["MAE_with_inst"].mean(), "MAE_no_inst": df["MAE_no_inst"].mean(),
        "dir_with_inst": df["dir_with_inst"].mean(), "dir_no_inst": df["dir_no_inst"].mean(),
        "improve": df["improve"].mean(),
    }])
    out_df = pd.concat([df, summary], ignore_index=True)
    out = os.path.join(config.RESULTS_DIR, "ablation_institutional.csv")
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n" + out_df.to_string(index=False))
    print(f"\n已存 -> {out}")
    try:
        from src.experiment_log import log_trial
        log_trial("ablation", "institutional_20stocks",
                  {"epochs": epochs},
                  {"mse_with_inst": round(float(df["MSE_with_inst"].mean()), 6),
                   "mse_no_inst": round(float(df["MSE_no_inst"].mean()), 6),
                   "n_better": int(n_better)})
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    main(ep)
