import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""全量訓練執行器: 對所有股票訓練指定模型, 增量寫入結果。

用法: python -m src.run_full [model] [epochs] [limit]
  - model: lstm_attention / lstm / rnn / cnn / rf / svm
  - epoch: 預設 config.EPOCHS
  - limit: 只跑前 N 檔 (除錯用)
"""
import os
import sys
import time

import pandas as pd

import config
from src.price_loader import list_all_stock_codes
from src.train import run_stock


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "lstm_attention"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else config.EPOCHS
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    codes = list_all_stock_codes()
    if limit:
        codes = codes[:limit]

    out_csv = os.path.join(config.RESULTS_DIR, f"results_{model}.csv")
    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv)["code"].astype(str))
    codes = [c for c in codes if c not in done]

    print(f"[{model}] 待訓練 {len(codes)} 檔, epochs={epochs}")

    header_written = os.path.exists(out_csv)
    t0 = time.time()
    for i, code in enumerate(codes):
        try:
            r = run_stock(code, model_name=model, epochs=epochs)
        except Exception as e:
            print(f"  {code} 失敗: {e}")
            continue
        if r is None:
            continue
        row = {k: r[k] for k in ("code", "model", "MSE", "RMSE", "MAE")}
        if "direction_acc" in r and r["direction_acc"] is not None:
            row["direction_acc"] = r["direction_acc"]
        pd.DataFrame([row]).to_csv(out_csv, mode="a",
                                    header=not header_written, index=False)
        header_written = True
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{model}] {i+1}/{len(codes)} done, elapsed {el/60:.1f}m, last {code} MSE={r['MSE']:.6f}")

    print(f"[{model}] 全部完成 -> {out_csv}")


if __name__ == "__main__":
    main()