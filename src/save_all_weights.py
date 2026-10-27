import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""一次性產生全部個股模型權重 (models/{model}_{code}.pt), 供部署與快速重評估。

- 沿用 run_stock 同 seed/切分 → 權重與既有結果完全一致, 不重寫任何 results 檔
- 已存在的權重自動跳過 → 可中斷續跑
- 之後重評估可用: python3 -m src.run_pipeline lstm_attention 60 --load-weights

用法: python3 -m src.save_all_weights [model] [epochs]
"""
import os
import sys
import time

import config
from src.price_loader import list_all_stock_codes
from src.train import run_stock, model_path


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "lstm_attention"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    codes = list_all_stock_codes()
    todo = [c for c in codes if not os.path.exists(model_path(model, c))]
    print(f"權重產生: {len(codes)} 檔中待訓練 {len(todo)} 檔 (model={model}, epochs={epochs})")
    t0 = time.time()
    n_done, n_skip = 0, len(codes) - len(todo)
    for i, code in enumerate(todo, 1):
        try:
            r = run_stock(code, model_name=model, epochs=epochs, save_model=True)
            n_done += 1 if r else 0
        except Exception as e:  # noqa: BLE001
            print(f"  {code} 失敗: {e}")
        if i % 20 == 0:
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"  {i}/{len(todo)} done ({el:.0f}s), 預估還要 {eta/60:.0f} 分", flush=True)
    print(f"完成: 新存 {n_done}, 原有 {n_skip} -> {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
