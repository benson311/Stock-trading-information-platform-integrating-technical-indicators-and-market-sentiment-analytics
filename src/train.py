import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""模型訓練與評估 (論文 Section 3.2)。

支援: lstm_attention / lstm / rnn / cnn (PyTorch) 與 rf / svm (sklearn 基線)。
輸出: 預測結果 (反標準化)、評量 (MSE/RMSE/MAE)。
"""
import os
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import config
from src.dataset import prepare_stock_data, FEATURE_COLS
from src.models import build_model


def train_torch_model(model, X, y, epochs=None, lr=None, batch_size=None, verbose=False):
    epochs = epochs or config.EPOCHS
    lr = lr or config.LEARNING_RATE
    batch_size = batch_size or config.BATCH_SIZE

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep+1}/{epochs} loss={total_loss/n:.6f}")
    return model


def predict_torch_model(model, X):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X).numpy()
    return preds


def evaluate(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}


def model_path(model_name: str, code: str) -> str:
    """個股模型權重路徑: models/{model}_{code}.pt"""
    return os.path.join(config.MODELS_DIR, f"{model_name}_{code}.pt")


def load_stock_model(code: str, model_name: str, n_features: int, window: int):
    """載入已存權重; 不存在回 None。架構由 (n_features, window) 決定, 與訓練時一致。"""
    path = model_path(model_name, code)
    if not os.path.exists(path):
        return None
    m = build_model(model_name, n_features, window)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def run_stock(code, model_name="lstm_attention", epochs=None, verbose=False, target="return",
              save_model=False, load_model=False):
    data = prepare_stock_data(code, target=target)
    if data is None or len(data["X_train"]) < 100:
        return None

    n_features = data["X_train"].shape[2]
    window = data["X_train"].shape[1]

    if model_name in ("rf", "svm"):
        Xtr = data["X_train"].reshape(len(data["X_train"]), -1)
        Xte = data["X_test"].reshape(len(data["X_test"]), -1)
        if model_name == "rf":
            m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0, n_jobs=-1)
        else:
            m = SVR(kernel="rbf", C=10, epsilon=0.01)
        m.fit(Xtr, data["y_train"])
        pred_n = m.predict(Xte)
    else:
        torch.manual_seed(0)
        np.random.seed(0)
        m = build_model(model_name, n_features, window)
        wpath = model_path(model_name, code)
        if load_model and os.path.exists(wpath):
            # 載入既有權重 (快速重評估/回測, 不重訓)
            m.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
            m.eval()
        else:
            m = train_torch_model(m, data["X_train"], data["y_train"], epochs=epochs, verbose=verbose)
            if save_model:
                os.makedirs(config.MODELS_DIR, exist_ok=True)
                torch.save(m.state_dict(), wpath)
        pred_n = predict_torch_model(m, data["X_test"])

    if target == "return":
        # 台股單日 ±10% 漲跌停: 預測報酬裁切至可行域 (模型輸出可能超界, 如 +50%)
        y_pred = np.clip(pred_n, -config.PRICE_LIMIT, config.PRICE_LIMIT)
        y_true = data["y_test"]
        pred_price = data["close_test"] * (1.0 + y_pred)
        true_price = data["close_test"] * (1.0 + y_true)
    else:
        y_pred = pred_n * data["y_range"] + data["y_lo"]
        y_true = data["y_test"] * data["y_range"] + data["y_lo"]
        pred_price = y_pred
        true_price = y_true

    metrics = evaluate(y_true, y_pred)
    # 方向準確率 (報酬率正負)
    acc = np.mean((y_pred > 0) == (y_true > 0)) if target == "return" else None
    return {
        "code": code, "model": model_name, **metrics,
        "direction_acc": acc,
        "y_true": y_true, "y_pred": y_pred,
        "pred_price": pred_price, "true_price": true_price,
        "dates_test": data["dates_test"],
    }


def run_many(codes, model_name="lstm_attention", epochs=None, out_tag="", target="return"):
    results = []
    preds_all = {}
    t0 = time.time()
    for i, code in enumerate(codes):
        r = run_stock(code, model_name=model_name, epochs=epochs, target=target)
        if r:
            results.append(r)
            preds_all[code] = {
                "dates": r["dates_test"], "y_true": r["y_true"],
                "y_pred": r["y_pred"], "pred_price": r["pred_price"],
                "true_price": r["true_price"],
            }
        if (i + 1) % 20 == 0:
            print(f"  [{model_name}] {i+1}/{len(codes)} done ({time.time()-t0:.0f}s)")

    cols = ["code", "model", "MSE", "RMSE", "MAE"]
    if target == "return":
        cols.append("direction_acc")
    summary = pd.DataFrame([{k: r[k] for k in cols} for r in results])
    out = os.path.join(config.RESULTS_DIR, f"results_{model_name}{out_tag}.csv")
    summary.to_csv(out, index=False)
    preds_file = os.path.join(config.RESULTS_DIR, f"predictions_{model_name}{out_tag}.json")
    with open(preds_file, "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                       for kk, vv in v.items()} for k, v in preds_all.items()}, f)
    print(f"\n[{model_name}] 完成 {len(results)} 檔, 平均 MSE={summary['MSE'].mean():.6f} -> {out}")
    return summary


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "2330"
    model = sys.argv[2] if len(sys.argv) > 2 else "lstm_attention"
    r = run_stock(code, model_name=model, epochs=config.EPOCHS, verbose=True)
    if r:
        extra = f" 方向準確率={r['direction_acc']:.3f}" if r.get("direction_acc") is not None else ""
        print(f"\n{code} [{model}]: MSE={r['MSE']:.6f} RMSE={r['RMSE']:.6f} MAE={r['MAE']:.6f}{extra}")