import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Pool 302 檔共同訓練 (跨股池化) — 解決單股樣本不足 (每股 ~2,700 → 聯合 ~80 萬樣本)。

設定:
  A (plain): 29 維特徵直接池化, 共用一組權重
  B (emb)  : A + 股票 embedding(16) + 產業 embedding(8), 逐時間步拼接
             (讓模型學習個股/產業固定效果, 同時共享跨股 pattern)

與逐股訓練完全可比:
  - 每股樣本切分 (前75% train / 後25% test) 與 dataset.py 相同
  - min-max 正規化「每股」僅以該股 train 統計 fit (與逐股流程一致)
  - 預測輸出 ±10% 裁切 (config.PRICE_LIMIT)

輸出: results/pooled_model_results.csv (逐股 MSE/方向準確率: 逐股基準 vs A vs B)
用法: python3 -m src.pooled_model [epochs] [configs]   例: 10 A,B
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

import config
from src.dataset import FEATURE_COLS
from src.models import LSTMAttention


class PooledLSTM(LSTMAttention):
    """LSTM+Attention + (可選) 股票/產業 embedding。"""

    def __init__(self, n_features, n_stocks, n_inds, use_emb=True,
                 emb_stock=16, emb_ind=8):
        self.use_emb = use_emb
        dim_in = n_features + (emb_stock + emb_ind if use_emb else 0)
        super().__init__(dim_in)
        if use_emb:
            self.emb_s = nn.Embedding(n_stocks, emb_stock)
            self.emb_i = nn.Embedding(n_inds, emb_ind)

    def forward(self, x, s_idx=None, i_idx=None):
        if self.use_emb:
            B, T, _ = x.shape
            es = self.emb_s(s_idx).unsqueeze(1).expand(B, T, -1)
            ei = self.emb_i(i_idx).unsqueeze(1).expand(B, T, -1)
            x = torch.cat([x, es, ei], dim=2)
        return super().forward(x)


def build_pooled_data(window=config.WINDOW_SIZE, ratio=config.TRAIN_RATIO,
                      min_windows=120, verbose=True):
    """回傳 dict: X_train, X_test, y_train, y_test (float32),
    stock_train/test, ind_train/test (int64), codes (依 stock idx 排列), test_dates per stock。"""
    df = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "features.parquet"))
    df["date"] = df["date"].astype(str)
    meta = json.load(open(os.path.join(config.PROCESSED_DIR, "stock_metadata.json"),
                          encoding="utf-8"))

    codes = sorted(df["code"].unique())
    code_idx = {c: i for i, c in enumerate(codes)}
    ind_names = sorted({meta.get(c, {}).get("industry", "00") for c in codes})
    ind_idx = {k: i for i, k in enumerate(ind_names)}

    Xtr, Xte, ytr, yte = [], [], [], []
    str_, itr_, ste_, ite_ = [], [], [], []
    test_dates_by_code = {}
    skipped = []

    for code in codes:
        g = df[df["code"] == code].sort_values("date")
        n = len(g)
        n_win = n - window
        if n_win < min_windows:
            skipped.append(code)
            continue
        feats = g[FEATURE_COLS].astype(np.float32).values          # (n, F)
        close = g["close"].astype(float).values
        dates = g["date"].values

        # 滑動窗 (view 產生 n-window+1 個窗, 最後一窗無對應目標 → 截去)
        X = sliding_window_view(feats, window_shape=window, axis=0)  # (n_win+1, F, w)
        X = np.ascontiguousarray(X.transpose(0, 2, 1))[:n_win]       # (n_win, w, F)
        y = close[window:] / close[window - 1:n - 1] - 1.0           # 次日報酬
        assert len(X) == len(y), f"{code}: X {len(X)} != y {len(y)}"

        n_train = int(n_win * ratio)
        # 每股 min-max 僅 fit train (與逐股流程一致)
        lo = X[:n_train].min(axis=(0, 1), keepdims=True)
        hi = X[:n_train].max(axis=(0, 1), keepdims=True)
        rng = np.where(hi - lo == 0, 1.0, hi - lo)
        Xn = ((X - lo) / rng).astype(np.float32)

        # 資料清潔: 捨棄「不可能目標」的訓練窗 (上市/上櫃 ±10% 上限)。
        # 源頭: (1) Yahoo還原價↔TWSE官方價縫接偽跳空 (2015-01-05 等)
        #       (2) 少數壞列 (如 1752 close 4215→42)
        #       (3) 興櫃股真實無漲跌幅限制 (>10%) — 一併捨棄以保訓練穩定。
        # 測試集不動, 維持與逐股基準完全可比。
        keep = np.abs(y[:n_train]) <= 0.11
        si, ii = code_idx[code], ind_idx[meta.get(code, {}).get("industry", "00")]
        Xtr.append(Xn[:n_train][keep]); Xte.append(Xn[n_train:])
        ytr.append(y[:n_train].astype(np.float32)[keep])
        yte.append(y[n_train:].astype(np.float32))
        str_.append(np.full(int(keep.sum()), si)); ste_.append(np.full(n_win - n_train, si))
        itr_.append(np.full(int(keep.sum()), ii)); ite_.append(np.full(n_win - n_train, ii))
        test_dates_by_code[code] = dates[window + n_train:]

    out = {
        "X_train": np.concatenate(Xtr), "X_test": np.concatenate(Xte),
        "y_train": np.concatenate(ytr), "y_test": np.concatenate(yte),
        "stock_train": np.concatenate(str_), "stock_test": np.concatenate(ste_),
        "ind_train": np.concatenate(itr_), "ind_test": np.concatenate(ite_),
        "codes": [c for c in codes if c not in skipped],
        "test_slice": {c: (s, s + len(d)) for c, d, s in
                       zip(sorted(test_dates_by_code), test_dates_by_code.values(),
                           np.cumsum([0] + [len(test_dates_by_code[c])
                                            for c in sorted(test_dates_by_code)]))},
        "skipped": skipped,
    }
    if verbose:
        print(f"池化資料: {len(out['codes'])} 檔 (skip {len(skipped)}), "
              f"train {out['X_train'].shape}, test {out['X_test'].shape}")
    return out


def train_pooled(model, data, epochs, batch=256, lr=config.LEARNING_RATE, use_emb=True, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X, y = data["X_train"], data["y_train"]
    s, i = data["stock_train"], data["ind_train"]
    n = len(X)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        tot, nb = 0.0, 0
        for a in range(0, n, batch):
            idx = perm[a:a + batch]
            xb = torch.from_numpy(X[idx])
            yb = torch.from_numpy(y[idx])
            opt.zero_grad()
            pred = model(xb, torch.from_numpy(s[idx]), torch.from_numpy(i[idx])) if use_emb \
                else model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        print(f"[{tag}] epoch {ep}/{epochs}: train MSE {tot/nb:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def predict_pooled(model, data, batch=1024, use_emb=True):
    model.eval()
    preds = []
    X, s, i = data["X_test"], data["stock_test"], data["ind_test"]
    for a in range(0, len(X), batch):
        xb = torch.from_numpy(X[a:a + batch])
        out = model(xb, torch.from_numpy(s[a:a + batch]),
                    torch.from_numpy(i[a:a + batch])) if use_emb else model(xb)
        preds.append(out.numpy())
    model.train()
    return np.clip(np.concatenate(preds), -config.PRICE_LIMIT, config.PRICE_LIMIT)


def per_stock_metrics(data, preds, col_prefix):
    y = data["y_test"]
    rows = []
    for code, (s0, s1) in data["test_slice"].items():
        yt, yp = y[s0:s1], preds[s0:s1]
        if len(yt) == 0:
            continue
        mse = float(np.mean((yt - yp) ** 2))
        dacc = float(np.mean((yp > 0) == (yt > 0)))
        rows.append({"code": code, f"{col_prefix}_MSE": mse, f"{col_prefix}_dir": dacc})
    return pd.DataFrame(rows)


def pooled_path(cfg: str) -> str:
    return os.path.join(config.MODELS_DIR, f"pooled_{cfg}.pt")


def load_pooled_model(cfg: str, n_feat: int, n_stocks: int, n_inds: int):
    """載入已存的池化模型權重; 不存在回 None。cfg: 'A' (無 emb) 或 'B' (含 emb)。"""
    path = pooled_path(cfg)
    if not os.path.exists(path):
        return None
    m = PooledLSTM(n_feat, n_stocks, n_inds, use_emb=(cfg == "B"))
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    epochs = int(pos[0]) if pos else 10
    configs = pos[1].split(",") if len(pos) > 1 else ["A", "B"]
    data = build_pooled_data()
    n_feat = data["X_train"].shape[2]
    n_stocks, n_inds = len(data["codes"]), int(data["ind_train"].max()) + 1

    res = per_stock_baseline()
    for cfg in configs:
        use_emb = cfg == "B"
        model = load_pooled_model(cfg, n_feat, n_stocks, n_inds) if "--reuse" in flags else None
        if model is not None:
            print(f"[Pool-{cfg}] 載入既有權重 {pooled_path(cfg)}")
        else:
            torch.manual_seed(0), np.random.seed(0)
            model = PooledLSTM(n_feat, n_stocks, n_inds, use_emb=use_emb)
            train_pooled(model, data, epochs, use_emb=use_emb, tag=f"Pool-{cfg}")
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), pooled_path(cfg))
            print(f"[Pool-{cfg}] 權重已存 -> {pooled_path(cfg)}")
        pred = predict_pooled(model, data, use_emb=use_emb)
        m = per_stock_metrics(data, pred, f"pool_{cfg}")
        res = res.merge(m, on="code", how="outer")

    cols_mse = [c for c in res.columns if c.endswith("_MSE")]
    print("\n" + "=" * 70)
    print(f"Pool 共訓 vs 逐股基準 (測試集 2023–2026, {epochs} epochs)")
    print("=" * 70)
    for c in cols_mse:
        d = res[c].dropna()
        print(f"  {c:<14}: MSE 中位 {d.median():.6f}, RMSE 均 {np.sqrt((d**2)).mean():.5f}, "
              f"方向準確率均 {res[c.replace('_MSE','_dir')].dropna().mean():.4f} (n={len(d)})")
    for c in cols_mse[1:]:
        base = res[cols_mse[0]]
        m = (res[c] < base).mean()
        print(f"  {c} 優於逐股基準比例: {m:.1%}")
    out = os.path.join(config.RESULTS_DIR, "pooled_model_results.csv")
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已存 -> {out}")


def per_stock_baseline():
    """逐股訓練基準 (±10% 裁切後之 accuracy CSV)。"""
    acc = pd.read_csv(os.path.join(config.RESULTS_DIR, "accuracy_lstm_attention.csv"),
                      dtype={"code": str})
    return acc[["code", "MSE", "direction_acc"]].rename(
        columns={"MSE": "perstock_MSE", "direction_acc": "perstock_dir"})


if __name__ == "__main__":
    main()
