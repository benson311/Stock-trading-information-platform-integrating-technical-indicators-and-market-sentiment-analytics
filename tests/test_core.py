"""核心邏輯鎖定測試 (合成資料, 不需真實 data/):
滑動窗對位、切分協議、台股費用回測。任何破壞這些行為的改動都應讓 CI 變紅。
"""
import numpy as np
import pandas as pd
import pytest

import config
from src.backtest import compute_metrics, run_backtest
from src.dataset import FEATURE_COLS, make_sequences_for_stock, split_train_test


def make_df(n=60):
    dates = [f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]
    df = pd.DataFrame({c: np.arange(n, dtype=float) * 100.0 + j
                       for j, c in enumerate(FEATURE_COLS)})
    df["close"] = np.arange(1, n + 1, dtype=float)
    df["date"] = dates
    return df


# ---------- 滑動窗 ----------

def test_sliding_window_shape_and_alignment():
    df = make_df(60)
    w, h = 20, 1
    X, y, out_dates, close_t = make_sequences_for_stock(df, window=w, horizon=h, target="return")
    assert X.shape == (60 - w - h + 1, w, len(FEATURE_COLS))
    # X[i, k, j] = 第 (i+k) 列的第 j 個特徵 (特徵值 = 列號×100+欄號)
    assert X[0, 0, 0] == pytest.approx(0.0)
    assert X[0, -1, 0] == pytest.approx((w - 1) * 100.0)
    assert X[7, 3, 0] == pytest.approx((7 + 3) * 100.0)
    assert X[2, 5, 9] == pytest.approx((2 + 5) * 100.0 + 9)
    # y[i] = close[i+w] / close[i+w-1] - 1
    close = df["close"].values
    assert y[0] == pytest.approx(close[w] / close[w - 1] - 1)
    assert y[5] == pytest.approx(close[25] / close[24] - 1)
    assert out_dates[0] == df["date"].iloc[w + h - 1]
    assert close_t[0] == pytest.approx(close[w - 1])


def test_sliding_window_horizon_2():
    df = make_df(60)
    X, y, out_dates, close_t = make_sequences_for_stock(df, window=20, horizon=2, target="return")
    assert X.shape[0] == 60 - 20 - 2 + 1
    close = df["close"].values
    assert y[0] == pytest.approx(close[21] / close[20] - 1)
    assert out_dates[0] == df["date"].iloc[21]


def test_sliding_window_count_matches_pooled_protocol():
    """與 pooled_model 相同的窗口數協議: n_win = n - window (不多不少)。"""
    df = make_df(50)
    X, y, _, _ = make_sequences_for_stock(df, window=20, horizon=1, target="return")
    assert len(X) == len(y) == 50 - 20


# ---------- 切分 ----------

def test_split_train_test_ordered_no_overlap():
    n = 40
    X = np.arange(n).reshape(n, 1, 1).astype(float)
    y = np.arange(n, dtype=float)
    dates = np.array([f"d{i}" for i in range(n)])
    Xtr, ytr, dtr, Xte, yte, dte = split_train_test(X, y, dates)
    ntr = int(n * config.TRAIN_RATIO)
    assert len(Xtr) == ntr and len(Xte) == n - ntr
    assert list(Xtr[:, 0, 0]) == list(range(ntr))              # 順序保留 (不 shuffle)
    assert list(Xte[:, 0, 0]) == list(range(ntr, n))
    assert len(set(dtr) & set(dte)) == 0                        # 無重疊
    assert dtr[-1] < dte[0]


# ---------- 台股費用回測 ----------

def test_backtest_fees_and_tax_roundtrip():
    """平盤價格 + 精準兩個訊號 → 驗證手續費/證交稅/張數取整的現金數學。"""
    n = 30
    close = pd.Series([100.0] * n)
    pr = pd.Series([0.0] * n)
    pr.iloc[9] = 0.06     # Buy1: 預測價 106 > MA5×1.05
    pr.iloc[10] = -0.06   # Sell1: 預測價 94 < MA5×0.95 (此時持股已交割可賣)
    trades, metrics, equity = run_backtest(close, pr, use_prediction=True, initial=1_000_000)
    assert len(trades) == 2
    assert trades[0]["action"] == "BUY" and trades[0]["shares"] == 9000
    assert trades[1]["action"] == "SELL" and trades[1]["shares"] == 9000
    cost = 9000 * 100.0
    expected = 1_000_000 - cost * config.FEE_RATE * 2 - cost * config.TAX_RATE
    assert metrics["final_assets"] == pytest.approx(expected, abs=1e-6)
    assert metrics["total_return"] == pytest.approx(expected / 1_000_000 - 1, abs=1e-9)


def test_backtest_no_signal_stays_flat():
    close = pd.Series([100.0] * 30)
    pr = pd.Series([0.0] * 30)
    trades, metrics, equity = run_backtest(close, pr, use_prediction=True, initial=1_000_000)
    assert trades == []
    assert metrics["total_return"] == pytest.approx(0.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_backtest_same_day_buy_sell_never_both():
    """同日買賣訊號同時出現 → 不成交 (buy != sell 防呆)。"""
    n = 30
    close = pd.Series([100.0] * n)
    pr = pd.Series([0.0] * n)
    pr.iloc[9] = 0.06
    trades, _, _ = run_backtest(close, pr, use_prediction=True, initial=1_000_000)
    assert len(trades) == 1 and trades[0]["action"] == "BUY"


def test_compute_metrics_drawdown():
    eq = np.array([100.0, 110.0, 90.0, 95.0, 100.0])
    m = compute_metrics(eq, 100.0)
    assert m["total_return"] == pytest.approx(0.0)
    assert m["max_drawdown"] == pytest.approx((90 - 110) / 110)
