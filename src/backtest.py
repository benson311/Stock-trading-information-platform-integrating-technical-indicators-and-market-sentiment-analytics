import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""台灣規則回測器 (論文 Section 3.2.4 量化策略 + 台灣市場特性)。

策略:
  - DMA+預測: 快線(5日MA)/慢線(10日MA) + 預測價門檻(±5%)
      Buy  1: 預測價 > 5日MA × 1.05
      Buy  2: 5日MA 上穿 10日MA (黃金交叉)
      Sell 1: 預測價 < 5日MA × 0.95
      Sell 2: 5日MA 下穿 10日MA (死亡交叉)
  - DMA: 僅 Buy 2 / Sell 2 (無預測)

台灣特性:
  - 單日漲跌幅 ±10% (漲跌停)
  - T+2 交割 (買進後第 2 個交易日始可賣出)
  - 證券交易稅 0.3% (賣出), 手續費 0.1425% (買賣)
  - 整股 1000 股/張
"""
import numpy as np
import pandas as pd

import config


def compute_ma(close, window):
    return close.rolling(window).mean()


def run_backtest(close: pd.Series, pred_return: pd.Series = None,
                 use_prediction: bool = True, initial=config.INITIAL_CAPITAL):
    """對單一序列回測, 回傳 (trade_log, metrics)。

    close: 收盤價 (index 為連續交易日 id)
    pred_return: 預測下一日報酬 (與 close 同 index, t 位預測 t->t+1)
    """
    close = close.reset_index(drop=True).astype(float).values
    if pred_return is not None:
        pr = pred_return.reset_index(drop=True).astype(float).values
    else:
        pr = None

    n = len(close)
    ma5 = pd.Series(close).rolling(config.FAST_MA).mean()
    ma10 = pd.Series(close).rolling(config.SLOW_MA).mean()

    cash = initial
    shares = 0                      # 可賣出股數
    pending = []                    # 待交割 (天數, 股數)
    trades = []
    equity = []

    for t in range(n - 1):
        # 處理交割 (T+2): 前天買入的今日可賣
        pending = [(d - 1, s) for d, s in pending]
        for d, s in pending:
            if d <= 0:
                shares += s
        pending = [(d, s) for d, s in pending if d > 0]

        valid = not (np.isnan(ma5[t]) or np.isnan(ma10[t]))
        prev_valid = not (np.isnan(ma5[t - 1]) or np.isnan(ma10[t - 1])) if t > 0 else False

        buy = False
        sell = False
        reason = ""

        if valid:
            if use_prediction and pr is not None and not np.isnan(pr[t]):
                pred_price = close[t] * (1.0 + pr[t])
                if pred_price > ma5[t] * (1 + config.PRED_THRESHOLD):
                    buy = True
                    reason = "Buy1(預測價>MA5x1.05)"
                elif pred_price < ma5[t] * (1 - config.PRED_THRESHOLD):
                    sell = True
                    reason = "Sell1(預測價<MA5x0.95)"

            if prev_valid:
                if ma5[t] > ma10[t] and ma5[t - 1] <= ma10[t - 1]:
                    buy = True
                    reason += ("" if not reason else "+") + "Buy2(黃金交叉)"
                elif ma5[t] < ma10[t] and ma5[t - 1] >= ma10[t - 1]:
                    sell = True
                    reason += ("" if not reason else "+") + "Sell2(死亡交叉)"

        # 次日成交價 (用次日收盤近似)
        exec_price = close[t + 1]

        if buy and buy != sell:
            # 全倉買進 (簡化: 投入全部現金, 張數向下取整)
            lots = int(cash * (1 - config.FEE_RATE) / (exec_price * config.ROUND_LOT))
            if lots > 0:
                cost = lots * config.ROUND_LOT * exec_price
                fee = cost * config.FEE_RATE
                cash -= (cost + fee)
                pending.append((config.SETTLEMENT_DAYS - 1, lots * config.ROUND_LOT))
                trades.append({"t": t + 1, "action": "BUY", "price": exec_price,
                               "shares": lots * config.ROUND_LOT, "reason": reason})
        elif sell and buy != sell:
            if shares > 0:
                lots = shares // config.ROUND_LOT
                if lots > 0:
                    n_sh = lots * config.ROUND_LOT
                    proceeds = n_sh * exec_price
                    fee = proceeds * config.FEE_RATE
                    tax = proceeds * config.TAX_RATE
                    cash += (proceeds - fee - tax)
                    shares -= n_sh
                    trades.append({"t": t + 1, "action": "SELL", "price": exec_price,
                                   "shares": n_sh, "reason": reason})

        equity.append(cash + shares * close[t + 1] +
                      sum(s * close[t + 1] for d, s in pending))

    final_equity = cash + shares * close[-1] + sum(s * close[-1] for d, s in pending)
    equity.append(final_equity)
    equity = np.array(equity)

    metrics = compute_metrics(equity, initial)
    return trades, metrics, equity


def compute_metrics(equity, initial):
    ret = equity / initial - 1
    total_return = ret[-1]
    # 日報酬
    daily = np.diff(equity) / equity[:-1]
    sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252) if len(daily) else 0.0
    # 最大回撤
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    return {
        "total_return": total_return,
        "final_assets": float(equity[-1]),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def backtest_stock_from_predictions(code, preds: dict, use_prediction=True):
    """由預測結果 (dates,y_pred) 配上實際收盤, 執行回測。"""
    from src.price_loader import load_best_price
    df = load_best_price(code)
    if df.empty:
        return None
    dates_test = preds["dates"]
    y_pred = np.array(preds["y_pred"])
    close_df = df.set_index("date")["close"]

    # 回測區間 = 測試集最後一個已知日之後
    pred_map = {d: r for d, r in zip(dates_test, y_pred)}
    # 取測試集期間的收盤
    sub = close_df.reindex(sorted(dates_test)).dropna()
    pr = pd.Series([pred_map.get(d, np.nan) for d in sub.index], index=sub.index)
    trades, metrics, equity = run_backtest(sub, pr, use_prediction=use_prediction)
    return trades, metrics, equity


if __name__ == "__main__":
    # 簡易自我測試
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500)))
    pr = pd.Series(rng.normal(0.0005, 0.01, 500))
    trades, m, eq = run_backtest(close, pr, use_prediction=True)
    print(m)
    trades_dma, md, _ = run_backtest(close, pr, use_prediction=False)
    print("DMA:", md)