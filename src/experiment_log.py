import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""實驗紀錄 (trials.jsonl): 每次訓練/消融/調參/評估追加一行, 只追加不改。

用途: 對抗資料挖掘偏差——「總共試過幾個設定」必須誠實可稽核
     (tw-stock-quant 式做法)。

用法:
  from src.experiment_log import log_trial
  log_trial("train", "run_pipeline:lstm_attention", {"epochs": 60}, {"mse_median": 0.000469})

  python3 -m src.experiment_log          # 檢視最近 20 筆
  python3 -m src.experiment_log 50       # 檢視最近 50 筆
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import config

TRIALS = os.path.join(config.RESULTS_DIR, "trials.jsonl")


def _git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or ""
    except Exception:
        return ""


def log_trial(kind: str, name: str, params: dict = None, metrics: dict = None, notes: str = ""):
    """追加一筆實驗紀錄; 任何失敗都不應中斷主流程 (呼叫端多包 try/except)。"""
    entry = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "name": name,
        "params": params or {},
        "metrics": metrics or {},
        "git_sha": _git_sha(),
        "notes": notes,
    }
    with open(TRIALS, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def show(n: int = 20):
    if not os.path.exists(TRIALS):
        print(f"(尚無紀錄: {TRIALS})")
        return
    lines = open(TRIALS, encoding="utf-8").read().strip().splitlines()
    print(f"共 {len(lines)} 筆實驗紀錄, 最近 {min(n, len(lines))} 筆:")
    for line in lines[-n:]:
        e = json.loads(line)
        m = ", ".join(f"{k}={v}" for k, v in e["metrics"].items()) or "-"
        print(f"  [{e['ts']}] {e['kind']:<12} {e['name']:<28} {m}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    show(n)
