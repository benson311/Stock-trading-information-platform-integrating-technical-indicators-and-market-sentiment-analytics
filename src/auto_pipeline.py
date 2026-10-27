"""自動編排器: 等 wave 3 完成 (25% 密度) → 自動執行完整新聞版管線。

流程:
  [監測] 每 5 分鐘檢查爬蟲 log; 爬蟲掛掉自動重啟 (最多 3 次);
         超過 14 小時絕對時限則以現有資料觸發。
  [觸發] 凍結語料快照 → 重建情緒指數 → 預測力檢定 → 備份 PTT 舊結果
         → 重建特徵 → 重訓 302 檔 LSTM (約 3h) → 特徵消融+基線比較
         → 自動交易訊號 → 新聞版 vs PTT 版 最終對照報告。

全部輸出記錄在 results/auto_pipeline.log
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import subprocess
import shutil
import time
from datetime import datetime

import pandas as pd

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
CRAWL_LOG = os.path.join(ROOT, "results", "crawl_cnyes_v2.log")
RAW = os.path.join(ROOT, "data", "processed", "cnyes_news_raw.parquet")
FROZEN = os.path.join(ROOT, "data", "processed", "cnyes_news_frozen.parquet")
AUTO_LOG = os.path.join(ROOT, "results", "auto_pipeline.log")

TARGET_WAVE = 2          # 完成此波即觸發 (0-2 波 ≈ 19% 密度; 官網已限速, wave3 需 25h+)
ABS_DEADLINE_H = 13      # 絕對時限 (官網持續限速時的保險)
CHECK_EVERY = 300        # 監測間隔 (秒)


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(AUTO_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def crawl_alive():
    r = subprocess.run(["pgrep", "-f", "src.crawl_cnyes_wayback"],
                       capture_output=True)
    return r.returncode == 0


def restart_crawl():
    subprocess.Popen(
        f"cd {ROOT} && PYTHONUNBUFFERED=1 nohup python3 -m src.crawl_cnyes_wayback run "
        f">> results/crawl_cnyes_v2.log 2>&1 &",
        shell=True)
    log("爬蟲已自動重啟")


def wait_for_target_wave():
    t0 = time.time()
    restarts = 0
    while True:
        try:
            txt = open(CRAWL_LOG, encoding="utf-8").read()
        except FileNotFoundError:
            txt = ""
        done = f"[wave {TARGET_WAVE}] 完成" in txt
        nxt = f"[wave {TARGET_WAVE + 1}/16]" in txt
        if done or nxt:
            log(f"wave {TARGET_WAVE} 已完成, 觸發管線")
            return
        el = (time.time() - t0) / 3600
        if el > ABS_DEADLINE_H:
            log(f"超過 {ABS_DEADLINE_H}h 時限, 以現有資料觸發")
            return
        if not crawl_alive():
            restarts += 1
            if restarts <= 3:
                restart_crawl()
            else:
                log("爬蟲重啟超過 3 次仍失敗, 繼續等待時限...")
        time.sleep(CHECK_EVERY)


def freeze_corpus():
    """複製語料快照 (避開與爬蟲寫入的瞬間碰撞)。"""
    for attempt in range(12):
        try:
            shutil.copy(RAW, FROZEN)
            df = pd.read_parquet(FROZEN, columns=["aid"])
            log(f"凍結語料: {len(df):,} 篇 -> cnyes_news_frozen.parquet")
            return True
        except Exception as e:
            log(f"凍結失敗 (attempt {attempt+1}): {type(e).__name__}, 重試...")
            time.sleep(15)
    return False


def run_stage(name, cmd):
    log(f"=== {name} ===")
    r = subprocess.run(cmd, shell=True, cwd=ROOT,
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    with open(AUTO_LOG, "a", encoding="utf-8") as f:
        f.write(out + "\n")
    tail = [l for l in out.strip().splitlines() if l.strip()][-3:]
    for l in tail:
        log(f"  | {l[:100]}")
    if r.returncode != 0:
        log(f"  !! {name} 失敗 (exit {r.returncode}), 繼續下一階段")
    return r.returncode == 0


def copy_if_exists(src, dst):
    if os.path.exists(os.path.join(ROOT, src)):
        shutil.copy(os.path.join(ROOT, src), os.path.join(ROOT, dst))
        log(f"  複製 {src} -> {dst}")


def main():
    log(f"自動編排器啟動: 等待 wave {TARGET_WAVE} 完成 "
        f"(時限 {ABS_DEADLINE_H}h, 每 {CHECK_EVERY//60} 分鐘檢查)")
    wait_for_target_wave()

    if not freeze_corpus():
        log("!! 凍結失敗, 中止 (資料風險過高)")
        return

    run_stage("重建新聞情緒指數 (凍結語料)",
              "python3 -m src.build_news_sentiment_index data/processed/cnyes_news_frozen.parquet")
    run_stage("情緒預測力檢定 (新聞 vs PTT)",
              "python3 -m src.test_news_predictive_power")
    run_stage("備份 PTT 舊結果", "python3 -m src.backup_ptt_results")
    run_stage("重建特徵 (新聞版)", "python3 -m src.build_features")
    run_stage("重訓 302 檔 LSTM (~3h)",
              "python3 -m src.run_pipeline lstm_attention 60")
    run_stage("特徵消融 + 6 模型基線比較 (~1h)",
              "python3 -m src.compare_models")
    copy_if_exists("results/model_comparison.csv", "results/model_comparison.news.csv")
    copy_if_exists("results/feature_ablation.csv", "results/feature_ablation.news.csv")
    run_stage("自動交易訊號 (新聞版)", "python3 -m src.autotrade_signals")
    copy_if_exists("results/auto_trade_signals.csv", "results/auto_trade_signals.news.csv")
    run_stage("最終對照報告 (新聞 vs PTT)", "python3 -m src.final_report")

    log("全部管線完成! 報告: results/ 與終端輸出")


if __name__ == "__main__":
    main()