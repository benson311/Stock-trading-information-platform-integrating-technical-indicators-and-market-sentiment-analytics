#!/bin/bash
# 等待 TWSE 爬蟲結束後, 自動執行:
#   1. 重建法人/融資融券快取 + features.parquet (29 維)
#   2. 新特徵預測力檢定
#   3. 20 檔代表股 A/B 消融
#   4. 備份舊 17 維結果 → 302 檔全面重訓 + 回測 (LSTM+Attention, 60 epochs)
#   5. 新舊對照摘要
# 用法: nohup bash auto_chain_after_crawl.sh <crawler_pid> &
PID=$1
LOG="results/auto_chain.log"
echo "等待爬蟲 PID=$PID 結束..." >> $LOG
while kill -0 $PID 2>/dev/null; do sleep 60; done
{
echo "=== 爬蟲結束 $(date) ==="

echo "--- [1/5] 重建快取與特徵 $(date) ---"
python3 -m src.load_institutional 2>&1
python3 -m src.build_features 2>&1

echo "--- [2/5] 預測力檢定 $(date) ---"
python3 -m src.test_inst_predictive_power 2>&1

echo "--- [3/5] 20 檔消融 $(date) ---"
python3 -m src.ablation_institutional 60 2>&1

echo "--- [4/5] 備份舊結果 + 302 檔重訓+回測 $(date) ---"
BK="results/backup_17feat_$(date +%Y%m%d_%H%M)"
mkdir -p "$BK"
for f in accuracy_lstm_attention.csv backtest_lstm_attention.csv predictions_lstm_attention.json; do
  [ -f "results/$f" ] && mv "results/$f" "$BK/" && echo "備份 $f -> $BK/"
done
python3 -m src.run_pipeline lstm_attention 60 2>&1

echo "--- [5/5] 新舊對照 $(date) ---"
python3 -m src.compare_old_new "$BK" 2>&1
echo "=== 全部完成 $(date) ==="
} >> $LOG 2>&1
touch results/auto_chain.DONE
