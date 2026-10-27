#!/bin/bash
# 一鍵執行台灣股市情緒分析系統 — 純新聞版 (鉅亨+Yahoo, 不用 PTT)
# 用法: ./run.sh            (完整流程, 已完成步驟自動跳過)
#       ./run.sh 某步驟     (單步驟)
#       ./run.sh 某步驟 --force
set -e
cd "$(dirname "$0")"

PY=python3
if ! $PY -c "import pandas" 2>/dev/null; then
    echo "錯誤: 請使用已安裝 pandas/torch 的 python3 (conda base), 而非 /usr/bin/python3"
    exit 1
fi

D="data/processed"
R="results"

step="${1:-all}"
FORCE=0
[ "$2" = "--force" ] && FORCE=1

need() {
    if [ $FORCE -eq 1 ]; then return 0; fi
    for f in "$@"; do
        [ -e "$f" ] || return 0
    done
    return 1
}

run_cdx()        { need "$D/cdx_cnyes_list.parquet" && { $PY -m src.build_cdx_list; } || echo "跳過 cdx (已存在)"; }
run_crawl()      { $PY -m src.crawl_cnyes_wayback run; }
run_yahoo()      { $PY -m src.backfill_yahoo_dates; }
run_index()      { need "$D/news_sentiment_index.parquet" "$D/news_sector_sentiment.parquet" && { $PY -m src.build_news_sentiment_index; } || echo "跳過 index (已存在)"; }
run_test()       { $PY -m src.test_news_predictive_power; }
run_backup()     { $PY -m src.backup_ptt_results; }
run_features()   { run_backup; $PY -m src.build_features; }
run_pipeline()   { $PY -m src.run_pipeline lstm_attention 60; }
run_compare()    { $PY -m src.compare_models && cp -n "$R/model_comparison.csv" "$R/model_comparison.news.csv" && cp -n "$R/feature_ablation.csv" "$R/feature_ablation.news.csv"; }
run_autotrade()  { $PY -m src.autotrade_signals && cp -n "$R/auto_trade_signals.csv" "$R/auto_trade_signals.news.csv"; }
run_report()     { $PY -m src.final_report; }

case "$step" in
    all)     run_cdx; run_crawl; run_yahoo; run_index; run_test; run_features; run_pipeline; run_compare; run_autotrade; run_report ;;
    cdx)     run_cdx ;;
    crawl)   run_crawl ;;
    yahoo)   run_yahoo ;;
    index)   run_index ;;
    test)    run_test ;;
    backup)  run_backup ;;
    features) run_features ;;
    pipeline) run_pipeline ;;
    compare) run_compare ;;
    autotrade) run_autotrade ;;
    report)  run_report ;;
    *) echo "未知步驟: $step (cdx/crawl/yahoo/index/test/backup/features/pipeline/compare/autotrade/report/all)"; exit 1 ;;
esac
echo "完成: $step"