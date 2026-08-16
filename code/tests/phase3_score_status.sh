#!/bin/bash
# Phase 3 打分进度查询
# 用法：bash code/tests/phase3_score_status.sh
cd /data/nfs/IC/baokun_yu/ConfuMPNN

PID=$(pgrep -f "phase3_antidrift_score" | head -1)
if [ -z "$PID" ]; then
  echo "打分进程：未运行"
else
  echo "打分进程：PID=$PID  已运行 $(ps -o etime= -p $PID | tr -d ' ')"
fi

echo "--- 打分进度（✅=完成 ⏳=待办）---"
for d in code/output/phase3_antidrift/*/; do
  pdb=$(basename "$d")
  parts=""
  [ -f "$d/plddt.csv" ] && parts="$parts plddt✅" || parts="$parts plddt⏳"
  [ -f "$d/tm.csv" ] && parts="$parts tm✅" || parts="$parts tm⏳"
  [ -f "$d/seqs.fa-protein_sol.csv" ] && parts="$parts sol✅" || parts="$parts sol⏳"
  [ -f "$d/seqs.fa.tm.csv" ] && parts="$parts Temp✅" || parts="$parts Temp⏳"
  echo "  $pdb:$parts"
done

echo "--- 日志尾部 (code/log/phase3_score.log) ---"
tail -6 code/log/phase3_score.log 2>/dev/null || echo "（无日志）"
