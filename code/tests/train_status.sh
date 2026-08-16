#!/bin/bash
# 微调训练进度查询入口（后台训练时用）
# 用法：bash code/tests/train_status.sh
cd /data/nfs/IC/baokun_yu/ConfuMPNN

PID=$(pgrep -f "train_finetune" | head -1)
if [ -z "$PID" ]; then
  echo "训练进程：未运行"
else
  echo "训练进程：PID=$PID  已运行 $(ps -o etime= -p $PID | tr -d ' ')"
  echo "GPU 占用：$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)"
fi

echo "--- 进度文件 (log/train_progress.json) ---"
if [ -f log/train_progress.json ]; then
  python3 -c "import json; d=json.load(open('log/train_progress.json')); print(f\"epoch {d.get('epoch')}/{d.get('total_epochs')}  loss={d.get('loss'):.4f}  时间={d.get('elapsed_min',0):.1f}min\")" 2>/dev/null || cat log/train_progress.json
else
  echo "（尚无进度文件）"
fi

echo "--- 日志尾部 (log/train.log) ---"
tail -8 log/train.log 2>/dev/null || echo "（无日志）"
