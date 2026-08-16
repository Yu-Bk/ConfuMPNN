#!/bin/bash
# 微调训练进度查询入口（后台训练时用）
# 用法：bash code/tests/train_status.sh
cd /data/nfs/IC/baokun_yu/ConfuMPNN

PID=$(pgrep -f "train_finetune" | head -1)
if [ -z "$PID" ]; then
  echo "训练进程：未运行"
else
  echo "训练进程：PID=$PID  已运行 $(ps -o etime= -p $PID | tr -d ' ')"
  # 从启动命令里解析 cuda:N，查询对应的 GPU（默认 cuda:1）
  DEV=$(ps -o args= -p $PID | grep -oE 'cuda:[0-9]+' | head -1 | cut -d: -f2)
  if [ -z "$DEV" ]; then DEV=1; fi
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader -i "$DEV" | awk -F',' -v g="$DEV" \
    '{print "GPU " g "：占用 " $2 "，显存 " $3 "/" $4}'
fi

echo "--- 进度文件 (code/log/train_progress.json) ---"
if [ -f code/log/train_progress.json ]; then
  python3 -c '
import json
d = json.load(open("code/log/train_progress.json"))
print("epoch {}/{}  loss={:.4f}  ce={:.4f}  charge={:.4f}  kl={:.4f}  已耗时 {:.1f} min".format(
    d.get("epoch"), d.get("total_epochs"), d.get("loss", 0), d.get("ce", 0),
    d.get("charge", 0), d.get("kl", 0), d.get("elapsed_min", 0)))
' 2>/dev/null || cat code/log/train_progress.json
else
  echo "（尚无进度文件——训练仍在预解析阶段或刚启动）"
fi

echo "--- 日志尾部 (code/log/train.log) ---"
tail -8 code/log/train.log 2>/dev/null || echo "（无日志）"
