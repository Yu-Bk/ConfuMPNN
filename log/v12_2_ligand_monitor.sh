#!/bin/bash
# v9 迁移训练（PID 3521281）低频防卡死监控，每 30min 检查一次。
# 只有触发以下事件才 exit（唤醒主会话），其余时间静默写状态到 monitor 文件：
#   完成 / 崩溃 / 日志异常(崩溃前兆) / 卡死
# 用法：nohup bash log/v12_2_ligand_monitor.sh &
LOG=/data/nfs/IC/baokun_yu/ConfuMPNN/log/v12_2_ligand_train.log
OUT=/data/nfs/IC/baokun_yu/ConfuMPNN/output/finetune_ligand_v12_2
MON=/data/nfs/IC/baokun_yu/ConfuMPNN/log/v12_2_ligand_train.monitor
PID=3521281

while true; do
  TS=$(date '+%m-%d %H:%M')

  # 1) 训练完成：epoch030 存在（训练正常走完会存满 30 个）
  if [ -f "$OUT/finetune_epoch030.pt" ]; then
    echo "$TS ✅ 训练完成：finetune_epoch030.pt 已生成" >> $MON
    echo "DONE" > /tmp/ligand_mon_status
    exit 0
  fi

  # 2) 进程退出但无 epoch030 → 崩溃
  if ! kill -0 $PID 2>/dev/null; then
    echo "$TS ❌ 训练异常退出（无 epoch030）！日志尾部：" >> $MON
    tail -15 $LOG >> $MON
    echo "CRASH" > /tmp/ligand_mon_status
    exit 1
  fi

  # 3) 日志异常关键词（崩溃前兆，不等进程退）。
  #    注意：不能匹配裸 "error:"——freesasa 对 UNK 残基的 "Error: Radius is <= 0" 是
  #    正常被捕获警告（跳过 L_add），2026-08-31 曾因此误报。只匹配真正的崩溃特征。
  if grep -qiE "cuda out of memory|traceback \(most recent call last\)|runtimeerror|loss[ =:]+nan" $LOG 2>/dev/null; then
    echo "$TS ⚠️ 日志出现异常关键词（OOM/报错/NaN），需要人工诊断" >> $MON
    tail -15 $LOG >> $MON
    echo "ALARM" > /tmp/ligand_mon_status
    exit 2
  fi

  # 4) 进度记录 + 卡死检测
  LATEST=$(ls -t $OUT/finetune_epoch*.pt 2>/dev/null | head -1)
  NOW=$(date +%s)
  if [ -n "$LATEST" ]; then
    EPOCH=$(basename "$LATEST" | sed 's/finetune_epoch//;s/.pt//')
    AGE=$(( (NOW - $(stat -c %Y "$LATEST")) / 60 ))
    echo "$TS ⏳ epoch $EPOCH 已生成（${AGE}min 前），当前训练中" >> $MON
    if [ $AGE -gt 120 ]; then
      echo "$TS ❌ 卡死：epoch $EPOCH checkpoint 已 ${AGE}min 未更新（单 epoch ~31min）" >> $MON
      tail -8 $LOG >> $MON
      echo "STUCK" > /tmp/ligand_mon_status
      exit 3
    fi
  else
    LAGE=$(( (NOW - $(stat -c %Y $LOG 2>/dev/null || echo 0)) / 60 ))
    echo "$TS ⏳ 尚无 epoch0（SASA 预处理中，日志 ${LAGE}min 前更新）" >> $MON
    if [ $LAGE -gt 90 ]; then
      echo "$TS ❌ 卡死：SASA 预处理阶段日志已 ${LAGE}min 未更新" >> $MON
      echo "STUCK" > /tmp/ligand_mon_status
      exit 3
    fi
  fi

  # 保持 monitor 文件最近 ~40 行
  tail -40 $MON > $MON.tmp 2>/dev/null; mv $MON.tmp $MON 2>/dev/null

  sleep 1800   # 30 分钟一次
done
