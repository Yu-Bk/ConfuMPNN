#!/usr/bin/env bash
# 用法: bash curl_parallel_download.sh <url> <out> [N=8]
# Range 分段并行下载（绕过单连接限速）。不依赖 Python 库。
set -e
URL="$1"; OUT="$2"; N="${3:-8}"
SIZE=$(curl -sIL "$URL" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{v=$2} END{print v+0}')
if [ -z "$SIZE" ] || [ "$SIZE" -le 0 ]; then echo "无法获取文件大小"; exit 1; fi
echo "总大小 $SIZE 字节，$N 段并行"
CHUNK=$(( SIZE / N ))
rm -f "${OUT}".part*
pids=()
for i in $(seq 0 $((N-1))); do
  START=$((i*CHUNK))
  if [ $i -eq $((N-1)) ]; then END=$((SIZE-1)); else END=$(((i+1)*CHUNK-1)); fi
  curl -sL -r "${START}-${END}" -o "${OUT}.part${i}" "$URL" &
  pids+=($!)
done
ok=0
for p in "${pids[@]}"; do wait "$p" && ok=$((ok+1)); done
echo "$ok/$N 段成功"
if [ "$ok" -ne "$N" ]; then echo "有失败段，请重试"; exit 1; fi
cat "${OUT}".part* > "$OUT"
rm -f "${OUT}".part*
echo "完成: $(stat -c%s "$OUT") 字节"
