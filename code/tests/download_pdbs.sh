#!/usr/bin/env bash
# 用法: bash download_pdbs.sh <candidates_file> <out_dir> [N=8]
# 从候选清单（第 2 列 PDB ID）提取唯一 PDB，并行下载整链结构。
# 优先 .pdb 格式；缺失（404/空/错误页）时降级 .cif。
set -e
LIST="$1"; OUT="$2"; N="${3:-8}"
mkdir -p "$OUT"
awk 'NR>1 && NF>=2 {print $2}' "$LIST" | sort -u > /tmp/pdb_ids_v7.txt
TOTAL=$(wc -l < /tmp/pdb_ids_v7.txt)
echo "待下载唯一 PDB: $TOTAL"
cat /tmp/pdb_ids_v7.txt | xargs -P "$N" -I{} bash -c '
  id="{}"
  f="'$OUT'/$id.pdb"
  if [ -f "$f" ] && [ -s "$f" ]; then exit 0; fi
  # -f = HTTP 错误码即失败（避免把错误页误当有效文件）；-s 静默
  if ! curl -sfL --max-time 90 -o "$f" "https://files.rcsb.org/download/$id.pdb"; then
    rm -f "$f"
    curl -sfL --max-time 90 -o "'$OUT'/$id.cif" "https://files.rcsb.org/download/$id.cif"
  fi
'
DONE=$(ls "$OUT" | wc -l)
echo "下载完成: $DONE / $TOTAL 文件（含 .cif 降级）"
