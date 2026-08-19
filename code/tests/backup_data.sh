#!/usr/bin/env bash
# ConfuMPNN 数据备份脚本：把 data/ 关键数据集打包为 tar.gz + SHA256（传组内 NAS）。
#
# 打包内容：data/cath（CATH 结构域 + v7 标签）、data/ligand_train（配体复合物 + v9 标签）、
#           data/validation_pdbs（v9 泛化验证）、data/ligand_test、data/transfer_test
#           + data/README.md + data/SHA256SUMS.txt
#
# 用法：
#   bash code/tests/backup_data.sh [输出目录]   # 默认 <项目父目录>/ConfuMPNN_backup/
#
# 产物：
#   <输出目录>/confumpnn_data_v1_<日期>.tar.gz
#   <输出目录>/confumpnn_data_v1_<日期>.tar.gz.sha256   # tar 包校验和（完整性保证）
#
# 上传到组内 NAS 后，新机器恢复方式见 data/README.md §6。
set -euo pipefail

# ConfuMPNN 根目录（脚本在 code/tests/ 下，上溯两级）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-$ROOT/../ConfuMPNN_backup}"
STAMP="$(date +%Y%m%d)"
OUT_TAR="$OUT_DIR/confumpnn_data_v1_$STAMP.tar.gz"

mkdir -p "$OUT_DIR"

echo "== ConfuMPNN 数据备份 =="
echo "源目录：$ROOT/data"
echo "输出：$OUT_TAR"
echo ""

# 检查数据在位
for d in cath ligand_train validation_pdbs ligand_test transfer_test; do
  [ -d "$ROOT/data/$d" ] || { echo "❌ 缺少 data/$d，中止"; exit 1; }
done

echo "[1/3] 打包 data/（含 symlink 目录 all_pdb / dompdb_pdb，tar 默认保留链接不重复复制）..."
tar -czf "$OUT_TAR" -C "$ROOT" \
    data/cath \
    data/ligand_train \
    data/validation_pdbs \
    data/ligand_test \
    data/transfer_test \
    data/README.md \
    data/SHA256SUMS.txt
echo "    完成：$(du -h "$OUT_TAR" | cut -f1)"

echo "[2/3] 生成 tar 包 SHA256..."
( cd "$OUT_DIR" && sha256sum "$(basename "$OUT_TAR")" > "$(basename "$OUT_TAR").sha256" )

echo "[3/3] 完成 ✅"
ls -lh "$OUT_TAR" "$OUT_TAR.sha256"
echo ""
echo "下一步：把 tar.gz 上传到组内 NAS（如 /data/nfs/.../ConfuMPNN_data/）。"
echo "恢复命令见 data/README.md §6。"
