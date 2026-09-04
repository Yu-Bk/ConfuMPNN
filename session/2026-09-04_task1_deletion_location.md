# 2026-09-04 Task1 删除定位分析（session 记录）

- 任务：v14 组成删减定位（CPU 诊断，零新采样）
- 脚本：code/tests/ligand_v9/deletion_location_analysis.py
- 产出：output/v14_deletion_location.json + analysis/report/2026-09-04_v14_deletion_location.md
- 口径：pocket=Cα-配体≤8Å；surface=frac_sasa≥0.25；core=其余；带电=DEKR；保留率=gen(n=50)均值/native
- v13 对比仅 5 共享单体；RNA/DNA 与 6D2O 无 v13 基线
- 状态：完成（主会话统一 git 归档，本 agent 不 commit）
