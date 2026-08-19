# analysis/archieved — 归档说明

> 按 `index/FILE_MANAGEMENT.md` 规则 4：被证伪 / 过时的方案文档移入本目录，并注明原因。

## 归档规则

1. **文档被证伪 / 被新版取代** → 移入本目录，文件名保留原名，并在文件头部加一行 `> 归档原因：...`
2. **过时的 checkpoint / 输出** → **不移入**（体积大且删除有风险），在原位置保留，在 `index/DOCUMENT_INDEX.md` 的目录填充状态中标注"过时"
3. 归档动作完成后，同步更新 `index/DOCUMENT_INDEX.md`（从主表删除或标注）

## 当前状态（2026-08-19 收尾审查）

**本轮收尾审查结果：无文档需移入归档。**

理由：
- 全部报告（`analysis/report/`，E1 → v9 泛化验证）是**历史记录**，如实记录迭代过程，不删除、不归档。
- 规划文档（`index/PROJECT_PLAN.md` / `PROJECT_EXTEND.md` / `PROJECT_V9_*`）是宏观规划与决策记录，保留。
- `docs/TECH/CONFIG/USAGE.md` 已更新至 v9 现状（非过时）。
- **旧 checkpoint**（`code/output/finetune_vN`，v2–v6 共 30+ 目录）保留原位，已在 `index/DOCUMENT_INDEX.md` 目录填充状态中标注"过时，最终版为 `output/finetune_v7|v9`"。

若后续出现被证伪的方案文档，移入此处并登记原因。
