# ConfuMPNN — 文档索引

> 本文件是项目**所有文档的定位索引**（规则见 `FILE_MANAGEMENT.md` 第 3 条）。
> **⚠️ 新增 / 移动 / 删除任何文档后，必须同步更新本文件。**

## 当前项目文档

| 文档 | 位置 | 说明 | 最后更新 |
|------|------|------|---------|
| 项目计划（第一版） | `index/PROJECT_PLAN.md` | 完整技术计划（文献调研 / 分阶段实施 / 风险表 / 决策记录 / 输出清单） | 2026-08-16 |
| 项目拓展（第二版） | `index/PROJECT_EXTEND.md` | 多目标可开发性微调（可设计/热稳定/可溶）；MoMPNN 接入 → 自微调 → 集成回主线 | 2026-08-16 |
| 项目说明 | `CLAUDE.md`（项目根） | Claude Code 项目级说明：环境 / 常用命令 / 文件结构 / 下一步 | 2026-08-15 |
| 文件管理规范 | `index/FILE_MANAGEMENT.md` | 文件分类存放的唯一规则 | 2026-08-15 |
| 文档索引 | `index/DOCUMENT_INDEX.md` | 本文件 | 2026-08-15 |
| README | `README.md` | 对外简介（含文件结构） | 2026-08-15 |

## Phase 1 代码模块（`code/`）

| 文档 | 位置 | 说明 |
|------|------|------|
| pKa 常量表 | `code/src/pka.py` | 侧链/末端 pKa、AA 索引、带电类型 |
| 可微电荷计算 | `code/src/differentiable_charge.py` | HH 方程平滑近似：`net_charge`（字符串）/ `net_charge_from_logits`（可微） |
| pI 查找器 | `code/src/isoelectric_point.py` | `find_pI` 二分搜索（验证用） |
| 结构感知过滤器 | `code/src/structure_aware_filter.py` | 4 条规则 → [L,21] bias；`load_preset` 读 YAML |
| 条件编码器 | `code/src/condition_embedding.py` | Soft Prompt MLP + mask-aware 条件向量（Phase 2） |
| 复合损失 | `code/src/losses.py` | CE + 电荷偏差 + 结构惩罚 + DPO + margin（Phase 2） |
| 引导采样器 | `code/src/guided_sampler.py` | 静态/动态 bias 解码，包装 LigandMPNN |
| 过滤器预设 | `code/configs/filter_presets.yaml` | default / nucleic_acid_binding / membrane / acidic |
| 条件默认配置 | `code/configs/condition_defaults.yaml` | 条件向量/标准化/编码器参数（Phase 2） |
| 单元测试 | `code/tests/test_all.py` | 29 项，全通过 |
| 冒烟测试 | `code/tests/smoke_guided.py` | 真实 LigandMPNN + 1BC8.pdb，通过 |

## 目录填充状态

| 目录 | 用途 | 当前内容 | 状态 |
|------|------|---------|------|
| `code/` | 实验模块代码 | `src/`（7 模块）+ `configs/`（2 yaml）+ `tests/`（2 脚本）+ `input/`（1BC8.pdb）+ `output/`（smoke 结果）+ `log/`（测试日志） | ✅ Phase 1 模块就绪 |
| `analysis/` | 实验结果分析 | （空） | ⬜ 待实验产生 |
| `literature/` | 论文笔记 | （空） | ⬜ 待论文笔记导入 |
| `session/` | 会话记录 | （空） | ⬜ 待记录 |
| `source/` | 论文源码/链接 | （空） | ⬜ 待填充 |

## 约定

- 文档归属不定时，先按 `FILE_MANAGEMENT.md` 的目录规则判断，拿不准就放 `index/` 并在本表登记。
- 论文笔记导入时，按 `literature/` 五个维度（baseline/innovation/pattern/tools/phenomena）分类，并在此登记。
