# ConfuMPNN — 文件管理规范

> **来源**：workspace 根目录 `/data/nfs/IC/baokun_yu/logical_chain.md`。
> 本文件是该项目文件分类存放的**唯一规则**，项目进行过程中的每一次实验、每一份文档都必须遵守。
> 若 logical_chain.md 更新，本文件需同步更新。
> 所有文档的定位见 [[DOCUMENT_INDEX.md]]。

---

## 目录总览

```
ConfuMPNN/
├── code/                     # 1. 实验模块代码 + 整体流程代码
│   ├── input/                #    输入数据（pdb 文件等）
│   ├── output/               #    输出结果
│   └── log/                  #    测试脚本的运行日志
├── analysis/                 # 2. 所有实验对结果的分析报告
│   ├── report/               #    最新的实验报告
│   ├── archieved/            #    过时的文档、被证伪的方案
│   ├── accident/             #    意外实验 / 人为制造的意外实验
│   │   ├── report/           #      意外现象的分析报告
│   │   └── root/             #      意外现象的根源定位
│   └── ablation/             #    验证时用的消融实验（按实验建子文件夹）
├── index/                    # 3. 所有文档的定位 + 关键决策 + 方向调整 + 宏观规划 + 论文规划
│   ├── PROJECT_PLAN.md       #    项目整体规划（已存在）
│   ├── FILE_MANAGEMENT.md    #    本文件：文件管理规范
│   └── DOCUMENT_INDEX.md     #    文档定位索引
├── literature/               # 4. 参考论文的笔记（所有论文分类存放）
│   ├── baseline/             #    论文全景分析得到的流程管线
│   ├── innovation/           #    论文 baseline 中的创新方式和创新模块
│   ├── pattern/              #    不同论文提取的规律 + 工作量分析
│   ├── tools/                #    论文用了什么外部工具、解决什么问题
│   └── phenomena/            #    论文中常见实验现象以及意外现象（含尝试解决的部分）
├── session/                  # 5. Claude Code 对话概要 + 导出的文档
└── source/                   # 6. 所有论文的开源源码或链接
```

> `LigandMPNN/`、`foundry/` 是克隆的官方源码，不属于上述分类体系，统一放项目根，**不提交 git**（见 `.gitignore`）。

---

## 规则详解

### 1. `code/` — 代码
- 放实验模块的代码脚本和整体流程的代码等。
- `code/input` 存放输入数据（pdb 文件等）；`code/output` 存放输出结果；`code/log` 存放测试脚本的运行日志。
- **不同模块的实验分类存放**（按模块/阶段建子文件夹）。
- 对像 PDB 文件这种**每次运行都需要输入**的文件，可以不进行子分类，直接放 `code/input`。

### 2. `analysis/` — 实验结果分析
- 存放所有实验对结果的分析报告。
- `analysis/report` 存放**最新的**实验报告。
- `analysis/archieved` 存放项目中过时的文档和**被证伪的方案**。
- `analysis/accident` 存放实验过程中出现的意外实验，或需要**人为制造的意外实验**。
  - `analysis/accident/report` 存放对这些意外现象的分析报告。
  - `analysis/accident/root` 存放分析定位得到的这些意外实验现象的**根源**。
  - 在意外实验中用到的代码、建立的因果链等笔记，直接放在 `analysis/accident` 即可。
- `analysis/ablation` 中**建立子文件夹**存放验证时用的消融实验的部分。

### 3. `index/` — 索引与规划
- 存放该项目中**所有文档的定位**（见 `DOCUMENT_INDEX.md`）。
- 存放项目的**关键决策和方向调整**（决策一旦做出，记录于此）。
- 存放项目整体的**宏观规划**，以及后续**写论文的规划**。
- `PROJECT_PLAN.md` 已存放于该目录下。

### 4. `literature/` — 论文笔记
- 存放项目在起始或进行过程中所参考的论文笔记，所有论文**分类存放**。
- `literature/baseline`：对论文进行全景分析得到的**流程管线**。
- `literature/innovation`：论文 baseline 中的**创新方式和创新模块**。
- `literature/pattern`：不同论文提取的**规律以及工作量分析**。
- `literature/tools`：论文用了什么**外部工具**及相应解决什么问题。
- `literature/phenomena`：论文中**常见实验现象以及意外现象**（包括尝试解决的部分）。

### 5. `session/` — 会话记录
- 记录和 Claude Code 的对话概要，以及导出的文档。

### 6. `source/` — 源码
- 放所有论文的开源源码或链接。

---

## 实验进行的规则（工作流约定）

为了让上面的目录真正成为"实验进行的规则"，约定每次实验都按以下流程走：

1. **实验开始前**：在 `code/` 下按模块建好子目录（或复用已有目录），明确本实验属于哪个模块。
2. **实验运行中**：输入数据放 `code/input`，脚本输出写 `code/output`，测试脚本的日志写 `code/log`。
3. **实验跑完**：**立即**在 `analysis/report/` 写一份分析报告（结论明确，附证据）。
4. **被证伪 / 过时**：对应的方案文档从 `report` 移入 `analysis/archieved/`，并注明原因。
5. **意外现象**：在 `analysis/accident/` 下记录现象 + 因果链笔记；分析报告放 `accident/report/`；根因定位后放 `accident/root/`。
6. **消融实验**：在 `analysis/ablation/<实验名>/` 下建子文件夹存放。
7. **读论文**：按五个维度（baseline / innovation / pattern / tools / phenomena）分别做笔记放入 `literature/` 对应子目录；论文源码链接或 clone 登记到 `source/`。
8. **每次与 Claude Code 的重要对话**：在 `session/` 写一篇概要（做了什么、关键决策、遗留问题）。
9. **关键决策 / 方向调整 / 规划变更**：更新到 `index/` 下的对应文档。
10. **任何新增文档**：同步登记到 `index/DOCUMENT_INDEX.md`。
