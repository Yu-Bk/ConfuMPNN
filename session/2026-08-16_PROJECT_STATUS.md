# 项目进展快照 — ConfuMPNN

> 汇总性进展记录（最后一次更新 2026-08-16）。明天继续时从本文件开始恢复上下文。
> 细粒度会话记录：`session/2026-08-15_phase1_modules.md`、`session/2026-08-16_charge_lookahead_fix.md`

---

## 一、项目一句话

把**工作环境 pH（及净电荷/局部电荷）作为条件约束**，整合进 LigandMPNN 结构逆折叠模型，生成「符合 pH 电荷约束」的蛋白序列。

两级计划：
- **第一版** `index/PROJECT_PLAN.md` — pH 电荷条件生成主线（Phase 0–4）
- **第二版拓展** `index/PROJECT_EXTEND.md` — 多目标可开发性微调（可设计/热稳/可溶），优先用开源 MoMPNN，把更好模型放回主线管线

---

## 二、当前阶段：Phase 1 已全部完成 ✅

**里程碑已达成**：不改模型代码，纯 logit bias 采样策略实现 pH 感知电荷约束生成。

### 已完成模块（`code/src/`，全部通过测试）
| 模块 | 作用 |
|------|------|
| `pka.py` | 侧链/末端 pKa 表、AA 索引、带电类型 |
| `differentiable_charge.py` | sigmoid 平滑 HH 方程，`net_charge`（字符串）/`net_charge_from_logits`（可微） |
| `isoelectric_point.py` | `find_pI` 二分搜索（验证用） |
| `structure_aware_filter.py` | 4 条结构规则 → [L,21] bias，YAML 预设 |
| `condition_embedding.py` | Soft Prompt MLP + mask-aware 条件向量 [7]（Phase 2 用） |
| `losses.py` | 复合损失 CE+电荷偏差+结构惩罚+DPO+margin（Phase 2 用） |
| `guided_sampler.py` | 静态/动态 bias 解码，包装 LigandMPNN |
| `charge_lookahead.py` | **动态电荷前瞻**：每步 bias = strength·(target−Q_current)·q_k |

### 一键入口与测试
- `code/run_guided.py` — 完整管线：PDB→filter→引导采样→电荷/pI 统计→fasta+json
- `code/tests/test_all.py` — **36 项全通过**
- `code/tests/smoke_guided.py` — 真实 LigandMPNN + 1BC8.pdb 冒烟通过

### 本次会话关键成果（2026-08-16）
1. **修复 charge_lookahead target 失效 bug**（提交 `6d76da4`）：
   - 根因：`bias=-strength·(Q_k−target)` 中 target 落在常数项，被 softmax 平移不变性抵消
   - 修复：`bias_k = strength·(target−Q_current)·q_k`，target 进入交叉项
   - **验证**：1BC8 pH7.4，target=+8/0/−8 → 平均净电荷 **+8.06 / +0.23 / −7.96**，精准命中；叠加结构过滤器、弱强度均正常
2. **第二版拓展计划** `index/PROJECT_EXTEND.md`（提交 `cac283f`）：
   - 路线 A：直接用开源 **MoMPNN**（ProtAlign ICLR 2026，GitHub: Qivon7/MoMPNN，多目标 DPO：可设计+溶解+热稳）
   - 路线 B：按 ProtAlign 方法自微调；路线 C：兜底自研
   - 第一版 `PROJECT_PLAN.md` 三处加指针，两版形成整体

---

## 三、待办（下一步）

按优先级：

1. **第二版 Stage E0（建议最先做）**：clone MoMPNN 仓库（`https://github.com/Qivon7/MoMPNN`），检查权重与 LigandMPNN 兼容性（是否含配体上下文 / 能否 `load_state_dict` 进我们的 `ProteinMPNN` 类），输出可用性结论到 `analysis/`
2. **第一版 Phase 1 收尾待办**（PROJECT_PLAN 中列出）：
   - 阈值统计：PDB 采样 1000 条确定结构过滤器 99 分位默认阈值
   - 3–5 个示例蛋白不同 pH/预设的对比实验
3. **第一版 Phase 2（条件编码器微调）**：从训练集计算条件向量标准化 μ/σ → 写入 `condition_defaults.yaml`；A100 微调
4. **第二版 E4 集成**：微调模型设为 `run_guided.py` 默认生成器 + 对照实验

---

## 四、运行速查

```bash
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn          # Python 3.11, torch 2.2.1+cu121
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
python tests/test_all.py          # 36 项单元测试
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --num_samples 5 --strength 0.5 --out_dir output/guided_1BC8_pH7.4/target_0
```

环境细节见 `CLAUDE.md` 与 memory `confumpnn-env-setup.md`。
ESMFold 回折在 `confumpnn-esmfold` 环境（conda, Python 3.10, torch 2.6.0+cu124，openfold 依赖需确认）。

---

## 五、Git 状态

- 分支 `main`，远程 `origin` = git@github.com:Yu-Bk/ConfuMPNN.git（未推送，本地领先 origin 若干提交）
- 最近提交：`6d76da4`（charge_lookahead 修复）← `cac283f`（第二版计划+代码）← `2d834a2`（Phase 1 核心）
- `LigandMPNN/`、`foundry/` 为 clone 源码不跟踪；`code/output/`、`code/log/`、`*.pt` 已 gitignore
