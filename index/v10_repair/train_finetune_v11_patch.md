# train_finetune.py v11 补丁（2026-08-28，v10 失败修复）

> 目标文件：服务器 `code/train_finetune.py`（先 `git pull` 到 9df9cf7 一致再改）。
> 共 3 处修改（A/B/C 各一处，均为**搜索-替换**式，行号只作参考）。
> 改完跑 `python -c "import ast; ast.parse(open('code/train_finetune.py').read())"` 验证语法，再冒烟 50 域。

---

## 修改 0：新增命令行参数（在 `--structure_boost` 附近加）

**搜索**：
```python
    p.add_argument("--structure_boost", type=float, default=1.5,
```

**在它后面插入**：
```python
    p.add_argument("--decouple_absolute", action="store_true",
                   help="v11 A-fix：绝对 target 采样——与 native 无关，直接覆盖 [lo,hi]，"
                        "解决 v10 相对解耦±12 无法覆盖验证深负靶区（−19~−35）的问题")
    p.add_argument("--decouple_abs_lo", type=float, default=-35.0,
                   help="绝对 target 下界（对标验证最负靶区，默认 -35）")
    p.add_argument("--decouple_abs_hi", type=float, default=20.0,
                   help="绝对 target 上界（默认 20）")
    p.add_argument("--add_target_scale", type=float, default=1.0,
                   help="v11 B-fix：L_add 的 delta 缩放（默认 1.0；修复建议 0.5）——"
                        "表面'新增电荷数'目标与电荷损失的净电荷目标语义叠加、再叠模型原有"
                        "'删减捷径'→ 净效果≈2Δ；半量/低权避免双算")
```

---

## 修改 1（A-fix，主因修复）：绝对 target 采样

**搜索**（`train_finetune.py` 当前 decouple 块）：
```python
                if args.decouple_perturb:
                    offset = torch.where(
                        mask_p,
                        (torch.rand(B, device=device) * 2 - 1) * args.decouple_range,
                        torch.zeros(B, device=device),
                    )
                else:
```

**替换为**：
```python
                if args.decouple_absolute:
                    # v11 A-fix：绝对 target ∈ Uniform[lo, hi]，与骨架 native 无关。
                    # 依然给出 offset = target − native 语义（B 的 L_add 与分组监控依赖），
                    # 共享随后的 charge_b = charge_b + offset 流程。
                    native_b = charge_b.clone()
                    target_abs = (torch.rand(B, device=device)
                                  * (args.decouple_abs_hi - args.decouple_abs_lo)
                                  + args.decouple_abs_lo)
                    offset = torch.where(mask_p, target_abs - native_b,
                                         torch.zeros(B, device=device))
                elif args.decouple_perturb:
                    offset = torch.where(
                        mask_p,
                        (torch.rand(B, device=device) * 2 - 1) * args.decouple_range,
                        torch.zeros(B, device=device),
                    )
                else:
```

> ⚠️ 注意：下面的 `charge_b = charge_b + offset` 是共享的、不要动。
> 绝对模式下 offset 已含 `−native_b`，相加后 target = 绝对采样值，仅扰动样本生效、自洽样本保持 native。

---

## 修改 2（B-fix）：L_add 的 delta 加缩放

**搜索**：
```python
                    delta = float(offset[i].item())
                    if abs(delta) < 1.0:
                        continue  # 需求过小，不启用
```

**替换为**：
```python
                    delta = float(offset[i].item()) * args.add_target_scale
                    if abs(delta) < 1.0:
                        continue  # 需求过小，不启用
```

> 修复建议先用 `--add_target_scale 0.5 --lambda_add 0.1`（半量 + 降权），
> 之后 A7 消融里扫 `{0.25, 0.5, 1.0} × λ_add {0.1, 0.3}`。

---

## 修改 3（C-fix）：结构惩罚改逐样本 boost + 逐样本 pH

> 旧实现两个问题：① boost 只要批内有一个扰动样本就 1.5× 横批全批（含自洽样本，约 94% 步）；② 整批共用 `pH_b[0]`（批内 8 个 pH 不同，是潜在 bug）。

**搜索**（整块替换）：
```python
            struct_pen = torch.zeros((), device=device)
            if args.ph_aware_filter:
                from src.structure_aware_filter import StructureAwareFilter
                coords = dom["X"][0, :, 1].cpu().numpy()  # [L,3] Cα
                filt = StructureAwareFilter(coords)
                seq_int_cur = dom["S"][0].long().cpu().numpy()
                # 按样本方向决定是否加强（大额扰动 → 加强）
                sp, sp_info = ph_aware_structure_penalty(
                    logits, filt, seq_int_cur, pH=float(pH_b[0].item()),
                    mask=ce_mask, scale_boost=1.0,
                )
                # 对扰动样本（尤其大额）动态加强
                boost = args.structure_boost if mask_p.any().item() else 1.0
                if boost > 1.0:
                    sp2, _ = ph_aware_structure_penalty(
                        logits, filt, seq_int_cur, pH=float(pH_b[0].item()),
                        mask=ce_mask, scale_boost=boost,
                    )
                    struct_pen = sp2
                else:
                    struct_pen = sp
```

**替换为**：
```python
            struct_pen = torch.zeros((), device=device)
            if args.ph_aware_filter:
                from src.structure_aware_filter import StructureAwareFilter
                coords = dom["X"][0, :, 1].cpu().numpy()  # [L,3] Cα
                filt = StructureAwareFilter(coords)
                seq_int_cur = dom["S"][0].long().cpu().numpy()
                # v11 C-fix：逐样本 boost（只有扰动样本加强）+ 逐样本 pH
                sp_vec = torch.zeros(B, device=device)
                for i in range(B):
                    boost_i = args.structure_boost if mask_p[i].item() else 1.0
                    sp_i, _ = ph_aware_structure_penalty(
                        logits[i:i+1], filt, seq_int_cur,
                        pH=float(pH_b[i].item()),
                        mask=ce_mask[i:i+1], scale_boost=boost_i,
                    )
                    sp_vec[i] = sp_i
                struct_pen = sp_vec.mean()
```

> 提示：C 仍用 **native 序列**算 bias（它是"参考布局"），对"生成序列自身成簇"的监督仍有限——验证时必须在生成序列上补 H3 成簇统计（见计划 §12.3）。

---

## 验证流程

```bash
# 0) 语法检查
python -c "import ast; ast.parse(open('code/train_finetune.py').read())" && echo OK

# 1) 冒烟 50 域（绝对 target + 半量 B + 逐样本 C）
python code/train_finetune.py --device cuda:3 --epochs 3 --max_domains 50 \
  --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
  --labels data/cath/labels_balanced_v7.npz --dompdb data/cath/S40/dompdb \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 --charge_temp 0.5 \
  --perturb_prob 0.3 --placeholder_prob 0.15 \
  --decouple_absolute --decouple_abs_lo -35 --decouple_abs_hi 20 \
  --add_supervision --lambda_add 0.1 --add_target_scale 0.5 \
  --ph_aware_filter --structure_boost 1.5 \
  --out_dir output/finetune_v11_mompnn_smoke

# 2) 正式 30 epoch（MoMPNN 侧；配体侧把 --weights/--labels/--dompdb/--ligand 换掉）
nohup python code/train_finetune.py --device cuda:3 --epochs 30 \
  --weights MoMPNN/...ckpt \
  --labels data/cath/labels_balanced_v7.npz --dompdb data/cath/S40/dompdb \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 --charge_temp 0.5 \
  --perturb_prob 0.3 --placeholder_prob 0.15 \
  --decouple_absolute --decouple_abs_lo -35 --decouple_abs_hi 20 \
  --add_supervision --lambda_add 0.1 --add_target_scale 0.5 \
  --ph_aware_filter --structure_boost 1.5 \
  --out_dir output/finetune_v11_mompnn \
  --log_file log/v11_train_mompnn.log --log_progress log/v11_train_mompnn_prog.json \
  > log/v11_train_mompnn.stdout 2>&1 &
```

> 纪律：**先只跑诊断脚本（不改训练），再上 v11**——诊断脚本用 v10 老 checkpoint
> 就能把根因定死（训练域 slope≈1 / 验证域 slope≈2 → 本次修复方向正确）。

---

## 诊断后实验矩阵（2026-08-28 追加，按 README 执行顺序）

> 结论已修正：外推非主因；负向响应增益失控（区内 slope 1.59±0.57）由 B 双算 + decouple 弱锚导致。
> 因此**仅 A-fix 不够**。执行顺序（每版 30ep MoMPNN，先 v11a）：

```bash
# v11a = B-OFF（先跑，隔离 B；其余与 v10 完全一致）
nohup python code/train_finetune.py --device cuda:3 --epochs 30 \
  --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
  --labels data/cath/labels_balanced_v7.npz --dompdb data/cath/S40/dompdb \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 --charge_temp 0.5 \
  --perturb_prob 0.3 --placeholder_prob 0.15 \
  --decouple_perturb --decouple_range 12.0 \
  --ph_aware_filter --structure_boost 1.5 \
  --out_dir output/finetune_v11a_boff \
  ... > log/v11a.stdout 2>&1 &

# v11b = A-fix 单开（隔离 decouple；不带 B/C）
nohup python code/train_finetune.py --device cuda:3 --epochs 30 \
  --weights MoMPNN/...ckpt --labels data/cath/labels_balanced_v7.npz --dompdb data/cath/S40/dompdb \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 --charge_temp 0.5 \
  --perturb_prob 0.3 --placeholder_prob 0.15 \
  --decouple_absolute --decouple_abs_lo -35 --decouple_abs_hi 20 \
  --out_dir output/finetune_v11b_afix \
  ... > log/v11b.stdout 2>&1 &

# v11c = 全 fix（A-fix + B 半量/低权 + C 逐样本）
```
每版跑完 → 用 `v10_diag_response_curve.py`（同 manifest/targets，换 --cond_encoder）闭环：
**区内 slope∈[0.9,1.15]、正负区外 <1.3、|截距|<1 即通过**。
