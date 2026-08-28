"""v10 诊断结果分析：分段拟合"训练覆盖区内 vs 区外外推"斜率。

读取 output/v10_diag_response.json，对每个蛋白：
  - 全区 slope（脚本已算）
  - 训练覆盖区内 slope：target ∈ [native−12, native+12]（v10 相对解耦±12 的覆盖域）
  - 负向外推区 slope：target < native−12（靶区深负侧，验证 1A65/1AXW 所在域）
  - 正向外推区 slope：target > native+12
输出汇总表 + 结论判读（对应 index/v10_repair/README.md 判据表）。

判读逻辑：
  若 区内 slope≈1 且 区外 slope≈2 → 外推假说坐实（v11 改 target 覆盖即够）
  若 区内 slope≈1.5~2   → 模型响应整体坏（B/C 叠加），v11 只改覆盖不够
"""
import json
import sys
from pathlib import Path

import numpy as np


def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = np.mean(xs), np.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    a = sxy / sxx if sxx > 1e-9 else float("nan")
    b = my - a * mx
    r2 = (sxy ** 2 / (sxx * syy)) if sxx > 1e-9 and syy > 1e-9 else float("nan")
    return a, b, r2


def main(path):
    d = json.load(open(path))
    print(f"meta: {d['meta']}\n")
    hdr = (f"{'name':10s} {'group':9s} {'L':>4s} {'native':>7s} | "
           f"{'slope_all':>9s} {'int_all':>7s} | "
           f"{'slope_in':>8s} {'n_in':>4s} | {'slope_negout':>11s} {'n_neg':>4s} | "
           f"{'slope_posout':>11s} {'n_pos':>4s}")
    print(hdr)
    print("-" * len(hdr))
    summary = {"trainish": [], "valid": []}
    for name, p in d["proteins"].items():
        g = p["group"]
        native = p["native_charge"]
        t = np.array(p["targets"])
        m = np.array(p["mean_charge"])
        slope_all, int_all, _ = p["slope"], p["intercept"], p["r2"]

        # 区内：|t−native| ≤ 12
        in_mask = np.abs(t - native) <= 12.0
        s_in, b_in, _ = linfit(t[in_mask].tolist(), m[in_mask].tolist()) if in_mask.sum() >= 2 else (np.nan, np.nan, np.nan)
        # 负向外推：t < native−12
        neg_mask = t < native - 12.0
        s_neg, _, _ = linfit(t[neg_mask].tolist(), m[neg_mask].tolist()) if neg_mask.sum() >= 2 else (np.nan, np.nan, np.nan)
        # 正向外推：t > native+12
        pos_mask = t > native + 12.0
        s_pos, _, _ = linfit(t[pos_mask].tolist(), m[pos_mask].tolist()) if pos_mask.sum() >= 2 else (np.nan, np.nan, np.nan)

        row = (f"{name:10s} {g:9s} {p['L']:4d} {native:7.1f} | "
               f"{slope_all:9.2f} {int_all:7.1f} | "
               f"{s_in:8.2f} {int(in_mask.sum()):4d} | "
               f"{s_neg:11.2f} {int(neg_mask.sum()):4d} | "
               f"{s_pos:11.2f} {int(pos_mask.sum()):4d}")
        print(row)
        summary[g].append(s_in)

    print("\n=== 区内 slope 均值（训练域 vs 验证域）===")
    for g, vals in summary.items():
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            print(f"  {g:9s}: n={len(vals):2d}  区内 slope 均值={np.mean(vals):.2f} ± {np.std(vals):.2f}")
    print("\n判读：")
    print("  区内 slope≈1 且 区外≈2 → 外推假说坐实，v11 改 target 覆盖即可")
    print("  区内 slope≈1.5~2   → 模型响应整体坏（B/C 叠加），需拆组件重训")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/v10_diag_response.json")
