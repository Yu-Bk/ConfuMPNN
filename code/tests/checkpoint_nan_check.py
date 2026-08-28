"""Checkpoint NaN/Inf 复检工具：遍历 checkpoint 全部 tensor，统计 NaN/Inf 数量。

背景：v10 训练曾因 1GTV 不完整结构 → freesasa NaN → L_add 污染 → 权重全 NaN
（2026-08-27，见 analysis/report/2026-08-27_v10_ligand_training.md §3.1）。
本工具在**验证前**对 checkpoint 做 NaN/Inf 复检，PASS 才放行泛化验证
（对应 run_v10_pipeline.sh 阶段 3.6 的检查逻辑，这里做成独立可复现脚本）。

用法：
  python checkpoint_nan_check.py <ckpt1> [ckpt2 ...]
输出（stdout，可重定向到文件）：
  <path>: NaN=<n> Inf=<n> PASS|FAIL
  对含 NaN/Inf 的 tensor 逐项列出 key（前 20 个）
"""
import sys

import torch


def check(path):
    """返回 (total_nan, total_inf, bad_list)。bad_list = [(key, nan, inf), ...]"""
    ck = torch.load(path, map_location="cpu")
    total_nan = total_inf = 0
    bad = []

    def walk(d, prefix=""):
        nonlocal total_nan, total_inf
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif torch.is_tensor(d):
            nan = int(torch.isnan(d).sum().item())
            inf = int(torch.isinf(d).sum().item())
            if nan or inf:
                bad.append((prefix, nan, inf))
            total_nan += nan
            total_inf += inf

    walk(ck)
    return total_nan, total_inf, bad


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python checkpoint_nan_check.py <ckpt1> [ckpt2 ...]")
        sys.exit(1)
    all_pass = True
    for p in sys.argv[1:]:
        nan, inf, bad = check(p)
        ok = (nan == 0 and inf == 0)
        all_pass = all_pass and ok
        print(f"{p}: NaN={nan} Inf={inf} {'PASS' if ok else 'FAIL'}")
        for name, n, i in bad[:20]:
            print(f"  {name}: NaN={n} Inf={i}")
    print("== 全部 PASS 可放行验证 ==" if all_pass else "== ⚠️ 存在 NaN/Inf，禁止进入验证 ==")
    sys.exit(0 if all_pass else 1)
