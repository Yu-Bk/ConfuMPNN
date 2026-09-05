#!/usr/bin/env python
"""命中率点估计 + Wilson 95% 置信区间（论文统计用，2026-09-06）。

用法（交互/脚本均可导入）：
  from hitrate_ci import wilson_ci
  wilson_ci(k, n) -> (rate, lo, hi)

对 n≈50/臂 的命中率，Wilson 区间显著优于正态近似（尤其率接近 0/1 时）。
"""
import math


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for proportion k/n."""
    if n <= 0:
        return (None, None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson_ci(k, n, alpha=0.05):
    """Clopper-Pearson (exact) 95% CI for k/n."""
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (k / n, lo, hi)


def fmt(k, n, method="wilson"):
    fn = wilson_ci if method == "wilson" else clopper_pearson_ci
    p, lo, hi = fn(k, n)
    return f"{k}/{n} = {p*100:.1f}%  [{lo*100:.1f}%, {hi*100:.1f}%]"


if __name__ == "__main__":
    import json, sys
    # 默认打几个论文关键 n50 命中率
    cases = {
        "v14-clean H2(总 45/50)": (45, 50),
        "v14-clean per-protein 5/5": (5, 5),
        "v14-clean 2FEO 0/5": (0, 5),
        "v13-in10 H2(32/50)": (32, 50),
        "7K00 native H2(26/46)": (26, 46),
        "Task3 三达标单条(约 5.2%)": (520, 10000),
    }
    out = {}
    for name, (k, n) in cases.items():
        row = {"wilson": wilson_ci(k, n), "exact": clopper_pearson_ci(k, n)}
        out[name] = row
        print(f"{name}: {fmt(k,n,'wilson')}  |  exact {fmt(k,n,'exact')}")
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
        print(f"已写 {sys.argv[1]}")
