"""pI（等电点）查找器：二分搜索。

等电点 pI = 使分子净电荷为 0 的 pH。对给定序列是**唯一确定**的（由氨基酸
组成决定），因此在项目中作为生成序列的**推导属性**（见 PROJECT_PLAN.md
"设计哲学"），在验证阶段用作一致性检查——不直接作为模型输入。

由于净电荷是 pH 的单调递减函数（pH 升高 → 酸性残基去质子化带负电 →
净电荷下降），可以用二分搜索高效定位 pI。
"""

import torch

from .differentiable_charge import net_charge


def find_pI(seq, lo=0.0, hi=14.0, tol=1e-4, max_iter=200):
    """在 [lo, hi] 范围内二分搜索使 net_charge(seq, pH) == 0 的 pH。

    参数:
        seq: 单字母氨基酸序列
        lo, hi: 搜索范围（默认整个可行 pH 范围 [0, 14]）
        tol: 电荷精度（电荷绝对值小于该值即认为收敛）
        max_iter: 最大迭代次数（保护，避免死循环）

    返回:
        float: 序列的等电点 pI
    """
    charge_lo = net_charge(seq, lo, include_termini=True)
    charge_hi = net_charge(seq, hi, include_termini=True)
    if charge_lo * charge_hi > 0:
        # 极端情况：整条序列在搜索范围内都保持同号电荷（几乎不会出现，
        # 因为净电荷随 pH 从正单调降至负）。仍返回边界内的中点并警告。
        return 0.5 * (lo + hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        charge_mid = net_charge(seq, mid, include_termini=True)
        if abs(charge_mid) < tol:
            return mid
        if charge_mid > 0:
            lo = mid   # 当前 pH 太低（净正电），需要更高 pH
        else:
            hi = mid   # 当前 pH 太高（净负电），需要更低 pH
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    # 自检：多碱性残基的序列 pI 应偏高，多酸性残基的序列 pI 应偏低
    for seq in ["RRRRR", "KKEEDD", "ACDEFGHIKLMNPQRSTVWY", "AAAAA"]:
        print(f"pI({seq}) = {find_pI(seq):.2f}")
