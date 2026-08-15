"""动态电荷前瞻：每一步对候选氨基酸做电荷 lookahead，输出 logit bias。

对应 PROJECT_PLAN.md 4.1 Step 3 的"可微净电荷 lookahead：每一步对候选
氨基酸做精确的前瞻计算"。这是 Phase 1 引导采样的**核心创新**——让模型在
解码每一步时"看到"选择某个氨基酸会把整条序列的净电荷推向哪个方向，
从而把"净电荷靠近目标值"转化为逐候选的 logit bias。

原理（电荷可加性，O(20·L) 轻量）：
    对当前位置，分别假设放 20 种标准氨基酸，整条序列的期望净电荷为：

        Q_k = Q_fixed + q(aa_k, pH) + Q_expect_others + Q_termini

    - Q_fixed        : 已解码位置的侧链电荷之和（用实际选择的残基）
    - q(aa_k, pH)    : 候选氨基酸在 pH 下的侧链电荷
    - Q_expect_others: 其余**未解码**位置的期望电荷（用 20 种 AA 平均电荷
                       近似；不依赖模型概率，轻量且稳定）
    - Q_termini      : N/C 端常数（~±1，只依赖 pH）

    然后把"偏离目标电荷"翻译成 bias：

        bias_k = -strength · (Q_k − target_charge)

    当 Q_k 高于目标（净电荷偏正）→ bias_k 为负（抑制该候选）；
    低于目标（偏负）→ 为正（促进）。整体把净电荷拉向目标值。
    strength 控制引导强度（过强会破坏模型先验，建议从 0.2–0.5 起步）。

与结构感知过滤器是正交约束：电荷 lookahead 管"总量"，过滤器管"空间分布"，
两者可叠加（见 make_dynamic_callback）。
"""

import numpy as np

from .differentiable_charge import (_termini_charge,
                                    sidechain_charge_vector)
from .pka import AAS, PKA_SIDECHAIN

# 未解码标记（20 = X）
UNDECODED = 20


class ChargeLookahead:
    """动态电荷前瞻 bias 计算器。

    参数:
        pH: 工作环境 pH（决定每个残基的电荷）
        target_charge: 目标净电荷；None 表示不引导电荷（bias 恒 0）
        strength: 引导强度（bias 缩放系数）
        include_termini: 是否计入 N/C 端电荷（默认 True）
    """

    def __init__(self, pH, target_charge=None, strength=0.5, include_termini=True):
        self.pH = float(pH)
        self.target_charge = target_charge
        self.strength = float(strength)
        self.include_termini = include_termini
        # 预计算：20 种 AA 在 pH 下的侧链电荷向量 [20] 及其均值
        self._q_vec = sidechain_charge_vector(self.pH).numpy()
        self._q_mean = float(self._q_vec.mean())

    def _charge_of(self, aa_idx):
        """位置 i 的侧链电荷（aa_idx 是 AAS 索引）。"""
        return float(self._q_vec[aa_idx])

    def bias_at(self, seq_int, position, mask=None):
        """计算 position 位置的逐候选 bias。

        参数:
            seq_int: [L] 当前部分解码序列（int，20=X 未解码）
            position: 当前待解码位置
            mask: [L] 有效残基掩码（1=有效）；None 全有效
        返回:
            [21] numpy 数组（前 20 为各 AA 的 bias，最后 1 位 X 恒 0）
        """
        if self.target_charge is None:
            return np.zeros(21, dtype=np.float32)

        seq_int = np.asarray(seq_int)
        L = len(seq_int)
        if mask is None:
            mask = np.ones(L, dtype=bool)
        mask = np.asarray(mask, dtype=bool)

        # 1) 已解码位置（排除 position 与无效位置）的固定电荷
        fixed = 0.0
        n_undecoded_excl = 0  # 其余未解码位置数
        for i in range(L):
            if i == position or not mask[i]:
                continue
            if seq_int[i] != UNDECODED:
                fixed += self._charge_of(int(seq_int[i]))
            else:
                n_undecoded_excl += 1

        # 2) 其余未解码位置的期望电荷（用平均侧链电荷近似）
        expect_others = n_undecoded_excl * self._q_mean

        # 3) 末端电荷（若整个序列非空）
        termini = 0.0
        if self.include_termini and mask.sum() > 0:
            termini = float(_termini_charge(self.pH, n_term=True).item()) + float(
                _termini_charge(self.pH, n_term=False).item()
            )

        # 4) 对 20 个候选计算 bias
        biases = np.zeros(21, dtype=np.float32)
        for k in range(20):
            q_k = fixed + self._charge_of(k) + expect_others + termini
            biases[k] = -self.strength * (q_k - self.target_charge)
        return biases


def make_dynamic_callback(pH, target_charge=None, structure_filter=None,
                          strength=0.5):
    """构造动态 bias 回调，供 GuidedSampler 使用。

    合并两条正交约束：
      - 电荷 lookahead（管"净电荷总量"）
      - 结构感知过滤器（管"电荷空间分布"，可选）

    返回的函数签名匹配 guided_sampler 的 bias_callback：
        fn(S_cur, t) -> [21] numpy

    参数:
        pH: 工作环境 pH
        target_charge: 目标净电荷（None = 不引导电荷）
        structure_filter: StructureAwareFilter 实例（None = 不用结构过滤）
        strength: 电荷引导强度
    """
    lookahead = ChargeLookahead(
        pH, target_charge=target_charge, strength=strength
    )

    def callback(S_cur, t):
        bias = lookahead.bias_at(S_cur, t)
        if structure_filter is not None:
            fb, _ = structure_filter.compute_bias(S_cur)
            bias = bias + fb[t].numpy()
        return bias

    return callback


if __name__ == "__main__":
    # 自检：全 K 序列 + 目标净电荷 0 → 当前位置的碱性候选（K/R）应被抑制
    import numpy as np
    from .pka import AA_TO_IDX

    la = ChargeLookahead(pH=7.4, target_charge=0.0, strength=0.5)
    seq = [AA_TO_IDX["K"]] * 5 + [UNDECODED] * 5  # 前 5 K，后 5 未解码
    b = la.bias_at(np.array(seq), position=5)
    k_idx = AA_TO_IDX["K"]
    d_idx = AA_TO_IDX["D"]
    print(f"bias[K] = {b[k_idx]:+.3f}（应 < 0，抑制碱性）")
    print(f"bias[D] = {b[d_idx]:+.3f}（应 > 0，促进酸性）")
    assert b[k_idx] < 0 and b[d_idx] > 0, "符号错误"
    print("自检通过 ✅")
