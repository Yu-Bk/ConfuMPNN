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

    然后把"偏离目标电荷"翻译成 bias。**关键设计**：bias 必须包含依赖
    候选 k 的交叉项，否则会被 softmax 常数平移不变性（softmax(x+c)=softmax(x)）
    抵消。因此不用朴素的 (Q_k − target)，而用"当前净电荷缺多少目标"去
    驱动"候选侧链电荷"：

        Q_current = Q_fixed + Q_expect_others + Q_termini   # 不含候选位
        bias_k   = strength · (target_charge − Q_current) · q(aa_k, pH)

    - 当 Q_current < target（还欠正电荷）→ (target−Q_current) > 0，
      正电候选（q_k>0）得正 bias 被促进、负电候选（q_k<0）被抑制；
    - 当 Q_current > target（正电荷过多）→ 反向推动。
    这样 target 通过 `target·q_k` 交叉项真正进入分布，且随解码推进
    Q_current 越准确、引导越精确（渐近收敛到 target）。

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

        # 4) 当前净电荷（不含候选位置）
        Q_current = fixed + expect_others + termini

        # 5) 对 20 个候选计算 bias：
        #    bias_k = strength · (target − Q_current) · q_k
        #    (target−Q_current) 是标量，q_k 依赖候选 → target 进入交叉项，不被 softmax 抵消
        biases = np.zeros(21, dtype=np.float32)
        biases[:20] = self.strength * (self.target_charge - Q_current) * self._q_vec
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
    # 自检：全 K 序列，分别验证 target 8/0/-8 三档 bias 是否真正不同
    # （本次修复的核心：target 必须进入与候选 k 相关的交叉项，否则被 softmax 抵消）
    import numpy as np
    from .pka import AA_TO_IDX

    seq = [AA_TO_IDX["K"]] * 5 + [UNDECODED] * 5  # 前 5 K，后 5 未解码
    k_idx = AA_TO_IDX["K"]
    d_idx = AA_TO_IDX["D"]
    results = {}
    for tgt in (8.0, 0.0, -8.0):
        la = ChargeLookahead(pH=7.4, target_charge=tgt, strength=0.5)
        b = la.bias_at(np.array(seq), position=5)
        results[tgt] = (b[k_idx], b[d_idx])
        print(f"target={tgt:+5.1f}  bias[K]={b[k_idx]:+.3f}  bias[D]={b[d_idx]:+.3f}")

    # 断言 1：三档 target 的 K bias 必须各不相同（修复前它们完全相同 → 本断言触发）
    k_biases = {t: v[0] for t, v in results.items()}
    assert len(set(round(v, 3) for v in k_biases.values())) == 3, \
        "target 未真正影响 bias（softmax 抵消 bug 未修复）"
    # 断言 2：target 偏正 → 促进正电候选（K bias 为正），target 偏负 → 抑制
    assert results[8.0][0] > 0 > results[-8.0][0], "target 与 K 的 bias 符号关系错误"
    # 断言 3：符号相反（全 K 序列当前偏正，D 的方向应与 K 相反）
    assert results[0.0][0] < 0 < results[0.0][1], "target=0 时符号错误"
    print("自检通过 ✅（target 8/0/-8 产生不同 bias，修复生效）")
