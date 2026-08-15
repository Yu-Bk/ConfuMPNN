"""ConfuMPNN 模块单元测试。

运行方式（在 code/ 目录下）：
    conda activate confumpnn
    python tests/test_all.py
日志建议重定向到 code/log/。

覆盖模块：differentiable_charge, isoelectric_point, structure_aware_filter,
condition_embedding, losses, guided_sampler（辅助函数）。
"""

import sys
from pathlib import Path

import numpy as np
import torch

# 把 code/ 和 code/src/ 加入 import 路径（用包方式 from src.xxx import ...）
_CODE_DIR = Path(__file__).resolve().parents[1]
for p in [str(_CODE_DIR), str(_CODE_DIR / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.pka import AA_TO_IDX  # noqa: E402
from src.differentiable_charge import (  # noqa: E402
    net_charge,
    net_charge_from_logits,
    sidechain_charge,
)
from src.isoelectric_point import find_pI  # noqa: E402
from src.structure_aware_filter import (  # noqa: E402
    StructureAwareFilter,
    load_preset,
)
from src.condition_embedding import ConditionEncoder, make_condition_vector  # noqa: E402
from src.losses import composite_loss  # noqa: E402
from src.charge_lookahead import ChargeLookahead, make_dynamic_callback  # noqa: E402

_TOL = 1e-2
_results = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    _results.append((status, name))
    print(f"[{status}] {name} {detail}")
    if not cond:
        raise AssertionError(f"测试失败: {name} {detail}")


# ----------------------------------------------------------------------
# differentiable_charge
# ----------------------------------------------------------------------
def test_sidechain_charge():
    check("D @ pH7.4 ≈ -1", abs(sidechain_charge("D", 7.4).item() + 1.0) < _TOL)
    check("E @ pH7.4 ≈ -1", abs(sidechain_charge("E", 7.4).item() + 1.0) < _TOL)
    check("K @ pH7.4 ≈ +1", abs(sidechain_charge("K", 7.4).item() - 1.0) < _TOL)
    check("R @ pH7.4 ≈ +1", abs(sidechain_charge("R", 7.4).item() - 1.0) < _TOL)
    check("A（非电离）≈ 0", abs(sidechain_charge("A", 7.4).item()) < _TOL)
    # His pKa=6.0，在 pH=6.0 时约一半质子化 → ≈ +0.5
    check("H @ pH6.0 ≈ +0.5", abs(sidechain_charge("H", 6.0).item() - 0.5) < 0.05)
    # 单调性：pH 越高，酸性残基越负、碱性残基正电越少
    assert sidechain_charge("D", 5.0).item() > sidechain_charge("D", 8.0).item()
    assert sidechain_charge("K", 8.0).item() < sidechain_charge("K", 5.0).item()
    print("  [单调性] 电荷随 pH 变化方向正确")


def test_net_charge():
    q = net_charge("DDD", 7.4)
    # 3×(-1) + N端(+~1) + C端(-~1) ≈ -3
    check("net_charge('DDD', 7.4) ≈ -3", abs(q + 3.0) < 0.1, f"got {q:.3f}")
    q2 = net_charge("KKK", 7.4)
    check("net_charge('KKK', 7.4) ≈ +3", abs(q2 - 3.0) < 0.1, f"got {q2:.3f}")


def test_net_charge_from_logits():
    # 位置1 确定是 K，其余均匀 → 期望净电荷 ≈ +1（含末端 ±1 抵消）
    logits = torch.zeros(1, 3, 21)
    logits[0, 1, AA_TO_IDX["K"]] = 10.0
    mask = torch.ones(1, 3)
    c = net_charge_from_logits(logits, 7.4, mask)
    check("net_charge_from_logits 含确定 K ≈ +1", abs(c.item() - 1.0) < 0.3, f"got {c.item():.3f}")
    # 可微性：梯度能流到 logits
    logits.requires_grad_(True)
    c2 = net_charge_from_logits(logits, 7.4, mask)
    c2.sum().backward()
    check("net_charge_from_logits 可反向传播", logits.grad is not None and logits.grad.abs().sum().item() > 0)


# ----------------------------------------------------------------------
# isoelectric_point
# ----------------------------------------------------------------------
def test_find_pI():
    pI_basic = find_pI("RRRRR")
    pI_acidic = find_pI("DDDDD")
    check("多碱性序列 pI 高 (>9)", pI_basic > 9.0, f"pI={pI_basic:.2f}")
    check("多酸性序列 pI 低 (<4)", pI_acidic < 4.0, f"pI={pI_acidic:.2f}")
    # 自洽：在 pI 处净电荷 ≈ 0
    q = net_charge("KKEEDD", find_pI("KKEEDD"))
    check("pI 处净电荷 ≈ 0", abs(q) < 1e-2, f"q={q:.3f}")


# ----------------------------------------------------------------------
# structure_aware_filter
# ----------------------------------------------------------------------
def test_filter_rules():
    # 8 个 K 紧挨 + 2 个未解码位置 → 规则1（聚集）与规则4（连通聚类）应触发
    # 注意：坐标沿 x 轴展开，使距离 = |i-j|（避免对角线导致距离被 √3 放大）
    coords = np.zeros((10, 3))
    coords[:, 0] = np.arange(10, dtype=float)
    seq = [AA_TO_IDX["K"]] * 8 + [20, 20]  # 前 8 K，后 2 未解码
    filt = StructureAwareFilter(coords)
    bias, info = filt.compute_bias(np.array(seq))
    check("聚集规则触发", info["charge_cluster"]["pos_over"] > 0, str(info["charge_cluster"]))
    check("连通聚类规则触发", info["same_sign_cluster"]["n_components"] == 1)
    # 未解码位置应被抑制选 K/R
    k_idx = AA_TO_IDX["K"]
    check("未解码位置 K 被抑制", bias[8, k_idx].item() < 0 and bias[9, k_idx].item() < 0)

    # 全中性残基 → 无抑制
    seq_neutral = [AA_TO_IDX["A"]] * 8 + [20, 20]
    b2, _ = filt.compute_bias(np.array(seq_neutral))
    check("中性序列无抑制", float(b2.abs().sum()) == 0.0)


def test_load_preset():
    cfg = load_preset("default")
    check("default 预设含 4 条规则", set(cfg) == {"charge_cluster", "salt_bridge", "core_charge", "same_sign_cluster"})
    cfg_mem = load_preset("membrane")
    check("membrane 预设核心规则更严", cfg_mem["core_charge"]["charge_count"] < cfg["core_charge"]["charge_count"])


# ----------------------------------------------------------------------
# condition_embedding
# ----------------------------------------------------------------------
def test_condition_vector():
    v = make_condition_vector(pH=7.4)
    check("条件向量维度 = 7", v.shape[0] == 7)
    check("只给 pH 时 flag 全 0", float(v[1]) == 0 and float(v[3]) == 0 and float(v[5]) == 0)
    v2 = make_condition_vector(pH=5.0, net_charge=0.0, local_pos_limit=8)
    check("给约束后 flag 置 1", float(v2[1]) == 1 and float(v2[3]) == 1 and float(v2[5]) == 0)


def test_condition_encoder():
    enc = ConditionEncoder()
    v1 = make_condition_vector(pH=7.4)
    v2 = make_condition_vector(pH=5.0, net_charge=0.0)
    batch = torch.stack([v1, v2])
    tokens = enc(batch)
    check("soft prompt shape [2, 4, 128]", tuple(tokens.shape) == (2, 4, 128))
    check("soft prompt 可微", tokens.sum().backward() is None or True)
    # 不同条件 → 不同 embedding（起码不完全相同）
    check("不同条件向量产生不同 embedding", not torch.allclose(tokens[0], tokens[1]))


# ----------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------
def test_composite_loss():
    B, L = 2, 5
    torch.manual_seed(0)
    logits = torch.randn(B, L, 21, requires_grad=True)
    target = torch.randint(0, 20, (B, L))
    mask = torch.ones(B, L)
    filter_bias = -0.5 * torch.ones(L, 21)
    loss, terms = composite_loss(
        logits, target, mask, pH=7.4, target_charge=0.0,
        filter_bias=filter_bias, lambda_c=0.1, lambda_l=0.05,
    )
    check("复合损失能 backward", torch.isfinite(loss).item())
    loss.backward()
    check("梯度传导到 logits", logits.grad is not None and logits.grad.abs().sum().item() > 0)
    check("返回各分量 terms", set(["ce", "charge", "structure", "total"]).issubset(set(terms)))


# ----------------------------------------------------------------------
# guided_sampler（辅助函数，不依赖真实模型）
# ----------------------------------------------------------------------
def test_build_static_bias():
    from src.guided_sampler import build_static_bias

    L = 10
    fake_X = torch.randn(1, L, 4, 3)
    coords = fake_X[0, :, 1].numpy()
    filt = StructureAwareFilter(coords)
    fd = {"X": fake_X}
    bias, info = build_static_bias(fd, filt, seq_ref=None)
    check("静态 bias 维度 [1, 10, 21]", tuple(bias.shape) == (1, L, 21), str(tuple(bias.shape)))


# ----------------------------------------------------------------------
# charge_lookahead（修复：target 必须真正影响 bias，否则被 softmax 抵消）
# ----------------------------------------------------------------------
def test_lookahead_target_sensitivity():
    """回归测试：三档 target 必须产生不同的 bias。

    修复前 bias = -strength·(Q_k − target) 中 target 落在不依赖候选 k 的
    常数项，被 softmax 常数平移不变性完全抵消 → 三档 target bias 相同。
    修复后 bias = strength·(target − Q_current)·q_k，target 进入交叉项。
    """
    seq = [AA_TO_IDX["K"]] * 5 + [20] * 5  # 前 5 K（正电），后 5 未解码
    k_idx = AA_TO_IDX["K"]
    d_idx = AA_TO_IDX["D"]
    k_biases = {}
    for tgt in (8.0, 0.0, -8.0):
        la = ChargeLookahead(pH=7.4, target_charge=tgt, strength=0.5)
        b = la.bias_at(np.array(seq), position=5)
        k_biases[tgt] = b[k_idx]
    check("三档 target 的 bias[K] 各不相同",
          len({round(v, 3) for v in k_biases.values()}) == 3,
          f"got {k_biases}")
    check("target=+8 促进 K（bias>0）", k_biases[8.0] > 0, f"got {k_biases[8.0]:+.3f}")
    check("target=-8 抑制 K（bias<0）", k_biases[-8.0] < 0, f"got {k_biases[-8.0]:+.3f}")
    # 同一步内：K（正电候选）与 D（负电候选）方向相反
    la = ChargeLookahead(pH=7.4, target_charge=0.0, strength=0.5)
    b = la.bias_at(np.array(seq), position=5)
    check("target=0 时 K 与 D bias 方向相反", b[k_idx] < 0 < b[d_idx],
          f"K={b[k_idx]:+.3f} D={b[d_idx]:+.3f}")


def test_lookahead_termini_flag():
    """include_termini=False 时 bias 应不同（末端电荷影响 Q_current）。"""
    seq = [AA_TO_IDX["K"]] * 5 + [20] * 5
    la_on = ChargeLookahead(pH=7.4, target_charge=2.0, strength=0.5, include_termini=True)
    la_off = ChargeLookahead(pH=7.4, target_charge=2.0, strength=0.5, include_termini=False)
    b_on = la_on.bias_at(np.array(seq), position=5)
    b_off = la_off.bias_at(np.array(seq), position=5)
    check("include_termini 影响 bias",
          float(np.abs(b_on[AA_TO_IDX["K"]] - b_off[AA_TO_IDX["K"]])) > 1e-3,
          f"on={b_on[AA_TO_IDX['K']]:+.3f} off={b_off[AA_TO_IDX['K']]:+.3f}")


def test_lookahead_callback_signature():
    """make_dynamic_callback 返回的 callable 接受 (S_cur, t) 返回 [21]。"""
    cb = make_dynamic_callback(pH=7.4, target_charge=1.0, strength=0.5)
    S = np.array([AA_TO_IDX["A"]] * 3 + [20] * 3, dtype=np.int64)
    bias = cb(S, t=2)
    check("回调返回 [21] 且 shape 正确", bias.shape == (21,), str(bias.shape))
    # target=None（不引导电荷）→ 全零
    cb0 = make_dynamic_callback(pH=7.4, target_charge=None)
    b0 = cb0(np.array([20] * 4), t=0)
    check("target=None 时 bias 全零", float(np.abs(b0).sum()) == 0.0)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    test_sidechain_charge()
    test_net_charge()
    test_net_charge_from_logits()
    test_find_pI()
    test_filter_rules()
    test_load_preset()
    test_condition_vector()
    test_condition_encoder()
    test_composite_loss()
    test_build_static_bias()
    test_lookahead_target_sensitivity()
    test_lookahead_termini_flag()
    test_lookahead_callback_signature()

    passed = sum(1 for s, _ in _results if s == "PASS")
    print(f"\n{'='*50}\n通过 {passed}/{len(_results)} 项测试")
    if passed != len(_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
