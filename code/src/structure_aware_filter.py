"""结构感知过滤器：实时检测极端电荷聚集，输出 logit bias。

对应 PROJECT_PLAN.md 4.1 Step 3 的"结构感知过滤器"（5 条检测规则的实时
logit bias 注入，不是事后过滤）。每条规则在检测到异常时，对相关位置的
候选氨基酸施加**负 bias**（抑制），从而在采样时就避免生成物理上不合理的序列。

实现 4 条核心规则（第 5 条"正负交替过滤"已被盐桥规则覆盖，见 PROJECT_PLAN 6.3）：

1. **空间电荷聚集**：某位置 10Å 邻域内已解码的同号强电荷残基 ≥ 阈值
   → 抑制在该位置继续放置同号带电残基（K/R 或 D/E）。
2. **盐桥过密**：某位置 10Å 邻域内正/负电荷对数量 ≥ 阈值
   → 抑制继续生成 K/R 或 D/E（避免 K-E-K-E 式的密集盐桥）。
3. **核心电荷渗入**：残基埋在核心（burial 高）且 8Å 内带电残基 ≥ 阈值
   → 抑制在该埋藏位置放置带电残基。
4. **同号电荷空间聚类**：8Å 连通图内同号电荷 ≥ 阈值（4+）
   → 抑制该连通区域继续放置同号带电残基。

坐标使用 Cα（残基位置近似）；距离阈值等参数可通过 YAML preset 配置。
注意：电荷规则只统计**已解码**的残基（seq_int 中非 X 的位置），
因此可配合引导采样器实现逐步实时检测。
"""

from pathlib import Path

import numpy as np
import torch

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .pka import AA_TO_IDX, STRONG_NEGATIVE, STRONG_POSITIVE

# 未解码标记（与 LigandMPNN 的 restype 一致：20 = X / 未知）
UNDECODED = 20

# AA 索引（在 [L, 21] bias 张量中的列位置，前 20 是标准氨基酸，最后是 X）
POS_AA_IDX = [AA_TO_IDX[a] for a in STRONG_POSITIVE]  # K, R
NEG_AA_IDX = [AA_TO_IDX[a] for a in STRONG_NEGATIVE]  # D, E
CHARGED_AA_IDX = POS_AA_IDX + NEG_AA_IDX


def default_config():
    """默认规则阈值（PROJECT_PLAN.md 4.1 中的设定值）。"""
    return {
        "charge_cluster": {
            "radius": 10.0, "threshold": 6, "strength": -1.0,
            "desc": "10Å 内同号电荷 ≥6 → 抑制继续放同号带电残基",
        },
        "salt_bridge": {
            "radius": 10.0, "threshold": 5, "strength": -1.0,
            "desc": "10Å 内正负电荷对 ≥5 → 抑制继续放带电残基",
        },
        "core_charge": {
            "burial_radius": 10.0, "charge_radius": 8.0,
            "burial_threshold": 0.8, "charge_count": 4, "strength": -1.0,
            "desc": "核心埋藏位置 8Å 内带电残基 ≥4 → 抑制放带电残基",
        },
        "same_sign_cluster": {
            "radius": 8.0, "threshold": 4, "strength": -1.0,
            "desc": "8Å 连通图同号电荷 ≥4 → 抑制该区域放同号带电残基",
        },
    }


class StructureAwareFilter:
    """结构感知过滤器。

    参数:
        coords: [L, 3] 残基 Cα 坐标（torch 或 numpy）
        mask: [L] 有效残基掩码（1=有效），默认全有效
        config: 规则阈值字典（None 用 default_config()）
    """

def load_preset(preset="default", path=None):
    """从 code/configs/filter_presets.yaml 加载预设规则。

    参数:
        preset: 预设名（default / nucleic_acid_binding / membrane / acidic）
        path: yaml 文件路径；None 时自动定位到 code/configs/filter_presets.yaml
    返回:
        dict：规则配置（可直接传给 StructureAwareFilter(config=...)）
    """
    import yaml

    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "filter_presets.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    presets = data["presets"]
    if preset not in presets:
        raise KeyError(f"未知预设 '{preset}'，可选：{list(presets)}")
    cfg = presets[preset]
    # 只保留规则字典（丢弃 desc/ph_hint 等元数据字段）
    return {k: v for k, v in cfg.items() if isinstance(v, dict)}


class StructureAwareFilter:
    """结构感知过滤器。

    参数:
        coords: [L, 3] 残基 Cα 坐标（torch 或 numpy）
        mask: [L] 有效残基掩码（1=有效），默认全有效
        config: 规则阈值字典（None 用 default_config()）
    """

    def __init__(self, coords, mask=None, config=None):
        coords = np.asarray(coords, dtype=np.float64)
        self.coords = coords
        self.L = coords.shape[0]
        self.mask = mask if mask is not None else np.ones(self.L, dtype=bool)
        self.config = config if config is not None else default_config()
        self._dist = self._compute_distance_matrix()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _compute_distance_matrix(self):
        """Cα 欧氏距离矩阵 [L, L]。"""
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    @staticmethod
    def _decode_charge_flags(seq_int):
        """从解码序列（[L] int，20=X）提取强正/强负标记（仅已解码位置）。"""
        pos = np.zeros(len(seq_int), dtype=bool)
        neg = np.zeros(len(seq_int), dtype=bool)
        for a in STRONG_POSITIVE:
            pos |= (seq_int == AA_TO_IDX[a])
        for a in STRONG_NEGATIVE:
            neg |= (seq_int == AA_TO_IDX[a])
        return pos, neg

    def _masked(self, mat):
        """只保留有效残基（mask）参与统计，无效位置置 False。"""
        return mat & self.mask[None, :]

    @staticmethod
    def _apply_bias(bias, mask, col_idxs, strength):
        """给 mask 为 True 的行、col_idxs 这些列加 strength。

        mask 可能是 numpy bool 数组；col_idxs 是列索引列表（如 [K, R]）。
        逐列写入，避免 bool mask 与多列 advanced indexing 的广播冲突
        （全 False 时多列索引会报 shape mismatch）。
        """
        m = (
            torch.from_numpy(mask)
            if not torch.is_tensor(mask)
            else mask.to(torch.bool)
        )
        for k in col_idxs:
            bias[m, k] += strength

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def compute_bias(self, seq_int, pH=None):
        """根据当前解码序列计算 bias 增量。

        参数:
            seq_int: [L] int 序列（20=X 表示尚未解码的位置）
            pH: 预留参数（目前强电荷判断不依赖 pH；若以后需要弱电荷可按 pH 计算）

        返回:
            bias: [L, 21] float 张量，负值抑制（将加到 logits 上）
            info: 每条规则触发的统计信息（便于记录/调试）
        """
        seq_int = np.asarray(seq_int)
        pos, neg = self._decode_charge_flags(seq_int)
        undecoded = (seq_int == UNDECODED) & self.mask  # 只有这些位置还能被抑制
        charged = pos | neg

        bias = torch.zeros(self.L, 21)
        info = {}

        # ---- 规则 1：空间电荷聚集（10Å 内同号 ≥ 阈值） ----
        cfg = self.config["charge_cluster"]
        nb = self._masked(self._dist <= cfg["radius"])
        # 邻域内（含自身）已解码的强正/强负数量
        pos_count = (nb & pos[None, :]).sum(axis=1)
        neg_count = (nb & neg[None, :]).sum(axis=1)
        over_pos = undecoded & (pos_count >= cfg["threshold"])
        over_neg = undecoded & (neg_count >= cfg["threshold"])
        self._apply_bias(bias, over_pos, POS_AA_IDX, cfg["strength"])
        self._apply_bias(bias, over_neg, NEG_AA_IDX, cfg["strength"])
        info["charge_cluster"] = {
            "pos_over": int(over_pos.sum()), "neg_over": int(over_neg.sum()),
        }

        # ---- 规则 2：盐桥过密（10Å 内正负电荷对 ≥ 阈值） ----
        cfg = self.config["salt_bridge"]
        # 以 min(正电荷数, 负电荷数) 近似已形成/潜在的盐桥对数
        pairs = np.minimum(pos_count, neg_count)
        over_bridge = undecoded & (pairs >= cfg["threshold"])
        self._apply_bias(bias, over_bridge, CHARGED_AA_IDX, cfg["strength"])
        info["salt_bridge"] = {"over": int(over_bridge.sum())}

        # ---- 规则 3：核心电荷渗入（burial 高 + 8Å 内带电 ≥ 阈值） ----
        cfg = self.config["core_charge"]
        nb_burial = self._masked(self._dist <= cfg["burial_radius"])
        burial_count = nb_burial.sum(axis=1)  # 10Å 内 Cα 数 → burial 近似
        max_burial = burial_count.max()
        burial_ratio = burial_count / max_burial if max_burial > 0 else burial_count
        nb_charge = self._masked(self._dist <= cfg["charge_radius"])
        charge_count = (nb_charge & charged[None, :]).sum(axis=1)
        core = undecoded & (burial_ratio > cfg["burial_threshold"]) & (
            charge_count >= cfg["charge_count"]
        )
        self._apply_bias(bias, core, CHARGED_AA_IDX, cfg["strength"])
        info["core_charge"] = {"core": int(core.sum())}

        # ---- 规则 4：同号电荷空间聚类（8Å 连通图 ≥ 阈值） ----
        cfg = self.config["same_sign_cluster"]
        adj = csr_matrix(
            (self._dist <= cfg["radius"]).astype(int)
        )  # 连通性仅基于结构，不需 mask
        n_comp, labels = connected_components(adj, directed=False)
        for c in range(n_comp):
            members = labels == c
            n_pos = int((members & pos).sum())
            n_neg = int((members & neg).sum())
            tgt = members & undecoded
            if n_pos >= cfg["threshold"]:
                self._apply_bias(bias, tgt, POS_AA_IDX, cfg["strength"])
            if n_neg >= cfg["threshold"]:
                self._apply_bias(bias, tgt, NEG_AA_IDX, cfg["strength"])
        info["same_sign_cluster"] = {"n_components": int(n_comp)}

        return bias, info


if __name__ == "__main__":
    # 自检：构造 8 个残基的一维坐标（沿 x 轴），全部放 K，应触发规则 1/4
    coords = np.zeros((8, 3))
    coords[:, 0] = np.arange(8, dtype=float)
    seq = np.array([AA_TO_IDX["K"]] * 8)
    filt = StructureAwareFilter(coords)
    b, info = filt.compute_bias(seq)
    print("rules:", info)
    print("bias shape:", b.shape, "| nonzero rows:", int((b != 0).any(dim=1).sum()))
