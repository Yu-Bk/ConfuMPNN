"""氨基酸 pKa 表与带电类型常量。

pKa 值来自 PROJECT_PLAN.md 4.3 节（游离 pKa，用于"结合层面"的电荷计算）。
注意：
- 这里是**游离氨基酸**的 pKa 近似，不含微环境修正。
- 验证阶段请用 PypKa / PROPKA 做微环境 pKa 修正。
- His 在此按"碱性"处理：pKa=6.0，pH>6 时咪唑去质子化，正电消失；
  它既可能在 pH≈6 附近带正电，也可能中性，具体由工作 pH 决定。
- Cys/Tyr 理论上也能去质子化带负电，但阈值 pH 较高（8.3 / 10.1）。
"""

# 侧链 pKa（只列出在生理 pH 范围内可能带电/改变带电态的残基）
PKA_SIDECHAIN = {
    "D": 3.9,   # Asp 侧链 -COOH
    "E": 4.3,   # Glu 侧链 -COOH
    "H": 6.0,   # His 咪唑基
    "C": 8.3,   # Cys 侧链 -SH
    "Y": 10.1,  # Tyr 酚羟基
    "K": 10.5,  # Lys 侧链 -NH3+
    "R": 12.5,  # Arg 胍基
}

# 主链末端 pKa
PKA_N_TERM = 9.7   # α-NH3+（氨基端）
PKA_C_TERM = 2.3   # α-COOH（羧基端）

# 标准 20 种氨基酸单字母（顺序与 LigandMPNN restype 一致：ACDEFGHIKLMNPQRSTVWY）
AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AAS)}

# 侧链带电类型（用于电荷/过滤器逻辑）
ACIDIC = ("D", "E", "C", "Y")   # 去质子化带负电（酸性）
BASIC = ("K", "R", "H")          # 质子化带正电（碱性）

# 空间过滤器用到的"强带电"残基（在生理 pH 下基本完全带电）
STRONG_POSITIVE = ("K", "R")     # 生理 pH 下几乎总是 +1
STRONG_NEGATIVE = ("D", "E")     # 生理 pH 下几乎总是 -1
