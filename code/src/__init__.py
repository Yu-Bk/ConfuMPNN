"""ConfuMPNN 核心模块库。

模块列表（按用途分类）：
- pka.py                     : 氨基酸 pKa 表与带电类型常量
- differentiable_charge.py   : 可微 pH 感知净电荷计算
- isoelectric_point.py       : pI（等电点）二分搜索
- structure_aware_filter.py  : 结构感知过滤器（logit bias 注入）
- condition_embedding.py     : pH 感知条件编码器（Soft Prompt）
- losses.py                  : 复合损失函数
- guided_sampler.py          : 引导采样器（包装 LigandMPNN decoder）

使用前请先激活环境：
    conda activate confumpnn
"""
