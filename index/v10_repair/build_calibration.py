"""从响应曲线诊断 JSON 构建电荷校准表（v12 §7.1 数据源）。

校准原理：诊断已拟合「生成电荷 ≈ slope·target + intercept」。
要让实际生成电荷 = desired，需喂给编码器的条件向量电荷维度
target_eff = (desired − intercept) / slope。

两种粒度：
  - per_protein：诊断过的蛋白用各自 (slope, intercept)（最准）；
  - global：未诊断蛋白回退——把全部蛋白的 (target, mean_charge) 点
    合并后重新拟合一条全局直线（等价"跨蛋白平均响应"）。

用法（在项目根运行，confumpnn 环境）：
  python index/v10_repair/build_calibration.py \
      --diag output/v10_diag_response.json --label v10 \
      --out output/charge_calibration_v10.json

输出 JSON 结构：
  {
    "label": "v10",
    "source_diag": "...",
    "note": "...",
    "global": {"slope": <float>, "intercept": <float>},
    "per_protein": {"7pujA01": {"slope": ..., "intercept": ...}, ...}
  }
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="从诊断 JSON 构建电荷校准表")
    ap.add_argument("--diag", required=True, help="诊断 JSON（v10_diag_response_curve.py 输出）")
    ap.add_argument("--label", default="v10", help="校准表标签（对应编码器版本）")
    ap.add_argument("--out", required=True, help="校准表 JSON 输出路径")
    args = ap.parse_args()

    d = json.load(open(args.diag))
    prot = d["proteins"]

    # per_protein：直接用诊断拟合值
    per = {}
    for name, p in prot.items():
        per[name] = {"slope": p["slope"], "intercept": p["intercept"]}

    # global：合并所有 (target, mean_charge) 点重新线性拟合
    ts, ms = [], []
    for name, p in prot.items():
        ts += list(p["targets"])
        ms += list(p["mean_charge"])
    ts, ms = np.array(ts, dtype=float), np.array(ms, dtype=float)
    A = np.vstack([ts, np.ones_like(ts)]).T
    slope_g, int_g = np.linalg.lstsq(A, ms, rcond=None)[0]

    out = {
        "label": args.label,
        "source_diag": args.diag,
        "note": "生成电荷 ≈ slope*target + intercept；校准 target_eff=(desired−intercept)/slope",
        "global": {"slope": float(slope_g), "intercept": float(int_g)},
        "per_protein": per,
    }
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"已写 {args.out}")
    print(f"global slope={slope_g:.3f}  intercept={int_g:.3f}  (n_point={len(ts)})")
    print(f"per_protein: {len(per)} 个蛋白")


if __name__ == "__main__":
    main()
