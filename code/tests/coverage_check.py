#!/usr/bin/env python3
"""覆盖核查工具：判断验证蛋白是否在训练集"覆盖范围"内（2026-09-03 用户原则固化）。

原则：验证集只放"训练覆盖内"的蛋白（验证训练效果），分布外（深负电/超长/稀疏）
单独作泛化外推讨论，两者分开。本工具给统一、可复制的判定标准。

覆盖判定（三档）：
  - coverage = in       ：训练中 ≥100 个"相近域"（|ΔL| ≤ max(0.15·L, 40) 且 |Δq| ≤ 4）
  - coverage = boundary ：30–99 个相近域（覆盖边缘）
  - coverage = out      ：<30 个相近域（训练几乎无相似样本）
辅助标签（独立于 coverage，供报告）：
  - charge_tail = 深负电/高正电  ：native_q 超出训练 q 分布的 [2, 98]% 分位
  - L 分位 / q 分位一并输出。

用法：
  python coverage_check.py --labels <train_labels.npz> --prots "P1:L1:q1,P2:L2:q2" --out <out.json>
  或 --prots-json <{"P": {"L":..,"q":..}}>
native_q 用与训练相同的 net_charge(seq, 7.4) 计算（build_labels 同源）。
"""
import argparse, json
import numpy as np


def load_train(labels):
    d = np.load(labels, allow_pickle=True)
    N = len(d["domain_ids"])
    pH = np.asarray(d["pH"]).reshape(N, 8)
    ch = np.asarray(d["charge"]).reshape(N, 8)
    col = int(np.argmin(np.abs(pH[0, :] - 7.4)))
    q = ch[:, col].astype(float)
    L = np.array([len(s) for s in d["seqs"]])
    return q, L, N


def check(prots, q, L, N):
    """prots: {pdb: {"L":int,"q":float}} → 判定"""
    out = {}
    q2 = np.percentile(q, [2, 98])
    for pdb, info in sorted(prots.items()):
        Lp = info["L"]; qp = info["q"]
        Lpct = 100 * float(np.mean(L > Lp))
        qpct = 100 * float(np.mean(q <= qp))
        tol_L = max(0.15 * Lp, 40)
        n_close = int(sum((L >= max(20, Lp - tol_L)) & (L <= Lp + tol_L) &
                          (q >= qp - 4) & (q <= qp + 4)))
        if n_close >= 100:
            cov = "in"
        elif n_close >= 30:
            cov = "boundary"
        else:
            cov = "out"
        tail = ""
        if qp < q2[0]:
            tail = "深负电长尾"
        elif qp > q2[1]:
            tail = "高正电长尾"
        out[pdb] = {
            "L": int(Lp), "q": round(float(qp), 2),
            "L_pct_gt": round(Lpct, 1),   # 比它长的训练域占比
            "q_pctile": round(qpct, 1),   # native_q 在训练分布的分位
            "n_close": n_close, "coverage": cov, "charge_tail": tail,
            "train_L_2_98": [int(np.percentile(L, 2)), int(np.percentile(L, 98))],
            "train_q_2_98": [round(float(x), 1) for x in np.percentile(q, [2, 98])],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--prots", default=None, help="P:L:q,P:L:q,...")
    ap.add_argument("--prots-json", default=None, help="json 文件 {pdb:{L,q}}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    q, L, N = load_train(a.labels)
    if a.prots:
        prots = {}
        for tok in a.prots.split(","):
            p, Ls, qs = tok.split(":")
            prots[p] = {"L": int(Ls), "q": float(qs)}
    else:
        prots = json.load(open(a.prots_json))
    res = check(prots, q, L, N)
    print(f"训练集 {N} 域 | L[2,98]%={res and next(iter(res.values()))['train_L_2_98']} "
          f"| q[2,98]%={res and next(iter(res.values()))['train_q_2_98']}")
    print(f"{'pdb':<8}{'L':>5}{'native_q':>9}{'L>pct':>7}{'q_pct':>7}{'n_close':>8}  {'coverage':<10}{'tail'}")
    for pdb, r in res.items():
        print(f"{pdb:<8}{r['L']:>5}{r['q']:>9}{r['L_pct_gt']:>7}{r['q_pctile']:>7}{r['n_close']:>8}  "
              f"{r['coverage']:<10}{r['charge_tail']}")
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"→ 写 {a.out}")


if __name__ == "__main__":
    main()
