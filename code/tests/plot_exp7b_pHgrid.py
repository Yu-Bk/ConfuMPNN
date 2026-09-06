"""exp7b 绘图：每蛋白 pH-包络曲线（H2 达标臂数 + 平均 dev_of_mean vs pH，T1/T2/T3 三条线）。

数据：output/exp_pH2_{mode}/_summary.json（先跑 analyze_exp7b_pHgrid.py）。
输出：figure/exp7b_{mode}_{pdb}_pHgrid.png（白底 PNG，~9x4.2in）。
三处理配色（dataviz default categorical slots 1-3）：
  T1 蓝 #2a78d6 / T2 橙 #eb6834 / T3 水绿 #1baf7a
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
GRID = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4, 8.0, 8.5, 9.0]
TREATS = ["T1", "T2", "T3"]
STYLE = {
    "T1": {"color": "#2a78d6", "ls": "-", "marker": "o", "label": "T1 raw (no cal)"},
    "T2": {"color": "#eb6834", "ls": "-", "marker": "s", "label": "T2 7.4-fixed cal extrap"},
    "T3": {"color": "#1baf7a", "ls": "-", "marker": "^", "label": "T3 per-pH onsite cal"},
}
INK = "#0b0b0b"
MUTED = "#898781"
GRIDC = "#e1e0d9"
ARM5 = ["native", "n2", "p2", "n8", "p8"]


def load(mode):
    return json.load(open(ROOT / f"output/exp_pH2_{mode}/_summary.json"))


def per_pH_H2_and_dev(ps, treat):
    """返回 {ph_str: (h2_count, mean_dev)}，mean_dev=5臂 dev_of_mean 均值。"""
    to = ps["treatments"][treat]
    out = {}
    for ph in GRID:
        phs = str(ph)
        if phs not in to["by_pH"]:
            continue
        h2 = sum(1 for a in ARM5
                 if to["by_pH"][phs].get(a) and to["by_pH"][phs][a]["H2"])
        devs = [to["by_pH"][phs][a]["dev_of_mean"] for a in ARM5
                if to["by_pH"][phs].get(a)]
        out[phs] = (h2, sum(devs) / len(devs) if devs else float("nan"))
    return out


def plot_protein(mode, pdb, ps, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    fig.suptitle(f"{mode} · {pdb} (L={ps['L']})  pH grid x 3 calibration treatments",
                 color=INK, fontsize=12, fontweight="bold")
    for ax, ylab, panel, panel_id in [
        (axes[0], "H2 passed arms (0-5)", "a", "a"),
        (axes[1], "mean |dev_of_mean| (5 arms)", "b", "b"),
    ]:
        ax.set_facecolor("#ffffff")
        ax.grid(True, color=GRIDC, lw=0.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(f"({panel_id}) {ylab}", color=INK, fontsize=9)
        ax.set_xlabel("pH", color=INK, fontsize=9)
    x = GRID
    for treat in TREATS:
        info = per_pH_H2_and_dev(ps, treat)
        xs = [ph for ph in x if str(ph) in info]
        h2 = [info[str(ph)][0] for ph in xs]
        dev = [info[str(ph)][1] for ph in xs]
        st = STYLE[treat]
        axes[0].plot(xs, h2, color=st["color"], ls=st["ls"], marker=st["marker"],
                     lw=2, ms=5, label=st["label"], zorder=3)
        axes[1].plot(xs, dev, color=st["color"], ls=st["ls"], marker=st["marker"],
                     lw=2, ms=5, label=st["label"], zorder=3)
    axes[0].axhline(5, color=MUTED, lw=0.8, ls="--", zorder=1)
    axes[0].set_ylim(-0.2, 5.4)
    axes[0].set_yticks(range(0, 6))
    axes[1].axhline(2.0, color=MUTED, lw=0.8, ls="--", zorder=1)
    axes[1].set_ylim(0, None)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)
    axes[1].legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = figdir / f"exp7b_{mode}_{pdb}_pHgrid.png"
    fig.savefig(out, dpi=160, facecolor="#ffffff")
    plt.close(fig)
    print("wrote", out)


def main():
    for mode in sys.argv[1:] or ["prot", "lig"]:
        summary = load(mode)
        figdir = ROOT / "figure"
        figdir.mkdir(exist_ok=True)
        for pdb, ps in summary["proteins"].items():
            if ps["treatments"]:
                plot_protein(mode, pdb, ps, figdir)


if __name__ == "__main__":
    main()
