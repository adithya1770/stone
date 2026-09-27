
import csv
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures")
DATASET = "ImageNette"
TIMELINE_RUN = "ImageNette|eightsignal_warm|1"

# Palette (colorblind-checked categorical order) and chart chrome
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, YELLOW, VIOLET, GRAY = "#2a78d6", "#eb6834", "#eda100", "#4a3aa7", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

LABEL = {
    "baseline": "Baseline (FP32)", "always_int8": "Always INT8", "egreedy": "ε-greedy",
    "linucb_cold": "LinUCB (cold)", "linucb_warm": "LinUCB (warm)",
    "eightsignal_cold": "EightSignal (cold)", "eightsignal_warm": "EightSignal (warm)",
}
COLOR = {"baseline": GRAY, "always_int8": MUTED, "egreedy": YELLOW, "linucb_cold": VIOLET,
         "linucb_warm": VIOLET, "eightsignal_cold": BLUE, "eightsignal_warm": BLUE}
MARKER = {"baseline": "s", "always_int8": "s", "egreedy": "o", "linucb_cold": "o",
          "linucb_warm": "^", "eightsignal_cold": "o", "eightsignal_warm": "^"}


def read(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = [r for r in read("trial_summary.csv") if r["dataset"] == DATASET]
    stats = {}
    for cfg in LABEL:
        rs = [r for r in runs if r["config"] == cfg]
        stats[cfg] = {}
        for key in ("stressed_latency_ms", "stressed_accuracy_pct", "stressed_int8_pct"):
            v = [float(r[key]) for r in rs]
            stats[cfg][key] = (st.mean(v), st.stdev(v))

    # 1. Latency vs accuracy trade-off
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    for cfg, s in stats.items():
        (lat, lat_sd), (acc, acc_sd) = s["stressed_latency_ms"], s["stressed_accuracy_pct"]
        ax.errorbar(lat, acc, xerr=lat_sd, yerr=acc_sd, fmt=MARKER[cfg], color=COLOR[cfg],
                    ecolor=COLOR[cfg], elinewidth=1.1, capsize=2.5, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
    spec = {  # label offsets (points), alignment, leader line
        "baseline": (8, 2, "left", False), "always_int8": (8, -2, "left", False),
        "egreedy": (-10, 10, "right", False), "linucb_cold": (-8, 22, "right", True),
        "linucb_warm": (14, -22, "left", True), "eightsignal_cold": (-70, -14, "right", True),
        "eightsignal_warm": (-70, 14, "right", True),
    }
    for cfg, (dx, dy, ha, arrow) in spec.items():
        kw = dict(fontsize=9, color=INK_2, ha=ha, va="center")
        if arrow:
            kw["arrowprops"] = dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=4)
        ax.annotate(LABEL[cfg], (stats[cfg]["stressed_latency_ms"][0], stats[cfg]["stressed_accuracy_pct"][0]),
                    xytext=(dx, dy), textcoords="offset points", **kw)
    ax.set_xlabel("Latency under stress (ms)  —  lower is better")
    ax.set_ylabel("Top-1 accuracy under stress (%)  —  higher is better")
    ax.set_title(f"Latency vs. accuracy — {DATASET}, mean ± SD over 3 trials",
                 fontsize=12, pad=14, loc="left")
    style(ax)
    ax.set_xlim(85, 145)
    ax.set_ylim(65, 78)
    ax.legend(handles=[Patch(facecolor=GRAY, label="Static baselines"), Patch(facecolor=VIOLET, label="LinUCB"),
                       Patch(facecolor=YELLOW, label="ε-greedy"), Patch(facecolor=BLUE, label="EightSignal (custom)")],
              loc="lower left", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_latency_vs_accuracy.png"))
    plt.close(fig)

    # 2. INT8 usage under stress
    order = ["always_int8", "eightsignal_warm", "eightsignal_cold", "linucb_cold", "linucb_warm", "egreedy", "baseline"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    vals = [stats[c]["stressed_int8_pct"][0] for c in order]
    errs = [stats[c]["stressed_int8_pct"][1] for c in order]
    ax.barh(range(len(order)), vals, xerr=errs, height=0.58, color=BLUE, ecolor=MUTED,
            error_kw=dict(elinewidth=1.1, capsize=3), zorder=3)
    for y, v in enumerate(vals):
        ax.text(v + 3.2, y, f"{v:.0f}%", va="center", fontsize=9.5, color=INK_2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[c] for c in order], fontsize=9.5, color=INK)
    ax.set_xlabel("Share of stressed frames run at INT8 (%)")
    ax.set_xlim(0, 115)
    ax.set_title(f"How often each policy picks the fast INT8 model — {DATASET}", fontsize=12, pad=14, loc="left")
    style(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_int8_usage.png"))
    plt.close(fig)

    # 3. Decision timeline for one run
    steps = sorted((r for r in read("steps.csv") if r["run_id"] == TIMELINE_RUN), key=lambda r: int(r["step"]))
    x = [int(r["step"]) for r in steps]
    temp = [float(r["temperature_c"]) for r in steps]
    model = [r["model"] for r in steps]
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=200)
    start = 0
    for i in range(1, len(model) + 1):
        if i == len(model) or model[i] != model[start]:
            ax.axvspan(x[start] - 0.5, x[i - 1] + 0.5, color=ORANGE if model[start] == "FP32" else BLUE,
                       alpha=0.15, lw=0, zorder=1)
            start = i
    ax.plot(x, temp, color=INK, linewidth=1.6, zorder=3)
    ax.set_xlabel("Inference step")
    ax.set_ylabel("CPU temperature (°C)")
    ax.set_title("EightSignal (warm), ImageNette, trial 1 — model choice vs. temperature",
                 fontsize=12, pad=14, loc="left")
    style(ax)
    ax.set_xlim(x[0], x[-1])
    ax.legend(handles=[Patch(facecolor=BLUE, alpha=0.4, label="Running INT8 (fast)"),
                       Patch(facecolor=ORANGE, alpha=0.4, label="Running FP32 (accurate)")],
              loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_decision_timeline.png"))
    plt.close(fig)
    print(f"Figures written to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
