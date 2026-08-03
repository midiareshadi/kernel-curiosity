"""
make_softmax_figures.py — regenerate the softmax investigation figures from CSVs.

Reads results/<hw>/l2_isolation_<dtype>.csv (written by benchmarks/l2_isolation.py)
and writes SVG figures into docs/figures/.

Figures:
  1. spec_vs_measured.svg — datasheet bandwidth vs what a copy actually achieves
  2. cache_mirage.svg     — % of achievable bandwidth vs tensor size, per GPU
  3. two_gpus.svg         — the same kernel on L4 vs MI300X

Usage: python3 plots/make_softmax_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIGDIR = os.path.join(ROOT, "docs", "figures")

# colours chosen to stay legible in both light and dark themes
INK = "#2b2b2b"
MUTED = "#6b6b6b"
NAIVE = "#D97757"
TORCH = "#2e7d52"
CACHE = "#f0e6d2"

# per-GPU facts, recorded at measurement time
GPUS = {
    "L4-dev": dict(label="NVIDIA L4", spec=300.0, measured=230.0, cache_mb=48.0),
    "MI300X": dict(label="AMD MI300X", spec=5300.0, measured=3880.0, cache_mb=256.0),
}


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e8e2d8", linewidth=0.8)
    ax.set_axisbelow(True)


def load(hw, dtype="fp16"):
    path = os.path.join(ROOT, "results", hw, f"l2_isolation_{dtype}.csv")
    if not os.path.exists(path):
        print(f"  (skip) no CSV yet: {path}")
        return None
    return pd.read_csv(path)


def _thousands(ax, divisor=1000.0, suffix=r"$\times 10^3$"):
    """Relabel a large y-axis as small numbers plus a multiplier note."""
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t/divisor:g}" for t in ticks])
    ax.set_ylabel(f"GB/s  ({suffix})", fontsize=9, color=MUTED)


def fig_spec_vs_measured():
    """Figure 1: the promised bandwidth vs what a plain copy actually reaches."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, (hw, g) in zip(axes, GPUS.items()):
        vals = [g["spec"], g["measured"]]
        bars = ax.bar(["promised\n(spec sheet)", "measured\n(copy test)"], vals,
                      color=["#cfc7b8", NAIVE], width=0.55)
        pct = 100.0 * g["measured"] / g["spec"]
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(f"{g['label']}\nyou get {pct:.0f}% of the promise",
                     fontsize=10, color=INK)
        ax.set_ylim(0, g["spec"] * 1.25)
        style(ax)
        # only use the x10^3 form when the numbers are big enough to need it
        if g["spec"] >= 2000:
            _thousands(ax)
        else:
            ax.set_ylabel("GB/s", fontsize=9, color=MUTED)
    fig.suptitle("What you are promised vs what you get", fontsize=12, color=INK)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "spec_vs_measured.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_cache_mirage(dtype="fp16"):
    """Figure 2: % of achievable bandwidth vs tensor size, per GPU."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    have = {hw: load(hw, dtype) for hw in GPUS}
    have = {k: v for k, v in have.items() if v is not None}
    if not have:
        print("  (skip) cache_mirage: no CSVs yet")
        return

    fig, axes = plt.subplots(1, len(have), figsize=(4.3 * len(have), 3.8),
                             squeeze=False)
    for ax, (hw, df) in zip(axes[0], have.items()):
        g = GPUS[hw]
        lo = df["mb"].min() * 0.6
        ax.axvspan(lo, g["cache_mb"], color=CACHE, zorder=0)
        ax.axhline(100, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
        ax.plot(df["mb"], df["naive_pct"], "o-", color=NAIVE,
                markersize=5, zorder=3)
        ax.plot(df["mb"], df["torch_pct"], "s-", color=TORCH,
                markersize=5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("tensor size (MB, log scale)", fontsize=9, color=MUTED)
        ax.set_ylabel("% of measured peak bandwidth", fontsize=9, color=MUTED)
        ax.set_title(f"{g['label']}  (cache ~{g['cache_mb']:.0f} MB)",
                     fontsize=10, color=INK)
        style(ax)

    # one shared legend, outside the plotting area -> no overlap with data
    handles = [
        Line2D([], [], color=NAIVE, marker="o", markersize=5, label="simple Triton"),
        Line2D([], [], color=TORCH, marker="s", markersize=5, label="torch.softmax"),
        Line2D([], [], color=MUTED, linestyle="--", label="100% = measured peak bandwidth"),
        Patch(facecolor=CACHE, label="data fits in cache — measuring cache, not memory"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Small data hides in fast cache", fontsize=12.5, color=INK, y=1.05)
    fig.text(0.5, 0.98, "In the shaded area the numbers are too good to be true — some even pass 100%",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "cache_mirage.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_two_gpus(dtype="fp16"):
    """Figure 3: the same simple kernel on two GPUs."""
    from matplotlib.lines import Line2D

    have = {hw: load(hw, dtype) for hw in GPUS}
    have = {k: v for k, v in have.items() if v is not None}
    if len(have) < 2:
        print("  (skip) two_gpus: need both CSVs")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    styles = {
        "L4-dev": dict(color="#8a5a3c", marker="o", label="NVIDIA L4"),
        "MI300X": dict(color=TORCH, marker="s", label="AMD MI300X"),
    }
    handles = []
    for hw, df in have.items():
        st = styles[hw]
        g = GPUS[hw]
        ax.plot(df["mb"], df["naive_pct"], linestyle="-", color=st["color"],
                marker=st["marker"], markersize=6, zorder=3)
        ax.axvline(g["cache_mb"], color=st["color"], linewidth=1.8,
                   linestyle=":", alpha=0.9, zorder=1)
        handles.append(Line2D([], [], color=st["color"], marker=st["marker"],
                              markersize=6,
                              label=f"{st['label']}  —  limit {g['measured']:,.0f} GB/s"))
        handles.append(Line2D([], [], color=st["color"], linestyle=":",
                              label=f"{st['label']} cache ends ({g['cache_mb']:.0f} MB)"))

    ax.axhline(100, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
    handles.append(Line2D([], [], color=MUTED, linestyle="--",
                          label="100% = measured peak bandwidth"))

    ax.set_xscale("log")
    ax.set_ylim(0, 115)
    ax.set_xlabel("tensor size (MB, log scale)", fontsize=9, color=MUTED)
    ax.set_ylabel("% of measured peak bandwidth", fontsize=9, color=MUTED)
    ax.set_title("The same simple kernel, two GPUs", fontsize=12, color=INK)
    style(ax)
    ax.legend(handles=handles, loc="lower right", frameon=True,
              framealpha=0.95, edgecolor="#e0d8c8", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "two_gpus.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("generating figures:")
    fig_spec_vs_measured()
    fig_cache_mirage()
    fig_two_gpus()
    print("done.")


if __name__ == "__main__":
    main()
