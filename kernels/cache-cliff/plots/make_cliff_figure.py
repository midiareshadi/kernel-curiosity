"""
make_cliff_figure.py — plot the read-only vs read-write cache cliff.

Reads results/L4-dev/cache-cliff/readonly_vs_readwrite_fp16.csv and draws both
curves on one axis: input size (MB, log) vs % of measured peak bandwidth. The
read-write cliff sits at ~27 MB; the read-only cliff at ~55 MB. The ~2x shift is
the evidence that the cliff is caused by input and output sharing the cache.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "..", "results", "L4-dev",
                   "cache-cliff", "readonly_vs_readwrite_fp16.csv")
OUT_DIR = os.path.join(HERE, "..", "..", "..", "docs", "figures")

BROWN = "#8c5a3c"
GREEN = "#3b6d4f"
MUTED = "#666666"
CACHE_MB = 48.0


def load():
    mb, rw, ro = [], [], []
    with open(os.path.abspath(CSV)) as f:
        for row in csv.DictReader(f):
            mb.append(float(row["in_mb"]))
            rw.append(float(row["rw_pct"]))
            ro.append(float(row["ro_pct"]))
    return mb, rw, ro


def main():
    mb, rw, ro = load()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.axhline(100, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.axvline(CACHE_MB, color=MUTED, lw=0.8, ls=":", zorder=1)
    # label in the empty lower-right region, below the flat tails
    ax.annotate("48 MB cache", xy=(CACHE_MB, 100), xytext=(120, 60),
                color=MUTED, fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))

    ax.plot(mb, ro, "-s", color=GREEN, ms=4, lw=1.4,
            label="read-only (input uses whole cache)")
    ax.plot(mb, rw, "-o", color=BROWN, ms=4, lw=1.4,
            label="read-write (input + output share cache)")

    ax.set_xscale("log")
    ax.set_xlabel("input size (MB, log scale)", fontsize=9)
    ax.set_ylabel("% of measured peak bandwidth", fontsize=9)
    ax.set_title("Removing the output doubles the cliff size",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    os.makedirs(os.path.abspath(OUT_DIR), exist_ok=True)
    out = os.path.join(os.path.abspath(OUT_DIR), "cache_cliff_rw_vs_ro.svg")
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
