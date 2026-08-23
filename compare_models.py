"""Figures comparing planners against the shot you asked for.

    python compare_models.py --compare trainsize
    python compare_models.py --compare method --n 1000
    python compare_models.py --compare mdn_200000_i8p256 policy_200000_i8p256 oracle
    python compare_models.py --compare trainsize --goal 0.9 0.3 6 250 0

By default the problems come from runs/data/evalset_3000.npz -- the same
held-out set train_ai.py scores on -- so a number here is directly
comparable to a row of results.csv. Pass --goal to ask every method for one
specific shot instead, across many different incoming balls: that answers
"how close does each one get to *this* request", which the sweep cannot.

Writes to runs/figures/compare_*.png plus a CSV of the same numbers.
"""

import argparse
import os

import numpy as np

from pingpong import compare, constants as C, dataset

FIG_DIR = os.path.join("runs", "figures")
EVAL_DEFAULT = os.path.join("runs", "data", "evalset_3000.npz")

# Light-surface chrome from the validated palette. Charts are printed and
# read on white, so they are built for the light surface only.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
TARGET = compare.TARGET_LIGHT

GOAL_UNITS = ["m", "m", "m/s", "rad/s", "rad/s"]
GOAL_TITLES = ["landing, long/short", "landing, across", "arrival speed",
               "topspin", "sidespin"]


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": BASELINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "figure.dpi": 130,
    })


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def load_problems(args):
    if args.evalset and os.path.exists(args.evalset):
        z = np.load(args.evalset)
        pos, vel, spin, tg = z["pos"], z["vel"], z["spin"], z["targets"]
    else:
        raise SystemExit(f"no problem set at {args.evalset}; "
                         "run `python train_ai.py data` first")
    n = min(args.n, len(pos))
    idx = np.random.default_rng(args.seed).choice(len(pos), n, replace=False)
    pos, vel, spin, tg = pos[idx], vel[idx], spin[idx], tg[idx]
    if args.goal is not None:
        # One request, many different incoming balls: the spread that
        # survives is the method's, not the problem's.
        tg = np.tile(np.asarray(args.goal, np.float32), (n, 1))
    return pos, vel, spin, tg


# ------------------------------------------------------------------ figures
def fig_landing(runs, targets, path):
    """Where each method actually put the ball, relative to where it was asked.

    Faceted rather than overplotted: a scatter is judged on every pair of
    colours at once, and only three slots of the palette clear that floor.
    One panel per method keeps a single series per axes, which has no pair
    to confuse, and stops five clouds of dots hiding each other.
    """
    import matplotlib.pyplot as plt
    _style(plt)

    k = len(runs)
    cols = min(k, 3)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.6 * rows),
                             squeeze=False)
    lim = 0.9
    for i, r in enumerate(runs):
        ax = axes[i // cols][i % cols]
        d = r.landing - np.column_stack([targets[:, 0], targets[:, 1],
                                         np.zeros(len(targets))])
        m = np.isfinite(d[:, 0]) & (r.outcome == 1)
        ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
        ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
        # The request sits at the origin by construction
        ax.plot(0, 0, marker="+", ms=13, mew=2.0, color=TARGET, zorder=5)
        ax.scatter(d[m, 0], d[m, 1], s=9, alpha=0.45, linewidths=0,
                   color=r.contender.colour_light, zorder=3)
        med = np.median(np.hypot(d[m, 0], d[m, 1])) if m.any() else np.nan
        ax.add_patch(plt.Circle((0, 0), med, fill=False, lw=1.6, ls="--",
                                color=r.contender.colour_light, zorder=4))
        ax.set_title(f"{r.contender.label}\nmedian {med * 100:.1f} cm  ·  "
                     f"{r.success_rate * 100:.0f}% on the table",
                     fontsize=9.5, color=INK, pad=8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        _despine(ax)
        # Bottom of its own column, which on a ragged last row is not the
        # same thing as being on the last row
        if i + cols >= k:
            ax.set_xlabel("long / short  (m)", fontsize=8.5)
        if i % cols == 0:
            ax.set_ylabel("across  (m)", fontsize=8.5)
    for j in range(k, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Where the ball landed, relative to the spot requested",
                 fontsize=12.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Two-line panel titles need room, or they sit on the ticks above them
    fig.subplots_adjust(hspace=0.42)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_per_dim(runs, path):
    """Median error in each part of the request, one panel per part.

    Five panels rather than five groups on one axis: the dimensions are
    metres, metres per second and radians per second, and putting them on a
    shared scale would be a second y-axis in disguise.
    """
    import matplotlib.pyplot as plt
    _style(plt)

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.6))
    names = [r.contender.label for r in runs]
    y = np.arange(len(runs))
    for d, ax in enumerate(axes):
        vals = [r.median(np.abs(r.error[:, d])) for r in runs]
        ax.barh(y, vals, height=0.62, zorder=3,
                color=[r.contender.colour_light for r in runs])
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.2f}" if d < 3 else f"  {v:.0f}",
                    va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(names if d == 0 else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(GOAL_TITLES[d], fontsize=10, color=INK, pad=6)
        ax.set_xlabel(GOAL_UNITS[d], fontsize=8.5)
        ax.set_xlim(0, max(vals) * 1.28 if max(vals) > 0 else 1)
        ax.grid(axis="y", visible=False)
        _despine(ax, keep=("bottom",))

    fig.suptitle("Median error per part of the request  ·  shorter is better",
                 fontsize=12.5, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_summary(runs, path):
    """The two numbers that decide it, each on its own axis."""
    import matplotlib.pyplot as plt
    _style(plt)

    # Thin marks: the row pitch grows with the count, the bar does not
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(runs) + 1.7))
    names = [r.contender.label for r in runs]
    y = np.arange(len(runs))
    cols = [r.contender.colour_light for r in runs]

    succ = [r.success_rate * 100 for r in runs]
    axes[0].barh(y, succ, height=0.5, color=cols, zorder=3)
    for i, v in enumerate(succ):
        axes[0].text(v, i, f"  {v:.1f}%", va="center", fontsize=9, color=INK2)
    axes[0].set_title("landed on the table", fontsize=11, color=INK, pad=6)
    axes[0].set_xlabel("% of shots", fontsize=8.5)
    axes[0].set_xlim(0, 108)

    goal = [r.median(r.goal_error) for r in runs]
    axes[1].barh(y, goal, height=0.5, color=cols, zorder=3)
    for i, v in enumerate(goal):
        axes[1].text(v, i, f"  {v:.3f}", va="center", fontsize=9, color=INK2)
    axes[1].set_title("weighted goal error  ·  lower is better",
                      fontsize=11, color=INK, pad=6)
    axes[1].set_xlabel("normalised units", fontsize=8.5)
    axes[1].set_xlim(0, max(goal) * 1.25 if max(goal) > 0 else 1)

    for j, ax in enumerate(axes):
        ax.set_yticks(y)
        ax.set_yticklabels(names if j == 0 else [], fontsize=9.5)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        _despine(ax, keep=("bottom",))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------------------------ table
def summary_rows(runs):
    rows = []
    for r in runs:
        rows.append({
            "method": r.contender.label,
            "spec": r.contender.spec,
            "success_rate": round(r.success_rate, 4),
            "place_err_median_m": round(r.median(r.place_error), 4),
            "goal_err_median": round(r.median(r.goal_error), 4),
            "err_land_x": round(r.median(np.abs(r.error[:, 0])), 4),
            "err_land_y": round(r.median(np.abs(r.error[:, 1])), 4),
            "err_speed": round(r.median(np.abs(r.error[:, 2])), 4),
            "err_topspin": round(r.median(np.abs(r.error[:, 3])), 2),
            "err_sidespin": round(r.median(np.abs(r.error[:, 4])), 2),
            "ms_per_stroke": round(r.plan_ms, 4),
        })
    return rows


def print_table(rows):
    """The table view. Three of the palette's light slots sit under 3:1 on
    paper, and the palette rule pays for that with visible labels and a
    readable table rather than with colour alone."""
    print(f"\n{'method':<20}{'in':>8}{'place':>10}{'goal':>9}"
          f"{'speed':>9}{'topspin':>10}{'sidespin':>10}{'ms':>10}")
    print("-" * 86)
    for r in sorted(rows, key=lambda r: r["goal_err_median"]):
        print(f"{r['method']:<20}"
              f"{r['success_rate'] * 100:7.1f}%"
              f"{r['place_err_median_m'] * 100:9.1f}cm"
              f"{r['goal_err_median']:9.3f}"
              f"{r['err_speed']:9.2f}"
              f"{r['err_topspin']:10.1f}"
              f"{r['err_sidespin']:10.1f}"
              f"{r['ms_per_stroke']:10.3f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs="+", default=["trainsize"], metavar="SPEC",
                    help="planners to compare; 'trainsize' and 'method' are presets")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--goal", type=float, nargs=5, default=None,
                    metavar=("X", "Y", "SPEED", "TOPSPIN", "SIDESPIN"),
                    help="ask every method for this one shot instead of the "
                         "held-out set's own targets")
    ap.add_argument("--n", type=int, default=800, help="problems to run")
    ap.add_argument("--evalset", default=EVAL_DEFAULT)
    ap.add_argument("--out", default=FIG_DIR)
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.list:
        for spec, label in compare.available():
            print(f"  {spec:<34} {label}")
        print("\npresets:")
        for name, specs in compare.PRESETS.items():
            print(f"  {name:<34} {', '.join(specs)}")
        return

    contenders = compare.build(args.compare)
    if not contenders:
        raise SystemExit("nothing to compare; try --list")

    pos, vel, spin, tg = load_problems(args)
    print(f"[compare] {len(contenders)} methods on {len(pos):,} problems"
          + (f", all asking for {args.goal}" if args.goal else
             " from the held-out set"))
    runs = compare.evaluate(contenders, pos, vel, spin, tg)

    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fig_landing(runs, tg, os.path.join(args.out, f"compare_landing{tag}.png"))
    fig_per_dim(runs, os.path.join(args.out, f"compare_per_dim{tag}.png"))
    fig_summary(runs, os.path.join(args.out, f"compare_summary{tag}.png"))

    rows = summary_rows(runs)
    print_table(rows)
    try:
        import pandas as pd
        csv = os.path.join(args.out, f"compare{tag}.csv")
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"\n  wrote {csv}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
