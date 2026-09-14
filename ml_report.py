"""Comparison figures for the training experiments.

    python ml_report.py            # reads runs/results.csv, writes runs/figures/

Colour follows the same rules as analyze_strokes.py: categorical hues in a
fixed validated order, one hue per method, never cycled. The speed/accuracy
scatter deliberately uses a single colour with direct labels instead --
scatter needs every pair of series to be distinguishable, which caps a
validated categorical palette at three, and there are more methods than that.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots, in order. Assigned per method and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Reading order for the legend and for colour assignment. Colour follows the
# method, so adding or removing one from a plot never repaints the others.
_BASE_ORDER = ["oracle", "knn", "policy-mlp", "mdn", "mdn+rank",
               "surrogate-cem", "surrogate-grad",
               "surrogate-cem-uniform", "surrogate-grad-uniform"]
# Only the policies see demonstrations; the forward models never do, so only
# these get demonstration-quality variants.
_TAUGHT = ("knn", "policy-mlp", "mdn", "mdn+rank")
# Demonstration variants, matching train_ai.demo_label(). Runs on better
# demonstrations appear as their own series, so a better-taught policy is
# compared against the original rather than replacing it.
VARIANTS = {
    "": ("", ("-", "o")),
    "hq": (", better demos", ("--", "s")),
    "robust": (", robust demos", (":", "^")),
    "hq+robust": (", better + robust demos", ("-.", "D")),
}
METHOD_ORDER = _BASE_ORDER + [f"{m} ({v} demos)" for v in VARIANTS if v
                              for m in _TAUGHT]
METHOD_LABEL = {
    "oracle": "CEM on true physics (oracle)",
    "knn": "nearest neighbour",
    "policy-mlp": "direct policy MLP",
    "mdn": "mixture density policy",
    "mdn+rank": "mixture + forward ranking",
    "surrogate-cem": "learned forward + CEM",
    "surrogate-grad": "learned forward + gradient",
    "surrogate-cem-uniform": "learned forward + CEM (uniform data)",
    "surrogate-grad-uniform": "learned forward + gradient (uniform data)",
}
METHOD_LABEL.update({f"{m} ({v} demos)": METHOD_LABEL[m] + text
                     for v, (text, _) in VARIANTS.items() if v
                     for m in _TAUGHT})


def split_variant(method):
    """``"mdn (hq demos)"`` -> ``("mdn", "hq")``."""
    if method.endswith(" demos)") and " (" in method:
        base, _, rest = method.rpartition(" (")
        return base, rest[:-len(" demos)")]
    return method, ""


# Hue identifies the method; the better-demonstration runs are the SAME
# method and inherit its hue, distinguished by line style instead. Giving
# them their own hues meant cycling the palette, which put two different
# series in the same colour -- categorical hues are assigned in fixed order
# and never cycled.
COLOUR = {m: SERIES[i % len(SERIES)] for i, m in enumerate(_BASE_ORDER)}
COLOUR.update({f"{m} ({v} demos)": COLOUR[m]
               for v in VARIANTS if v for m in _TAUGHT})
# The reference solver is not a series at all, so it gets neutral ink
COLOUR["oracle"] = "#52514e"


def style_for(method):
    """(linestyle, marker) -- the dash pattern marks demonstration quality."""
    return VARIANTS.get(split_variant(method)[1], VARIANTS[""])[1]


# Dash patterns for variants the report has not seen before, e.g. the ones a
# --run-tag sweep invents. Hue still follows the method, so a new variant can
# never repaint an existing series.
_SPARE_STYLES = [((0, (3, 1, 1, 1)), "v"), ((0, (5, 1)), "P"),
                 ((0, (1, 1)), "X"), ((0, (4, 1, 1, 1, 1, 1)), "*")]


def register_variants(families):
    """Give any unrecognised ``method (... demos)`` series a slot to plot in.

    Without this a swept run is silently dropped from every figure -- the
    plots iterate METHOD_ORDER, not the CSV.
    """
    for fam in dict.fromkeys(families):
        if fam in COLOUR:
            continue
        base, variant = split_variant(fam)
        if not variant or base not in COLOUR:
            continue
        if variant not in VARIANTS:
            style = _SPARE_STYLES[len(VARIANTS) % len(_SPARE_STYLES)]
            VARIANTS[variant] = (f", {variant} demos", style)
        COLOUR[fam] = COLOUR[base]
        METHOD_LABEL[fam] = METHOD_LABEL.get(base, base) + VARIANTS[variant][0]
        METHOD_ORDER.append(fam)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)


def new_fig(size=(6.8, 4.4)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def save(fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {p}")


def _sweep(df, out_dir, column, scale, title, ylabel, name, pct=False):
    """One line per method against training set size.

    Note the x axis mixes two kinds of sample. The forward models learn from
    randomly sampled strokes, which are cheap; the policies learn from CEM
    demonstrations, which each cost a solve. Equal counts are not equal
    effort, so read the curves as "samples of the kind this method needs".
    """
    # Wider than the other figures, with the legend outside the axes: placed
    # inside it sat exactly on top of the surrogate curves in the lower left
    fig, ax = new_fig((8.6, 4.6))
    plotted = 0
    for m in METHOD_ORDER:
        sub = df[df["family"] == m].sort_values("n_train")
        if sub.empty:
            continue
        if m == "oracle":
            # No training set, so it is a horizontal reference, not a line
            ax.axhline(sub[column].iloc[0] * scale, color=COLOUR[m],
                       linestyle=":", linewidth=1.8, zorder=3)
            ax.text(0.995, sub[column].iloc[0] * scale, " oracle",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    color=COLOUR[m], fontsize=8)
            plotted += 1
            continue
        ls, mk = style_for(m)
        ax.plot(sub["n_train"], sub[column] * scale, marker=mk, markersize=7,
                linewidth=2.0, linestyle=ls, color=COLOUR[m],
                label=METHOD_LABEL[m], markeredgecolor=SURFACE,
                markeredgewidth=1.0, zorder=3)
        plotted += 1
    ax.set_xscale("log")
    if pct:
        ax.set_ylim(0, 103)
    style(ax, title,
          "training samples (log scale)\n"
          "solid = 5x64 demonstrations,   dashed = 8x256 demonstrations\n"
          "policies learn from CEM demonstrations, forward models from random strokes",
          ylabel)
    ax.xaxis.label.set_fontsize(8)
    if plotted >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY,
                  loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, out_dir, name)


def plot_speed_accuracy(df, out_dir):
    """Success rate against inference cost.

    Single colour with direct labels: a scatter needs every pair of series to
    be separable, which caps a validated categorical palette at three, and
    there are more methods than that here.
    """
    fig, ax = new_fig((9.2, 5.4))
    best = df.sort_values("success_rate").groupby("family").tail(1)
    x = best["infer_ms_per_stroke"].to_numpy()
    y = best["success_rate"].to_numpy() * 100
    labels = [METHOD_LABEL.get(f, f) for f in best["family"]]
    ax.scatter(x, y, s=90, color=SERIES[0], edgecolors=SURFACE,
               linewidths=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(0, 112)

    # The methods worth comparing are the fast accurate ones, so they pile
    # into one corner and their labels landed on top of each other. Push the
    # labels apart along y and run a leader back to the point each belongs
    # to; a label is only readable if it is attached to something.
    lx = np.log10(np.clip(x, 1e-6, None))
    span = lx.max() - lx.min() or 1.0
    order = np.argsort(-y)
    min_gap = 5.6                                   # in y units (percent)
    placed = {}
    last = None
    for i in order:
        ly = y[i] if last is None else min(y[i], last - min_gap)
        placed[i] = ly
        last = ly
    for i, (xi, yi, text) in enumerate(zip(x, y, labels)):
        # Labels to the right of the point, except on the far right of the
        # axis where there is no room
        right = lx[i] < lx.min() + 0.72 * span
        tx = xi * (1.45 if right else 0.62)
        ha = "left" if right else "right"
        ax.annotate(text, (tx, placed[i]), fontsize=8.5, color=INK_SECONDARY,
                    va="center", ha=ha, zorder=5)
        ax.plot([xi, tx], [yi, placed[i]], lw=0.7, color=INK_SECONDARY,
                alpha=0.45, zorder=2)
    style(ax, "Best result per method: accuracy against inference cost",
          "inference time per stroke (ms, log scale)", "success rate (%)")
    save(fig, out_dir, "speed_vs_accuracy.png")


def plot_outcomes(df, out_dir):
    """Where the misses go, for the best run of each method."""
    best = df.sort_values("success_rate").groupby("family").tail(1)
    best = best.set_index("family").reindex(
        [m for m in METHOD_ORDER if m in set(best["family"])])
    if best.empty:
        return
    labels = [METHOD_LABEL.get(m, m) for m in best.index]
    y = np.arange(len(best))
    good = best["success_rate"].to_numpy() * 100
    net = best["net_rate"].to_numpy() * 100
    out = best["out_rate"].to_numpy() * 100
    other = np.clip(100 - good - net - out, 0, None)

    fig, ax = new_fig((7.4, 0.52 * len(best) + 2.0))
    left = np.zeros(len(best))
    # A sequential progression from "good" to "worst", not categorical hues:
    # these are ordered outcomes of one measure, not independent series.
    for vals, name, colour in ((good, "landed in", "#2a78d6"),
                               (net, "into the net", "#9ec5f4"),
                               (out, "out", "#cde2fb"),
                               (other, "no return", "#e1e0d9")):
        ax.barh(y, vals, left=left, height=0.62, label=name,
                color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    for yi, g in zip(y, good):
        ax.text(min(g + 1.5, 96), yi, f"{g:.0f}%", va="center",
                fontsize=8, color=INK_SECONDARY)
    style(ax, "Outcome breakdown, best run per method", "share of strokes (%)", "")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY,
              ncol=4, loc="lower right", bbox_to_anchor=(1.0, -0.28))
    save(fig, out_dir, "outcome_breakdown.png")


def plot_per_dimension(df, out_dir):
    """Median error in each goal dimension, in its own natural units.

    Separate panels rather than one chart: metres, m/s and rad/s share no
    axis, and forcing them onto one would be the dual-axis mistake wearing a
    different hat.
    """
    from pingpong import dataset as D
    best = df.sort_values("success_rate").groupby("family").tail(1)
    fams = [m for m in METHOD_ORDER if m in set(best["family"])]
    if not fams:
        return
    best = best.set_index("family").reindex(fams)

    cols = [f"err_{k}" for k in D.GOAL_NAMES]
    have = [c for c in cols if c in best.columns]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(3.0 * len(have), 4.2))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    y = np.arange(len(fams))
    for ax, col in zip(axes, have):
        i = cols.index(col)
        ax.set_facecolor(SURFACE)
        ax.barh(y, best[col].to_numpy(), height=0.62, color=SERIES[0], zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABEL.get(f, f) for f in fams], fontsize=7)
        ax.invert_yaxis()
        style(ax, D.GOAL_NAMES[i], f"median error ({D.GOAL_UNITS[i]})", "")
        ax.grid(axis="y", visible=False)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("Error per goal dimension, best run of each method",
                 color=INK, fontsize=11)
    save(fig, out_dir, "per_dimension_error.png")


def print_table(df):
    """One row per method: its best run, and how much data that took.

    The row is picked by success rate, not by dataset size, and those are
    not always the same row -- nearest neighbour peaks at 20k and is flat or
    slightly worse at 200k. Under a column headed `n_train` that reads as
    "the top of the ladder", so the column says what it actually is and a
    marker flags any method whose best run is not its largest.
    """
    best = df.sort_values("success_rate").groupby("family").tail(1)
    largest = df.groupby("family")["n_train"].max()
    best = best.set_index("family").reindex(
        [m for m in METHOD_ORDER if m in set(best["family"])])
    print(f"\n{'method':<32}{'best at n':>11}{'success':>10}"
          f"{'place err':>12}{'goal err':>11}{'ms/stroke':>11}")
    print("-" * 84)
    for fam, r in best.iterrows():
        n = int(r["n_train"])
        # "<" means more data was tried and did not help
        mark = "<" if n < largest.get(fam, n) else " "
        print(f"{METHOD_LABEL.get(fam, fam):<32}{n:>10,}{mark}"
              f"{r['success_rate'] * 100:9.1f}%{r['place_err_median_m'] * 100:10.1f} cm"
              f"{r['goal_err_median']:11.3f}{r['infer_ms_per_stroke']:11.2f}")
    if (best["n_train"] < best.index.map(largest)).any():
        print("\n  < best run is not the largest one: more data did not help")


def build_report(results_csv, out_dir):
    if not os.path.exists(results_csv):
        raise SystemExit(f"{results_csv} not found -- run `python train_ai.py eval` first")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    # A run scored on fewer held-out problems is a smoke test, not a result;
    # mixing the two into one figure compares numbers that are not comparable.
    if "eval_n" in df.columns and df["eval_n"].notna().any():
        df = df[df["eval_n"].fillna(0) >= df["eval_n"].max()]
    register_variants(df["family"])

    print(f"[report] {len(df)} runs -> {out_dir}/")
    _sweep(df, out_dir, "success_rate", 100.0,
           "Does it get the ball on the table?",
           "success rate (%)", "success_vs_data.png", pct=True)
    _sweep(df, out_dir, "place_err_median_m", 100.0,
           "How close to the spot it was aiming at",
           "median placement error (cm)", "error_vs_data.png")
    _sweep(df, out_dir, "goal_err_median", 1.0,
           "Whole shot: place, speed and spin combined",
           "weighted goal error (normalised units)", "goal_error_vs_data.png")
    plot_per_dimension(df, out_dir)
    plot_speed_accuracy(df, out_dir)
    plot_outcomes(df, out_dir)
    print_table(df)


if __name__ == "__main__":
    build_report(os.path.join("runs", "results.csv"), os.path.join("runs", "figures"))
