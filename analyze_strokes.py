"""Analyse stroke data written by the game.

    python analyze_strokes.py
    python analyze_strokes.py --csv strokes.csv --out outputs

Replaces the old analyze_states.py and plot_landing_trajectories.py, which
both parsed the old 7-column states.csv. Those two overlapped heavily and
carried machinery that is now obsolete:

* string vector parsing -- the new schema stores flat numeric columns;
* drag-free ballistic trajectory reconstruction -- wrong for a ball whose
  drag exceeds gravity, and unnecessary now that pingpong.physics can
  replay the exact trajectory the game used;
* hardcoded table dimensions -- they live in pingpong.constants;
* kNN "how should I hit this" retrieval -- superseded by
  coach_game.solve_stroke, which inverts the real physics instead of
  averaging neighbours.

What is left is descriptive: what did the player actually do, and what
happened as a result.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from pingpong import constants as C
from pingpong import physics

# ---------------------------------------------------------------- palette
# Validated categorical slots (all-pairs safe up to three series).
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
SERIES_3 = "#1baf7a"   # aqua

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Topspin vs backspin is a polarity, so it gets a diverging ramp: two hues
# with a neutral midpoint. Never a rainbow, and never a hue in the middle.
SPIN_CMAP = LinearSegmentedColormap.from_list(
    "spin", ["#e34948", INK_MUTED, SERIES_1]     # backspin - neutral - topspin
)
# Density is a magnitude, so it gets one hue, light to dark.
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95"]
)


def style_axes(ax, title, xlabel, ylabel):
    """Recessive grid and axes; text wears ink tokens, never a series colour."""
    ax.set_title(title, color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


def new_fig(figsize=(6.4, 4.4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def draw_table_outline(ax):
    hx, hy = C.TABLE_L / 2, C.TABLE_W / 2
    ax.plot([-hx, hx, hx, -hx, -hx], [-hy, -hy, hy, hy, -hy],
            color=BASELINE, linewidth=1.4, zorder=1)
    ax.axvline(0, color=BASELINE, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_aspect("equal")


# ---------------------------------------------------------------- loading
def vec(df, name):
    """Pull an (N, 3) array out of the flat <name>_x/_y/_z columns."""
    return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(dtype=float)


def load(csv_path):
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"{csv_path} not found. Play a few rallies first "
            f"(python main_wss.py) -- every stroke appends a row."
        )
    df = pd.read_csv(csv_path)
    if "schema" not in df.columns:
        raise SystemExit(
            f"{csv_path} looks like the old states.csv format. The new schema "
            f"is written to strokes.csv by the current main_wss.py."
        )
    return df


# ---------------------------------------------------------------- report
def summarise(df):
    print("=== dataset ===")
    print(f"  rows           {len(df)}")
    print(f"  rallies        {df['rally_id'].nunique()}")

    counts = df["result"].value_counts()
    total = len(df)
    print("  outcomes")
    for k, v in counts.items():
        print(f"    {k:<10} {v:5d}  ({v / total * 100:4.1f}%)")

    hit = df[df["result"] != "miss"]
    if hit.empty:
        print("  (no contacts recorded yet)")
        return hit

    out_speed = np.linalg.norm(vec(hit, "out_vel"), axis=1)
    in_speed = np.linalg.norm(vec(hit, "in_vel"), axis=1)
    swing = np.linalg.norm(vec(hit, "paddle_vel"), axis=1)
    spin = np.linalg.norm(vec(hit, "out_spin"), axis=1)

    print("  contacts")
    print(f"    incoming speed  {in_speed.mean():5.2f} +/- {in_speed.std():4.2f} m/s")
    print(f"    outgoing speed  {out_speed.mean():5.2f} +/- {out_speed.std():4.2f} m/s")
    print(f"    swing speed     {swing.mean():5.2f} +/- {swing.std():4.2f} m/s")
    print(f"    spin            {spin.mean():5.0f} +/- {spin.std():4.0f} rad/s")
    return hit


# ---------------------------------------------------------------- plots
def plot_landing(df, out_dir):
    good = df[df["result"] == "in"]
    if good.empty:
        return
    land = vec(good, "landing")

    # One series, so no legend -- the title names it.
    fig, ax = new_fig((6.4, 4.0))
    draw_table_outline(ax)
    ax.scatter(land[:, 0], land[:, 1], s=26, color=SERIES_1,
               edgecolors=SURFACE, linewidths=0.8, zorder=3)
    style_axes(ax, f"Landing points that stayed in ({len(good)} strokes)",
               "x  (towards opponent, m)", "y  (across table, m)")
    save(fig, out_dir, "landing_scatter.png")

    if len(good) >= 25:
        fig, ax = new_fig((6.4, 4.0))
        hb = ax.hexbin(land[:, 0], land[:, 1], gridsize=22, mincnt=1,
                       cmap=DENSITY_CMAP, zorder=2)
        draw_table_outline(ax)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("strokes", color=INK_SECONDARY, fontsize=9)
        cb.ax.tick_params(colors=INK_MUTED, labelsize=8)
        style_axes(ax, "Landing density", "x (m)", "y (m)")
        save(fig, out_dir, "landing_density.png")


def plot_outcomes(df, out_dir):
    order = ["in", "out", "net", "own_side", "miss"]
    counts = df["result"].value_counts()
    labels = [o for o in order if o in counts.index]
    values = [int(counts[o]) for o in labels]
    if not values:
        return

    # A single measure across named categories: one colour, direct labels on
    # the bars. Colour would carry no extra information here.
    fig, ax = new_fig((6.0, 0.5 * len(labels) + 1.8))
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=SERIES_1, height=0.6, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    total = sum(values)
    for y, v in zip(ypos, values):
        ax.text(v + total * 0.012, y, f"{v}  ({v / total * 100:.0f}%)",
                va="center", color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, max(values) * 1.22)
    style_axes(ax, f"Stroke outcomes ({total} strokes)", "count", "")
    ax.grid(axis="y", visible=False)
    save(fig, out_dir, "outcomes.png")


def plot_swing_relations(hit, out_dir):
    if len(hit) < 5:
        return
    swing = vec(hit, "paddle_vel")
    swing_speed = np.linalg.norm(swing, axis=1)
    out_speed = np.linalg.norm(vec(hit, "out_vel"), axis=1)
    in_speed = np.linalg.norm(vec(hit, "in_vel"), axis=1)
    spin_y = vec(hit, "out_spin")[:, 1]

    # Swing speed -> ball speed
    fig, ax = new_fig()
    ax.scatter(swing_speed, out_speed, s=26, color=SERIES_1,
               edgecolors=SURFACE, linewidths=0.8, zorder=3)
    style_axes(ax, "Swing speed drives ball speed",
               "swing speed (m/s)", "outgoing ball speed (m/s)")
    save(fig, out_dir, "swing_vs_ballspeed.png")

    # Incoming -> outgoing. The old physics discarded the incoming ball
    # entirely, so this plot was flat by construction; it should now show
    # real structure.
    fig, ax = new_fig()
    ax.scatter(in_speed, out_speed, s=26, color=SERIES_2,
               edgecolors=SURFACE, linewidths=0.8, zorder=3)
    style_axes(ax, "Incoming speed feeds through to the return",
               "incoming ball speed (m/s)", "outgoing ball speed (m/s)")
    save(fig, out_dir, "incoming_vs_outgoing.png")

    # Vertical swing component -> spin. Brushing up gives topspin, chopping
    # down gives backspin, and the sign of the y-axis says which.
    fig, ax = new_fig()
    ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=2)
    ax.scatter(swing[:, 2], spin_y, s=26, color=SERIES_3,
               edgecolors=SURFACE, linewidths=0.8, zorder=3)
    style_axes(ax, "Brushing up creates topspin, chopping down creates backspin",
               "vertical swing velocity (m/s)", "spin about y  (rad/s)")
    # Anchored right so they cannot collide with the y-axis tick labels
    ax.text(0.985, 0.965, "topspin", transform=ax.transAxes,
            ha="right", va="top", color=INK_SECONDARY, fontsize=9)
    ax.text(0.985, 0.035, "backspin", transform=ax.transAxes,
            ha="right", va="bottom", color=INK_SECONDARY, fontsize=9)
    save(fig, out_dir, "swing_vs_spin.png")


def plot_trajectories(hit, out_dir, max_traj=200):
    """Replay post-contact trajectories with the real physics.

    The old script integrated a drag-free parabola. Drag on a table tennis
    ball exceeds gravity, so that reconstruction did not match what the game
    actually simulated; this replays the same integrator the game uses.
    """
    good = hit[hit["result"] == "in"]
    if good.empty:
        return
    if len(good) > max_traj:
        good = good.iloc[np.linspace(0, len(good) - 1, max_traj).astype(int)]

    pos = vec(good, "in_pos")
    vel = vec(good, "out_vel")
    spin = vec(good, "out_spin")

    fig = plt.figure(figsize=(8.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURFACE)

    # Real table proportions, otherwise a 2.74 x 1.525 m table renders
    # nearly square and the trajectories look wrong
    ax.set_box_aspect((C.TABLE_L, C.TABLE_W, 0.85))
    # Recessive panes and grid; matplotlib's 3D defaults are far too heavy
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(GRIDLINE)
        axis._axinfo["grid"].update(color=GRIDLINE, linewidth=0.6)

    # Table surface
    hx, hy = C.TABLE_L / 2, C.TABLE_W / 2
    X, Y = np.meshgrid([-hx, hx], [-hy, hy])
    ax.plot_surface(X, Y, np.full_like(X, C.TABLE_H), alpha=0.12,
                    color=SERIES_1, linewidth=0, zorder=1)
    ax.plot([-hx, hx, hx, -hx, -hx], [-hy, -hy, hy, hy, -hy],
            [C.TABLE_H] * 5, color=BASELINE, linewidth=1.2)
    # Net
    ax.plot([0, 0], [-hy, hy], [C.NET_TOP_Z] * 2, color=INK_MUTED, linewidth=1.4)

    spin_y = spin[:, 1]
    lim = max(60.0, float(np.abs(spin_y).max()))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    for p, v, w in zip(pos, vel, spin):
        res = physics.simulate(p, v, w, record_trajectory=True, record_stride=4,
                               max_time=3.0)
        t = res.trajectory
        if t is None or len(t) < 2:
            continue
        ax.plot(t[:, 0], t[:, 1], t[:, 2],
                color=SPIN_CMAP(norm(w[1])), linewidth=0.9, alpha=0.75)

    sm = plt.cm.ScalarMappable(cmap=SPIN_CMAP, norm=norm)
    # pad has to clear the 3D z-axis tick labels, which sit outside the axes box
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.12, aspect=18)
    cb.set_label("spin about y (rad/s)", color=INK_SECONDARY, fontsize=8)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=7)
    # Name the two poles instead of a long rotated label
    cb.ax.text(0.5, 1.04, "topspin", transform=cb.ax.transAxes, ha="center",
               va="bottom", color=INK_SECONDARY, fontsize=8)
    cb.ax.text(0.5, -0.04, "backspin", transform=cb.ax.transAxes, ha="center",
               va="top", color=INK_SECONDARY, fontsize=8)

    ax.set_xlabel("x (m)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("y (m)", color=INK_SECONDARY, fontsize=9)
    ax.set_zlabel("z (m)", color=INK_SECONDARY, fontsize=9)
    # suptitle, not set_title: the 3D axes box is offset by the colorbar, so an
    # axes-anchored title drifts off the left edge
    fig.suptitle(f"Post-contact trajectories, replayed with drag and Magnus "
                 f"({len(good)} strokes)", color=INK, fontsize=11, y=0.97)
    ax.tick_params(colors=INK_MUTED, labelsize=7)
    ax.view_init(elev=22, azim=-60)
    fig.subplots_adjust(left=0.02, right=0.86, top=0.99, bottom=0.01)
    path = os.path.join(out_dir, "trajectories_3d.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="strokes.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-traj", type=int, default=200)
    args = ap.parse_args()

    df = load(args.csv)
    os.makedirs(args.out, exist_ok=True)

    hit = summarise(df)

    print(f"\n=== figures -> {args.out}/ ===")
    plot_outcomes(df, args.out)
    plot_landing(df, args.out)
    if not hit.empty:
        plot_swing_relations(hit, args.out)
        plot_trajectories(hit, args.out, args.max_traj)

    print("\nTo work out how to reach a specific landing spot, use the solver "
          "rather than this script:\n"
          "  from coach_game import solve_stroke\n"
          "It inverts the real physics instead of averaging past strokes.")


if __name__ == "__main__":
    main()
