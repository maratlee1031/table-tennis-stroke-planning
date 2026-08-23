"""What does GOAL_WEIGHTS actually buy, and what does it cost?

    python sweep_weights.py
    python sweep_weights.py --n 400 --iters 8 --pop 256

`dataset.GOAL_WEIGHTS` is [1.0, 1.0, 0.6, 0.5, 0.5] and has never been
varied, yet it decides what the search gives up once placement is nearly
exhausted. The 200k sweep made that concrete: the mixture policy's placement
kept improving to 6.1 cm while its sidespin error went 40.5 -> 63.7, because
under these weights a centimetre of placement is worth more than the spin it
costs. That is the objective working as specified, not the model failing.

This maps the frontier directly, on the true physics, with no learning in
the way: run the reference solver under several weightings of the same
problems and measure the result in natural units. Whatever the solver cannot
trade away is a property of the task; everything else is a choice.

Writes runs/figures/weights_frontier.png and runs/weights_sweep.csv.
"""

import argparse
import os

import numpy as np

from pingpong import compare, dataset, serve, solver

FIG_DIR = os.path.join("runs", "figures")
OUT_CSV = os.path.join("runs", "weights_sweep.csv")

# Each row is (label, weights over land_x, land_y, speed, topspin, sidespin).
# The default sits in the middle; the others walk the emphasis from placement
# to spin so the frontier has both ends.
SETTINGS = [
    ("placement only", [1.0, 1.0, 0.0, 0.0, 0.0]),
    ("placement heavy", [1.0, 1.0, 0.3, 0.15, 0.15]),
    ("default", list(dataset.GOAL_WEIGHTS)),
    ("balanced", [1.0, 1.0, 0.8, 1.0, 1.0]),
    ("spin heavy", [1.0, 1.0, 0.8, 2.0, 2.0]),
    ("spin only", [0.0, 0.0, 0.0, 1.0, 1.0]),
]


def run(args):
    rng = np.random.default_rng(args.seed)
    pos, vel, spin = serve.incoming_at(dataset.STRIKE_X, args.n,
                                       np.random.default_rng(args.seed + 1))
    goals, ok = dataset.achievable_goals(pos, vel, spin, rng)
    # Only problems with a known solution, so a miss is the method's fault
    pos, vel, spin, goals = pos[ok], vel[ok], spin[ok], goals[ok]
    print(f"[weights] {len(pos):,} solvable problems, "
          f"CEM {args.iters}x{args.pop}\n")

    rows = []
    for label, w in SETTINGS:
        w = np.asarray(w, dtype=float)
        acts, _, _, _ = solver.solve_batch(
            pos, vel, spin, goals, iters=args.iters, pop=args.pop,
            rng=np.random.default_rng(args.seed), weights=w)
        achieved, outcome, _ = dataset.apply_actions(pos, vel, spin, acts)
        err = np.abs(achieved - goals)
        m = np.isfinite(err[:, 0])
        place = np.hypot(err[m, 0], err[m, 1])
        spin_err = 0.5 * (err[m, 3] + err[m, 4])
        rows.append({
            "setting": label,
            "w_land": w[0], "w_speed": w[2], "w_spin": w[3],
            "success_rate": round(float((outcome == 1).mean()), 4),
            "place_cm": round(float(np.median(place)) * 100, 2),
            "speed_err": round(float(np.median(err[m, 2])), 3),
            "topspin_err": round(float(np.median(err[m, 3])), 1),
            "sidespin_err": round(float(np.median(err[m, 4])), 1),
            "spin_err": round(float(np.median(spin_err)), 1),
        })
        r = rows[-1]
        print(f"  {label:<18} place {r['place_cm']:6.2f} cm   "
              f"speed {r['speed_err']:5.2f}   spin {r['spin_err']:6.1f} rad/s   "
              f"in {r['success_rate'] * 100:5.1f}%")
    return rows


def figure(rows, path):
    """Placement against spin, one point per weighting: the frontier itself."""
    import matplotlib.pyplot as plt
    from compare_models import _style, _despine, INK, INK2, MUTED

    _style(plt)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    xs = [r["place_cm"] for r in rows]
    ys = [r["spin_err"] for r in rows]
    ax.plot(xs, ys, "-", lw=1.6, color=MUTED, zorder=2, alpha=0.7)
    for i, r in enumerate(rows):
        c = compare.PALETTE_LIGHT[i % len(compare.PALETTE_LIGHT)]
        default = r["setting"] == "default"
        ax.scatter([r["place_cm"]], [r["spin_err"]],
                   s=190 if default else 110, zorder=4, color=c,
                   edgecolors="#0b0b0b" if default else "none",
                   linewidths=1.4 if default else 0)
        ax.annotate(r["setting"] + ("  (shipped)" if default else ""),
                    (r["place_cm"], r["spin_err"]),
                    textcoords="offset points", xytext=(10, 6),
                    fontsize=9.5, color=INK2)
    ax.set_xlabel("median placement error (cm)  ->  worse", fontsize=9.5)
    ax.set_ylabel("median spin error (rad/s)  ->  worse", fontsize=9.5)
    ax.set_title("What the goal weights trade away\n"
                 "reference solver on true physics, same problems throughout",
                 fontsize=12, color=INK, pad=10)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = run(args)
    os.makedirs(FIG_DIR, exist_ok=True)
    figure(rows, os.path.join(FIG_DIR, "weights_frontier.png"))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"  wrote {OUT_CSV}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
