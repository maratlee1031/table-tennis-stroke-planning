"""Run several planners on the same problem and compare what they produced.

One incoming ball, one requested shot, every contender planning against it.
Shared by the live overlay in ai_play.py and the static figures in
compare_models.py, so the two can never disagree about what a method scored.

A contender is named by a spec string:

    oracle                  CEM on the true physics (the reference)
    oracle:8,256            ... with a specific iterations,population budget
    mdn_20000_i8p256        a checkpoint in runs/checkpoints (.pt optional)
    runs/checkpoints/x.pt   or a path to one
    knn:expert_20000_i8p256 retrieval over a demonstration set in runs/data

and two presets expand to a ready-made comparison:

    trainsize   the same method at every data size, plus the oracle
    method      every method at the largest data size, plus the oracle
"""

import os
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from . import agents, constants as C, dataset, models, physics

CKPT_DIR = os.path.join("runs", "checkpoints")
DATA_DIR = os.path.join("runs", "data")

# Categorical slots from the validated palette, in the documented order but
# with green removed -- green is the requested-landing marker throughout, and
# a contender wearing it would read as "this one is the target".
#
# Validated as a set with scripts/validate_palette.js: adjacent-pair CVD
# worst 9.1 light / 8.4 dark, normal-vision worst 19.6 / 19.3, all slots
# inside the lightness band and over 3:1 on the dark scene surface. Three
# light slots fall under 3:1 on paper, so every figure carries direct labels
# and a printed table -- the relief the palette rule requires.
#
# The adjacent pairlist is what bars and lines are held to. A scatter is
# judged on all pairs, where only the first three slots clear the floors, so
# the landing scatter is facetted one panel per contender rather than
# overplotted -- see compare_models.py.
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#4a3aa7", "#e34948"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#9085e9", "#e66767"]

TARGET_LIGHT = "#008300"
TARGET_DARK = "#1fbd4a"

# The first entry plays the ball for real in the viewer, and it is planned
# while the ball is still in the air -- so it must be a fast one. The oracle
# needs a few hundred milliseconds and goes last, where it is planned during
# the frozen breakdown and costs nothing visible.
PRESETS = {
    "trainsize": ["mdn_200000_i8p256", "mdn_20000_i8p256", "mdn_2000_i8p256",
                  "mdn_500_i8p256", "oracle"],
    "method": ["mdn_200000_i8p256", "policy_200000_i8p256",
               "knn:expert_200000_i8p256", "forwardmix_200000", "oracle"],
}


def _hex_to_rgba(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
            int(h[4:6], 16) / 255, a)


def _human_n(n):
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def label_for(stem):
    """A short display name for a checkpoint or dataset stem.

    ``mdn_20000_i8p256`` -> ``MDN 20k hq``. The budget suffix matters: two
    checkpoints differing only in it are different experiments, and a label
    that dropped it would put two unrelated curves under one name.
    """
    kinds = {"mdn": "MDN", "policy": "MLP", "forward": "fwd-uniform",
             "forwardmix": "fwd+CEM", "expert": "kNN"}
    parts = stem.split("_")
    kind = kinds.get(parts[0], parts[0])
    bits = [kind]
    for p in parts[1:]:
        if p.isdigit():
            bits.append(_human_n(int(p)))
        elif p.startswith("i") and "p" in p:
            bits.append("hq")
        else:
            bits.append(p)
    return " ".join(bits)


@dataclass
class Contender:
    """One planner in the comparison."""

    spec: str
    label: str
    agent: object
    slot: int = 0
    on: bool = True                      # display only; toggled live in the overlay
    # Lets a caller score the stroke it actually played rather than a fresh
    # plan. The oracle and the surrogate search both advance an RNG, so
    # re-planning them would put a different stroke in the table from the one
    # on screen.
    forced_action: np.ndarray = None

    @property
    def colour_light(self):
        return PALETTE_LIGHT[self.slot % len(PALETTE_LIGHT)]

    @property
    def colour_dark(self):
        return PALETTE_DARK[self.slot % len(PALETTE_DARK)]

    @property
    def rgba(self):
        return _hex_to_rgba(self.colour_dark)

    @property
    def is_oracle(self):
        return isinstance(self.agent, agents.CEMOracle)


@dataclass
class Shot:
    """What one contender did with one problem."""

    contender: Contender
    action: np.ndarray
    achieved: np.ndarray                 # the 5-dim goal it actually produced
    error: np.ndarray                    # achieved - requested, per dimension
    goal_error: float                    # weighted, the single number
    landing: np.ndarray = None
    outcome: int = 0                     # dataset/physics batch code; 1 is in
    trajectory: np.ndarray = None
    plan_ms: float = 0.0

    @property
    def landed_in(self):
        return self.outcome == 1

    @property
    def measured(self):
        """A shot that never reached the table has no per-dimension result.

        apply_actions returns NaN for it, and NaN formatted into a table
        reads as a broken number rather than as "this one missed".
        """
        return bool(np.isfinite(self.goal_error))

    @property
    def place_error(self):
        return float(np.hypot(self.error[0], self.error[1]))


def expand(specs):
    """Expand preset names; leave everything else alone."""
    out = []
    for s in specs:
        out.extend(PRESETS[s] if s in PRESETS else [s])
    return out


def available(ckpt_dir=CKPT_DIR, data_dir=DATA_DIR):
    """Every spec that could be compared right now, for --list."""
    rows = []
    for f in sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []:
        if f.endswith(".pt"):
            stem = f[:-3]
            rows.append((stem, label_for(stem)))
    for f in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []:
        if f.startswith("expert_") and f.endswith(".npz"):
            stem = f[:-4]
            rows.append((f"knn:{stem}", label_for(stem)))
    return rows


def build_one(spec, dev=None, ckpt_dir=CKPT_DIR, data_dir=DATA_DIR):
    """One spec string -> (label, agent). Raises if it cannot be resolved."""
    if spec == "oracle" or spec.startswith("oracle:"):
        iters, pop = 8, 256
        if ":" in spec:
            a, _, b = spec.partition(":")[2].partition(",")
            iters, pop = int(a), int(b)
        return f"CEM oracle {iters}x{pop}", agents.CEMOracle(iters=iters, pop=pop)

    if spec.startswith("knn:"):
        name = spec[4:]
        path = name if os.path.exists(name) else os.path.join(
            data_dir, name if name.endswith(".npz") else name + ".npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        stem = os.path.basename(path)[:-4]
        return label_for(stem), agents.KNNAgent(dataset.load(path), k=5)

    path = spec if os.path.exists(spec) else os.path.join(
        ckpt_dir, spec if spec.endswith(".pt") else spec + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model, _ = models.load(path)
    kind = getattr(model, "kind", "?")
    stem = os.path.basename(path)[:-3]
    if kind == "mdn":
        return label_for(stem), agents.MDNAgent(model, dev=dev)
    if kind == "policy":
        return label_for(stem), agents.PolicyAgent(model, dev=dev)
    return label_for(stem), agents.SurrogateCEMAgent(model, dev=dev)


def build(specs, dev=None, ckpt_dir=CKPT_DIR, data_dir=DATA_DIR, quiet=False):
    """Resolve a list of specs into contenders, skipping any that are missing.

    A missing checkpoint is reported and dropped rather than fatal: the point
    of the presets is that they name the whole ladder, and half a ladder is
    still a useful comparison when a sweep has not finished.
    """
    out = []
    for spec in expand(specs):
        try:
            label, agent = build_one(spec, dev, ckpt_dir, data_dir)
        except FileNotFoundError as e:
            if not quiet:
                print(f"[compare] skipping {spec}: {e} not found")
            continue
        out.append(Contender(spec=spec, label=label, agent=agent, slot=len(out)))
    return out


def play(contenders, pos, vel, spin, goal, trajectory=True):
    """Every contender plans for the same ball, then all are played for real.

    The achieved goal comes from dataset.apply_actions -- the same call the
    evaluation in train_ai.py uses -- so a number here is directly comparable
    to a row of results.csv. The drawable path is a second, separate
    simulation because apply_actions reports outcomes, not trajectories.

    Every contender passed in is planned, including ones currently switched
    off in a viewer: ``on`` is a display flag, and planning them all means
    toggling one back on is instant rather than costing a rally.
    """
    active = list(contenders)
    if not active:
        return []

    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    spin = np.asarray(spin, dtype=float)
    goal = np.asarray(goal, dtype=np.float32)

    state = dataset.encode_state(pos[None, :], vel[None, :],
                                 spin[None, :]).astype(np.float32)
    g = goal[None, :]

    acts, times = [], []
    for c in active:
        if c.forced_action is not None:
            acts.append(np.asarray(c.forced_action, dtype=float))
            times.append(0.0)
            continue
        t0 = perf_counter()
        if c.is_oracle:
            a = c.agent.act(state, g, pos[None, :], vel[None, :], spin[None, :])[0]
        else:
            a = c.agent.act(state, g)[0]
        times.append((perf_counter() - t0) * 1000.0)
        acts.append(np.asarray(a, dtype=float))

    acts = np.stack(acts)
    n = len(acts)
    achieved, outcome, landing = dataset.apply_actions(
        np.repeat(pos[None, :], n, 0), np.repeat(vel[None, :], n, 0),
        np.repeat(spin[None, :], n, 0), acts)

    shots = []
    normals, pvels = dataset.decode_action(acts)
    for i, c in enumerate(active):
        err = achieved[i] - goal
        traj = None
        if trajectory:
            out_v, out_w = physics.collide(
                vel[None, :], spin[None, :], normals[i][None, :],
                C.RESTITUTION_PADDLE, C.FRICTION_PADDLE,
                surface_vel=pvels[i][None, :])
            start = pos + normals[i] * (C.BALL_RADIUS * 1.6)
            r = physics.simulate(start, out_v[0], out_w[0], max_time=3.0,
                                 record_trajectory=True)
            traj = r.trajectory
        shots.append(Shot(
            contender=c, action=acts[i], achieved=achieved[i], error=err,
            goal_error=float(dataset.goal_error(achieved[i][None, :], g)[0]),
            landing=landing[i] if np.all(np.isfinite(landing[i])) else None,
            outcome=int(outcome[i]), trajectory=traj, plan_ms=times[i]))
    return shots


@dataclass
class Run:
    """One contender's result over a whole set of problems."""

    contender: Contender
    actions: np.ndarray
    achieved: np.ndarray                 # (N, 5), NaN where the shot never landed
    error: np.ndarray                    # (N, 5)
    goal_error: np.ndarray               # (N,)
    landing: np.ndarray                  # (N, 3)
    outcome: np.ndarray                  # (N,)
    plan_ms: float = 0.0

    @property
    def ok(self):
        return np.isfinite(self.goal_error)

    @property
    def success_rate(self):
        return float((self.outcome == 1).mean())

    @property
    def place_error(self):
        return np.hypot(self.error[:, 0], self.error[:, 1])

    def median(self, arr):
        m = self.ok
        return float(np.median(arr[m])) if m.any() else float("nan")


def evaluate(contenders, pos, vel, spin, targets):
    """Every contender over the same batch of problems.

    Batched rather than looped: the agents all take (N, ...) inputs, and this
    is the same call shape training.evaluate uses, so the medians here line
    up with the rows in results.csv.
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    spin = np.asarray(spin, dtype=float)
    targets = np.asarray(targets, dtype=np.float32)
    state = dataset.encode_state(pos, vel, spin).astype(np.float32)

    runs = []
    for c in contenders:
        t0 = perf_counter()
        if c.is_oracle:
            acts = c.agent.act(state, targets, pos, vel, spin)
        else:
            acts = c.agent.act(state, targets)
        ms = (perf_counter() - t0) * 1000.0 / max(1, len(pos))
        acts = np.asarray(acts, dtype=float)
        achieved, outcome, landing = dataset.apply_actions(pos, vel, spin, acts)
        err = achieved - targets
        runs.append(Run(contender=c, actions=acts, achieved=achieved, error=err,
                        goal_error=dataset.goal_error(achieved, targets),
                        landing=landing, outcome=outcome, plan_ms=ms))
    return runs


def table(shots, goal):
    """The comparison as plain text, for the terminal and for --list runs."""
    L = [f"{'method':<18}{'land err':>10}{'speed':>9}{'topspin':>10}"
         f"{'sidespin':>10}{'goal':>8}{'in?':>6}{'ms':>9}"]
    L.append("-" * len(L[0]))
    # Unmeasurable shots sort last whatever their NaN would have done
    for s in sorted(shots, key=lambda s: s.goal_error if s.measured else np.inf):
        if s.measured:
            body = (f"{s.place_error * 100:9.1f}cm"
                    f"{s.error[2]:+9.2f}"
                    f"{s.error[3]:+10.0f}"
                    f"{s.error[4]:+10.0f}"
                    f"{s.goal_error:8.3f}")
        else:
            body = f"{'--':>11}{'--':>9}{'--':>10}{'--':>10}{'--':>8}"
        L.append(f"{s.contender.label:<18}{body}"
                 f"{'IN' if s.landed_in else 'miss':>6}"
                 f"{s.plan_ms:9.2f}")
    return "\n".join(L)
