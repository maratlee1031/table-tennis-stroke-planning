"""Training data: (incoming ball, stroke) -> where it lands.

Generated headlessly in batch. Nothing here touches Panda3D and nothing
depends on a human having played -- the physics *is* the forward model, so
the data can be manufactured as fast as the simulator runs.

One dataset feeds both kinds of model:

* **Forward surrogate** learns ``(state, action) -> (landing, success)``.
  Single-valued, so ordinary regression works.
* **Direct policy** learns ``(state, target) -> action`` by *hindsight
  relabelling*: for every stroke that landed in, the place it actually
  landed is treated as the target it was aiming at. Successful samples
  become supervised policy pairs for free.

Representation
--------------
State (8): the ball at the strike plane. Its x is fixed there by definition,
so only y and z carry information.

    y, z, vx, vy, vz, wx, wy, wz

Action (5): what the paddle does.

    normal yaw (deg), normal pitch (deg), paddle vx, vy, vz

Target / landing (2): where it comes down on the opponent's half.

    x, y
"""

import numpy as np

from . import constants as C
from . import physics, progress, serve

STRIKE_X = -1.12          # matches main_wss.PADDLE_STRIKE_X

STATE_DIM = 8
ACTION_DIM = 5

# The shot is specified by five properties, not just where it lands: the
# action space is also five-dimensional, so this is exactly determined at
# best. Not every combination is reachable, which is part of what the
# experiments measure.
GOAL_DIM = 5
TARGET_DIM = GOAL_DIM          # kept as an alias; models size off this

STATE_NAMES = ["y", "z", "vx", "vy", "vz", "wx", "wy", "wz"]
ACTION_NAMES = ["yaw_deg", "pitch_deg", "pvx", "pvy", "pvz"]
GOAL_NAMES = ["land_x", "land_y", "speed", "topspin", "sidespin"]
GOAL_UNITS = ["m", "m", "m/s", "rad/s", "rad/s"]

# Action sampling ranges. Same box the CEM solver searches, so every method
# in the comparison is choosing from the same set of possible strokes.
ACTION_LOW = np.array([-75.0, -50.0, -6.0, -12.0, -12.0])
ACTION_HIGH = np.array([75.0, 50.0, 16.0, 12.0, 12.0])

# Scales used to normalise inputs before they reach a network. Rough
# magnitudes, not statistics -- a fixed scale keeps a model trained on one
# dataset valid on another.
STATE_SCALE = np.array([0.8, 1.4, 8.0, 3.0, 4.0, 200.0, 200.0, 200.0])
ACTION_SCALE = np.array([75.0, 50.0, 16.0, 12.0, 12.0])
# Goal dims live in wildly different units (metres, m/s, rad/s), so every
# error and every cost is computed in these normalised units. Per-dimension
# results are still reported in natural units.
GOAL_SCALE = np.array([1.4, 0.8, 9.0, 300.0, 300.0])
TARGET_SCALE = GOAL_SCALE

# Relative importance when scoring a shot. Landing dominates -- put the ball
# in the wrong place and the speed and spin you achieved do not matter --
# but speed and spin still carry real weight, which is the point of the
# richer goal.
GOAL_WEIGHTS = np.array([1.0, 1.0, 0.6, 0.5, 0.5])


def encode_state(pos, vel, spin):
    """(pos, vel, spin) arrays -> the (N, 8) state used by the models."""
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)
    spin = np.atleast_2d(spin)
    return np.concatenate([pos[:, 1:3], vel, spin], axis=1)


def decode_action(actions):
    """(N, 5) actions -> (normals, paddle velocities)."""
    actions = np.atleast_2d(actions)
    yaw = np.radians(actions[:, 0])
    pitch = np.radians(actions[:, 1])
    cp = np.cos(pitch)
    normals = np.stack([cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)], axis=1)
    return normals, actions[:, 2:5]


# A returned ball is on the table well inside 2 s; the 5 s default just makes
# the batch loop keep running for the few stragglers.
SIM_MAX_TIME = 2.0


def encode_goal(landing, land_vel, land_spin):
    """Pack what a shot achieved into the 5-dim goal vector.

    ``land_x, land_y, speed, topspin, sidespin`` -- everything the receiving
    player experiences. Spin about y is topspin/backspin (signed), spin about
    z is sidespin.
    """
    speed = np.linalg.norm(land_vel, axis=-1)
    return np.stack([landing[:, 0], landing[:, 1], speed,
                     land_spin[:, 1], land_spin[:, 2]], axis=1)


def goal_error(a, b, weights=None):
    """Weighted distance between two goal vectors, in normalised units."""
    w = GOAL_WEIGHTS if weights is None else np.asarray(weights)
    d = (np.asarray(a) - np.asarray(b)) / GOAL_SCALE * w
    return np.linalg.norm(d, axis=-1)


def apply_actions(pos, vel, spin, actions, dt=C.DEFAULT_DT,
                  max_time=SIM_MAX_TIME):
    """Play the strokes and report what each shot achieved.

    Returns (goal (N,5), outcome (N,), landing (N,3)) with
    physics.simulate_batch's outcome codes: 0 own half, 1 opponent half,
    2 out, 3 net, 4 timeout.
    """
    normals, pvel = decode_action(actions)
    out_v, out_w = physics.collide(
        vel, spin, normals, C.RESTITUTION_PADDLE, C.FRICTION_PADDLE,
        surface_vel=pvel,
    )
    start = pos + normals * (C.BALL_RADIUS * 1.6)
    landing, outcome, lv, lw = physics.simulate_batch(
        start, out_v, out_w, dt=dt, max_time=max_time)
    return encode_goal(landing, lv, lw), outcome, landing


def generate(n, rng=None, chunk=20000, verbose=False):
    """Build ``n`` samples of (state, action) -> (landing, success).

    Actions are sampled uniformly from the action box, so the dataset covers
    the whole space rather than only the strokes a good player would pick.
    That matters: a forward model has to be accurate about bad strokes too,
    or the optimiser that inverts it will happily walk into a region the
    model has never seen and be confidently wrong.
    """
    rng = rng or np.random.default_rng()
    S, A, L, OK, HAS = [], [], [], [], []
    have = 0
    pbar = progress.bar(n, "random actions", "samples", enabled=verbose)
    while have < n:
        m = int(min(chunk, n - have))
        pos, vel, spin = serve.incoming_at(STRIKE_X, m, rng)
        actions = rng.uniform(ACTION_LOW, ACTION_HIGH, size=(m, ACTION_DIM))
        goal, outcome, landing = apply_actions(pos, vel, spin, actions)

        S.append(encode_state(pos, vel, spin))
        A.append(actions)
        L.append(goal)
        OK.append(outcome == 1)
        # A ball that sails past the end of the table still has an outcome,
        # and the forward model badly needs those: an optimiser inverting it
        # has to be told "that stroke lands two metres long", not merely
        # "that stroke failed".
        HAS.append(~np.isnan(landing[:, 0]))
        have += m
        pbar.update(m)
        pbar.set_postfix_str(f", {np.mean(np.concatenate(OK)) * 100:.1f}% in")
    pbar.close()

    return {
        "state": np.concatenate(S).astype(np.float32),
        "action": np.concatenate(A).astype(np.float32),
        "goal": np.nan_to_num(np.concatenate(L)).astype(np.float32),
        "success": np.concatenate(OK),
        "has_landing": np.concatenate(HAS),
    }


def generate_mixed(n, rng=None, expert_frac=0.35,
                   perturb=(0.0, 0.12, 0.3, 0.6, 1.0), chunk=8000, verbose=False):
    """Like :func:`generate`, but with the good region of action space covered.

    Why this exists: uniformly sampled strokes land on the opponent's half
    about 0.7% of the time, so a forward model trained on them alone sees
    almost nothing near a good stroke. Measured on such a model, the landing
    prediction error was 22 cm on random actions -- its own training
    distribution -- and **145 cm on solved actions**, while it put the
    probability of landing in at 0.02 for strokes that in fact succeeded
    99.6% of the time.

    That is covariate shift, and it is fatal for anything that inverts the
    model: the optimiser walks straight into the region the model has never
    seen and is confidently wrong there.

    The fix is to solve a smaller number of problems properly and include
    those actions **and their neighbourhood** (hence ``perturb``, several
    noise scales around each solution). The neighbourhood matters as much as
    the solution: an optimiser needs a correct gradient around the optimum,
    not just a correct value at it.
    """
    from . import serve, solver

    rng = rng or np.random.default_rng()
    n_expert = int(n * expert_frac)
    n_uniform = n - n_expert

    parts = []
    if n_uniform > 0:
        if verbose:
            print(f"  uniform part: {n_uniform:,}")
        parts.append(generate(n_uniform, rng, verbose=verbose))

    if n_expert > 0:
        per_solve = max(1, len(perturb))
        n_solve = int(np.ceil(n_expert / per_solve))
        if verbose:
            print(f"  expert part: {n_expert:,} samples from {n_solve:,} CEM solves"
                  f" x {per_solve} perturbations")
        S, A, L, OK, HAS = [], [], [], [], []
        done = 0
        pbar = progress.bar(n_solve, "expert-region solves", "solves",
                            enabled=verbose)
        while done < n_solve:
            m = int(min(chunk, n_solve - done))
            pos, vel, spin = serve.incoming_at(STRIKE_X, m, rng)
            tg, feasible = achievable_goals(pos, vel, spin, rng)
            if not feasible.any():
                continue
            pos, vel, spin, tg = (pos[feasible], vel[feasible],
                                  spin[feasible], tg[feasible])
            base, _, _, _ = solver.solve_batch(pos, vel, spin, tg, rng=rng)

            for scale in perturb:
                a = base + scale * solver.FLOOR_STD * 4.0 * rng.normal(size=base.shape)
                a = np.clip(a, ACTION_LOW, ACTION_HIGH)
                goal, outcome, landing = apply_actions(pos, vel, spin, a)
                S.append(encode_state(pos, vel, spin))
                A.append(a)
                L.append(goal)
                OK.append(outcome == 1)
                HAS.append(~np.isnan(landing[:, 0]))
            done += len(tg)
            pbar.update(len(tg))
        pbar.close()

        parts.append({
            "state": np.concatenate(S).astype(np.float32),
            "action": np.concatenate(A).astype(np.float32),
            "goal": np.nan_to_num(np.concatenate(L)).astype(np.float32),
            "success": np.concatenate(OK),
            "has_landing": np.concatenate(HAS),
        })

    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = rng.permutation(len(out["state"]))
    out = {k: v[order][:n] for k, v in out.items()}
    if verbose:
        print(f"  mixed dataset: {len(out['state']):,} samples, "
              f"{out['success'].mean() * 100:.1f}% land in")
    return out


def policy_pairs(data):
    """Hindsight-relabelled ``(state, goal) -> action`` pairs.

    Only successful strokes are usable: a stroke that went out was not
    'aiming' at anything on the table.
    """
    ok = data["success"]
    return (data["state"][ok], data["goal"][ok], data["action"][ok])


def save(path, data):
    np.savez_compressed(path, **data)


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def achievable_goals(pos, vel, spin, rng, tries=48, chunk=200000):
    """Goals known to be reachable, found by playing strokes and watching.

    Asking for an arbitrary combination of place, speed and spin is usually
    impossible -- the action space has five dimensions and so does the goal,
    so at best it is exactly determined. Sampling goals out of thin air would
    mean scoring every method on tasks that have no solution, which measures
    nothing.

    Instead a random stroke is played and whatever it produced becomes the
    goal. Every problem then has at least one solution by construction; the
    method still has to find one, and it will usually not be the same stroke.
    """
    n = len(pos)
    goals = np.full((n, GOAL_DIM), np.nan)
    todo = np.ones(n, dtype=bool)
    for _ in range(tries):
        idx = np.flatnonzero(todo)
        if idx.size == 0:
            break
        a = rng.uniform(ACTION_LOW, ACTION_HIGH, size=(idx.size, ACTION_DIM))
        g, outcome, _ = apply_actions(pos[idx], vel[idx], spin[idx], a)
        good = outcome == 1
        if good.any():
            goals[idx[good]] = g[good]
            todo[idx[good]] = False
    return goals, ~todo


def make_eval_set(n, rng=None):
    """A fixed set of (incoming ball, desired shot) problems.

    Every method is scored on exactly these, so the comparison is not
    measuring who got the luckier sample.
    """
    rng = rng or np.random.default_rng(20260805)
    keep_p, keep_v, keep_w, keep_g = [], [], [], []
    have = 0
    for _ in range(20):
        m = max(256, int((n - have) * 1.4))
        pos, vel, spin = serve.incoming_at(STRIKE_X, m, rng)
        goals, ok = achievable_goals(pos, vel, spin, rng)
        if ok.any():
            keep_p.append(pos[ok]); keep_v.append(vel[ok])
            keep_w.append(spin[ok]); keep_g.append(goals[ok])
            have += int(ok.sum())
        if have >= n:
            break
    return (np.concatenate(keep_p)[:n], np.concatenate(keep_v)[:n],
            np.concatenate(keep_w)[:n],
            np.concatenate(keep_g)[:n].astype(np.float32))
