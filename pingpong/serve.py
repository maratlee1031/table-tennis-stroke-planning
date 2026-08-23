"""Serve generation, shared by the game and by dataset generation.

Kept out of main_wss.py so that producing training data never has to import
Panda3D -- data generation runs headless and in batch.
"""

import numpy as np

from . import constants as C
from . import physics

# Where the server stands and where the ball is aimed. These match the game;
# training on a different serve distribution than the one you play against
# would quietly wreck the comparison.
SERVE_X = (0.75, 1.15)
SERVE_Y = (-0.45, 0.45)
SERVE_Z = (0.16, 0.30)          # above the table
TARGET_X = (-1.05, -0.42)
TARGET_Y = (-0.34, 0.34)
SPIN_X = (-40.0, 40.0)
SPIN_Y = (-120.0, 90.0)
SPIN_Z = (-60.0, 60.0)
FLIGHT = (0.32, 0.46)           # short enough to stay flat; see main_wss


def solve_serve(start, target, flight_time, spin=None):
    """Launch velocity that puts the ball on ``target`` after ``flight_time``.

    Starts from the drag-free ballistic solution and iterates against the real
    physics, because drag on a table tennis ball exceeds gravity and the
    drag-free answer is well off.
    """
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    spin = np.zeros(3) if spin is None else np.asarray(spin, dtype=float)

    d = target - start
    v = np.array([
        d[0] / flight_time,
        d[1] / flight_time,
        (d[2] + 0.5 * C.GRAVITY * flight_time ** 2) / flight_time,
    ])

    for _ in range(6):
        res = physics.simulate(start, v, spin, max_time=flight_time * 3 + 1.0)
        if res.landing is None:
            v[2] += 0.8            # never landed: loft it and shorten
            v[0] *= 0.92
            continue
        err = target - res.landing
        if np.linalg.norm(err[:2]) < 0.01:
            break
        v[0] += err[0] / flight_time * 0.9
        v[1] += err[1] / flight_time * 0.9
        v[2] += err[2] / flight_time * 0.5
    return v


def random_serve(rng=None):
    """One random serve: (start, velocity, spin)."""
    rng = rng or np.random.default_rng()
    start = np.array([
        rng.uniform(*SERVE_X),
        rng.uniform(*SERVE_Y),
        C.TABLE_H + rng.uniform(*SERVE_Z),
    ])
    target = np.array([
        rng.uniform(*TARGET_X),
        rng.uniform(*TARGET_Y),
        C.TABLE_TOP_Z + C.BALL_RADIUS,
    ])
    spin = np.array([
        rng.uniform(*SPIN_X), rng.uniform(*SPIN_Y), rng.uniform(*SPIN_Z),
    ])
    vel = solve_serve(start, target, rng.uniform(*FLIGHT), spin)
    return start, vel, spin


# Drag makes the ball fall short of the drag-free prediction, so the cheap
# ballistic sampler below scales its horizontal velocity up to compensate.
# Calibrated against solve_serve, whose median solved vx was -4.80.
DRAG_COMPENSATION = 1.18

# A serve is only usable as training input if the ball is actually reachable
# when it arrives. Anything outside this is a ball the player could never
# have played anyway.
REACHABLE_Z = (C.TABLE_H - 0.02, C.TABLE_H + 0.60)
REACHABLE_Y = 0.85


def random_serve_batch(n, rng=None):
    """``n`` serves, sampled with correlated velocities.

    Sampling vx, vy and vz from independent marginal ranges looks equivalent
    but is not: in a real serve they are tied together by where the ball is
    aimed, and independent draws produce serves that sail wide or long. Here
    the velocity comes from the drag-free ballistic solution for a sampled
    target, which keeps that correlation, at a fraction of the cost of the
    full inverse solve.
    """
    rng = rng or np.random.default_rng()
    start = np.stack([
        rng.uniform(*SERVE_X, n),
        rng.uniform(*SERVE_Y, n),
        C.TABLE_H + rng.uniform(*SERVE_Z, n),
    ], axis=1)
    target = np.stack([
        rng.uniform(*TARGET_X, n),
        rng.uniform(*TARGET_Y, n),
        np.full(n, C.TABLE_TOP_Z + C.BALL_RADIUS),
    ], axis=1)
    flight = rng.uniform(*FLIGHT, n)

    d = target - start
    vel = np.stack([
        d[:, 0] / flight * DRAG_COMPENSATION,
        d[:, 1] / flight * DRAG_COMPENSATION,
        (d[:, 2] + 0.5 * C.GRAVITY * flight ** 2) / flight,
    ], axis=1)
    spin = np.stack([
        rng.uniform(*SPIN_X, n),
        rng.uniform(*SPIN_Y, n),
        rng.uniform(*SPIN_Z, n),
    ], axis=1)
    return start, vel, spin


def incoming_at(x_plane, n, rng=None, oversample=3.0):
    """``n`` realistic ball states arriving at ``x_plane``.

    Serves are rolled and propagated to the strike plane, so the training
    distribution is the one the player actually faces rather than an invented
    box of numbers. Arrivals are kept only if the serve bounced on our half
    and the ball is somewhere a paddle could reach -- that filter is what
    makes the distribution correct by construction rather than by hoping the
    sampler was tuned right.

    Returns (pos, vel, spin), each (n, 3).
    """
    rng = rng or np.random.default_rng()
    got_p, got_v, got_w = [], [], []
    have = 0
    # Chunked: one huge propagation batch thrashes memory and ends up slower
    # per sample than several moderate ones.
    for _ in range(400):
        m = int(np.clip((n - have) * oversample, 512, 30000))
        s, v, w = random_serve_batch(m, rng)
        p2, v2, w2, _, ok, bounced = physics.propagate_to_plane_batch(
            s, v, w, x_plane)
        keep = (ok & bounced
                & (p2[:, 2] > REACHABLE_Z[0]) & (p2[:, 2] < REACHABLE_Z[1])
                & (np.abs(p2[:, 1]) < REACHABLE_Y))
        if keep.any():
            got_p.append(p2[keep]); got_v.append(v2[keep]); got_w.append(w2[keep])
            have += int(keep.sum())
        if have >= n:
            break
    if not got_p:
        raise RuntimeError("no serve produced a reachable arrival")
    return (np.concatenate(got_p)[:n],
            np.concatenate(got_v)[:n],
            np.concatenate(got_w)[:n])
