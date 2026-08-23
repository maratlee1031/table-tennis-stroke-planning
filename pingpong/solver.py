"""Inverse solve: which stroke sends this ball to that spot.

The forward model is already known -- it is the physics -- so the useful
thing to compute is its inverse. Cross-entropy method (CEM) does that by
sampling actions, keeping the best, and refitting the distribution.

Two entry points:

* :func:`solve_one` for a single problem, used live by the AI coach.
* :func:`solve_batch` for N independent problems **at once**, which is what
  makes expert training data affordable. Solving one problem at a time costs
  50-450 ms; generating 20k policy examples that way would run for hours.
  Batched, every problem keeps its own distribution but all N x K candidate
  strokes are simulated in a single vectorised call.
"""

import concurrent.futures as cf
import os

import numpy as np

from . import constants as C
from . import dataset, progress

# Same box the dataset samples from, so learned methods and the solver are
# choosing from an identical set of possible strokes.
LOW = dataset.ACTION_LOW
HIGH = dataset.ACTION_HIGH

INIT_MEAN = np.array([0.0, 6.0, 5.0, 0.0, 2.5])
INIT_STD = np.array([32.0, 22.0, 4.5, 3.5, 4.0])
FLOOR_STD = np.array([2.5, 2.5, 0.35, 0.35, 0.35])

MISS_PENALTY = 3.0        # cost added when the ball does not land in
EFFORT_WEIGHT = 0.01      # mild preference for the cheaper stroke


def _cost(achieved, outcome, goals, actions, weights=None):
    """Weighted goal error in normalised units, plus a miss penalty.

    Normalised because the goal mixes metres, m/s and rad/s; weighted because
    landing in the wrong place matters more than landing with slightly the
    wrong spin.

    ``weights`` defaults to dataset.GOAL_WEIGHTS. It is a parameter because
    that constant decides what the search gives up once placement is nearly
    exhausted -- the trade-off sweep_weights.py maps -- and a hardcoded
    global cannot be swept.
    """
    err = dataset.goal_error(achieved, goals, weights)
    err = np.where(np.isnan(err), 4.0, err)
    penalty = np.where(outcome == 1, 0.0, MISS_PENALTY)
    effort = EFFORT_WEIGHT * np.linalg.norm(actions[:, 2:5], axis=1)
    return err + penalty + effort


# Integration step for the candidate rollouts. Measured against re-verifying
# the chosen strokes on the 1/480 simulator:
#   1/480  10.9 problems/s  100.0% solved  10.1 cm median error
#   1/240  22.5             100.0%          9.9 cm
#   1/120  45.8              99.8%         10.0 cm   <- default
#   1/60   90.0              97.8%         10.1 cm
# Accuracy is flat down to 1/120 and only the solve rate starts to slip
# below that, so this is four times the throughput for nothing.
SOLVER_DT = 1.0 / 120.0


# Perturbation scale for the robustness pass, as a multiple of FLOOR_STD --
# the spread the search itself cannot resolve. A solution worth imitating
# should survive errors of at least that size.
ROBUST_SIGMA_MULT = 2.0
# How much a stroke's fragility counts against its accuracy, in the same
# units as the weighted goal error. Measured trade-off on 171 solved
# problems, executing each chosen stroke under simulated error:
#
#   weight  nominal goal   success at 1 sigma   at 2 sigma
#   0.00    0.203          65.0%                42.0%
#   0.25    0.220          70.8%                47.5%   <- default
#   0.50    0.300          75.4%                52.1%
#   1.00    0.374          82.7%                58.6%
#
# 0.25 buys most of the available margin for 8% of accuracy; past it the
# exchange rate turns sharply against you.
ROBUST_WEIGHT = 0.25


def _robust_rerank(pos, vel, spin, goals, elites, rng, sim_dt, k, sigma,
                   weights=None):
    """Pick, from each problem's elites, the stroke that survives being wrong.

    The plain cost asks only how close a stroke gets to the goal, never how
    close it is to failing. It therefore favours solutions pressed against
    the edge of feasibility: exact, but with no margin. Imitating those
    faithfully is what cost the mixture policy several points of success rate
    while its accuracy improved.

    Each elite is scored as its **nominal accuracy plus a penalty for how
    often it fails when executed imperfectly**, the two kept separate on
    purpose. Scoring by mean perturbed cost instead sounds equivalent and is
    not: the miss penalty inside the cost then dominates, the search
    optimises almost purely for not missing, and nominal goal error more than
    doubled (0.203 -> 0.462) while robustness improved. Separating them makes
    the trade-off explicit and tunable through ``weight``.

    Applied to the elites only, after the search has converged: perturbing
    the whole population every iteration would multiply generation cost by k
    for no extra information.
    """
    n, m, d = elites.shape
    noise = rng.normal(size=(n, m, k, d)) * sigma
    cand = np.clip(elites[:, :, None, :] + noise, LOW, HIGH)
    flat = cand.reshape(-1, d)

    rep = m * k
    _, outcome, _ = dataset.apply_actions(
        np.repeat(pos, rep, axis=0), np.repeat(vel, rep, axis=0),
        np.repeat(spin, rep, axis=0), flat, dt=sim_dt)
    # Fragility: share of perturbed executions that no longer land in
    fragility = (outcome != 1).reshape(n, m, k).mean(axis=2)

    # Nominal accuracy of each elite, unperturbed
    flat_e = elites.reshape(-1, d)
    ach_e, out_e, _ = dataset.apply_actions(
        np.repeat(pos, m, axis=0), np.repeat(vel, m, axis=0),
        np.repeat(spin, m, axis=0), flat_e, dt=sim_dt)
    nominal = _cost(ach_e, out_e, np.repeat(goals, m, axis=0), flat_e,
                    weights).reshape(n, m)
    return nominal, fragility


def solve_batch(pos, vel, spin, goals, iters=5, pop=64, elite_frac=0.15,
                rng=None, sim_dt=SOLVER_DT, robust_k=0, robust_sigma=None,
                robust_weight=ROBUST_WEIGHT, weights=None):
    """Solve N stroke problems simultaneously.

    Every problem carries its own CEM distribution; the candidates for all of
    them are simulated together. Returns (actions, cost, achieved, outcome).

    With ``robust_k`` > 0 the winner is chosen by :func:`_robust_rerank`
    instead of by nominal cost alone -- worth it for demonstrations that a
    policy will imitate, unnecessary for a one-off live solve.
    """
    rng = rng or np.random.default_rng()
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    spin = np.asarray(spin, dtype=float)
    goals = np.asarray(goals, dtype=float)
    n = pos.shape[0]
    n_elite = max(4, int(pop * elite_frac))

    mean = np.tile(INIT_MEAN, (n, 1))
    std = np.tile(INIT_STD, (n, 1))

    best_a = np.zeros((n, dataset.ACTION_DIM))
    best_c = np.full(n, np.inf)
    best_l = np.full((n, dataset.GOAL_DIM), np.nan)
    best_o = np.full(n, 4, dtype=np.int8)

    # Repeated once instead of per iteration
    rep_pos = np.repeat(pos, pop, axis=0)
    rep_vel = np.repeat(vel, pop, axis=0)
    rep_spin = np.repeat(spin, pop, axis=0)
    rep_tgt = np.repeat(goals, pop, axis=0)

    for _ in range(iters):
        cand = rng.normal(mean[:, None, :], std[:, None, :],
                          size=(n, pop, dataset.ACTION_DIM))
        cand = np.clip(cand, LOW, HIGH)
        flat = cand.reshape(-1, dataset.ACTION_DIM)

        achieved, outcome, _ = dataset.apply_actions(rep_pos, rep_vel, rep_spin,
                                                     flat, dt=sim_dt)
        cost = _cost(achieved, outcome, rep_tgt, flat, weights).reshape(n, pop)

        order = np.argsort(cost, axis=1)
        top = order[:, 0]
        rows = np.arange(n)
        improved = cost[rows, top] < best_c
        best_c = np.where(improved, cost[rows, top], best_c)
        best_a = np.where(improved[:, None], cand[rows, top], best_a)
        flat_top = rows * pop + top
        best_l = np.where(improved[:, None], achieved[flat_top], best_l)
        best_o = np.where(improved, outcome[flat_top], best_o)

        elite_idx = order[:, :n_elite]
        elites = np.take_along_axis(cand, elite_idx[:, :, None], axis=1)
        mean = elites.mean(axis=1)
        std = elites.std(axis=1) + FLOOR_STD

    if robust_k > 0:
        sigma = (FLOOR_STD * ROBUST_SIGMA_MULT if robust_sigma is None
                 else np.asarray(robust_sigma, dtype=float))
        # Include the incumbent best, so robustness can only replace it with
        # something it actually prefers
        pool = np.concatenate([best_a[:, None, :], elites], axis=1)
        nominal, fragility = _robust_rerank(pos, vel, spin, goals, pool, rng,
                                            sim_dt, robust_k, sigma, weights)
        pick = (nominal + robust_weight * fragility).argmin(axis=1)
        rows = np.arange(n)
        best_a = pool[rows, pick]
        # Report the nominal cost and outcome of the chosen stroke, so the
        # numbers stay comparable with a non-robust run
        achieved, outcome, _ = dataset.apply_actions(pos, vel, spin, best_a,
                                                     dt=sim_dt)
        best_c = _cost(achieved, outcome, goals, best_a, weights)
        best_l, best_o = achieved, outcome

    return best_a, best_c, best_l, best_o


def solve_one(pos, vel, spin, goal, **kw):
    """Single-problem convenience wrapper."""
    a, c, l, o = solve_batch(np.atleast_2d(pos), np.atleast_2d(vel),
                             np.atleast_2d(spin), np.atleast_2d(goal), **kw)
    return a[0], float(c[0]), l[0], int(o[0])


def _expert_chunk(seed, m, iters, pop, robust_k, robust_sigma, robust_weight):
    """One independent block of demonstrations. Top level so it can be pickled
    to a worker process."""
    from . import serve

    rng = np.random.default_rng(seed)
    while True:
        pos, vel, spin = serve.incoming_at(dataset.STRIKE_X, m, rng)
        goals, feasible = dataset.achievable_goals(pos, vel, spin, rng)
        if feasible.any():
            break
    pos, vel, spin = pos[feasible], vel[feasible], spin[feasible]
    goals = goals[feasible]

    actions, cost, _, outcome = solve_batch(
        pos, vel, spin, goals, iters=iters, pop=pop, rng=rng,
        robust_k=robust_k, robust_sigma=robust_sigma,
        robust_weight=robust_weight)

    return (dataset.encode_state(pos, vel, spin), goals, actions, cost,
            outcome == 1)


def expert_dataset(n, rng=None, chunk=None, iters=5, pop=64, verbose=False,
                   robust_k=0, robust_sigma=None,
                   robust_weight=ROBUST_WEIGHT, workers=None):
    """Expert ``(state, goal) -> action`` demonstrations from the solver.

    Uniformly sampled strokes land on the opponent's half only about 0.7% of
    the time, so a random dataset yields almost no usable policy pairs. These
    are solved instead, which is what a policy network can actually learn to
    imitate.

    Goals are drawn from what is demonstrably achievable rather than invented,
    so the demonstrations are of solvable problems.

    Chunks are independent, and the work inside one is elementwise numpy that
    never leaves a single core, so ``workers`` spreads them over processes.
    That is what makes a high solver budget affordable -- demonstration
    quality is the ceiling on every policy trained from them:

        budget    robust_k   1 worker   12 workers   20k solves
        5x64         0       45.8/s                   0.12 h
        8x256        8        6.9/s      21.5/s       0.26 h
        10x512       8        1.5/s       7.1/s       0.78 h
        12x1024      8        0.6/s       2.4/s       2.30 h

    Scaling is ~4-5x rather than 12x -- the rollouts are memory-bandwidth
    bound, not compute bound -- but it is the difference between a two-hour
    run and an overnight one.
    """
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    if chunk is None:
        # Two competing pressures: chunks must outnumber workers or cores
        # sit idle, and chunk x pop candidate rollouts are simulated in one
        # array per worker, so a large pop needs a smaller chunk to stay
        # within memory.
        chunk = int(np.clip(min(250_000 // max(pop, 1),
                                -(-n // workers)), 100, 2000))

    rng = rng or np.random.default_rng()
    # One independent seed per chunk, drawn from the caller's stream so the
    # whole run stays reproducible from a single seed
    sizes = [int(min(chunk, n - i)) for i in range(0, n, chunk)]
    seeds = rng.integers(0, 2**63 - 1, size=len(sizes))
    job = (iters, pop, robust_k, robust_sigma, robust_weight)

    parts = []
    have = 0
    pbar = progress.bar(n, "expert demonstrations", "solves", enabled=verbose)

    def collect(res):
        nonlocal have
        parts.append(res)
        have += len(res[1])
        pbar.update(len(res[1]))
        ok = np.concatenate([p[4] for p in parts])
        pbar.set_postfix_str(f", {ok.mean() * 100:.1f}% solved")

    if workers <= 1 or len(sizes) == 1:
        for s, m in zip(seeds, sizes):
            collect(_expert_chunk(s, m, *job))
    else:
        with cf.ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futures = [ex.submit(_expert_chunk, int(s), m, *job)
                       for s, m in zip(seeds, sizes)]
            for f in cf.as_completed(futures):
                collect(f.result())
    pbar.close()

    return {
        "state": np.concatenate([p[0] for p in parts]).astype(np.float32),
        "goal": np.concatenate([p[1] for p in parts]).astype(np.float32),
        "action": np.concatenate([p[2] for p in parts]).astype(np.float32),
        "cost": np.concatenate([p[3] for p in parts]).astype(np.float32),
        "success": np.concatenate([p[4] for p in parts]),
    }
