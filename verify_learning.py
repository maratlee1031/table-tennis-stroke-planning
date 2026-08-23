"""Verification suite for the learning half: encoding, agents, comparison.

    python verify_learning.py

Companion to verify_physics.py, same style and no Panda3D. That suite covers
the simulator; this one covers everything built on top of it -- the state /
action / goal encoding, the scoring, the agents, and the comparison layer
that ai_play.py and compare_models.py both read.

Checks that need a trained model are skipped rather than failed when the
checkpoint is absent, so this runs on a fresh clone before any training.
"""

import os
import time

import numpy as np

from pingpong import agents, compare, constants as C, dataset, physics, serve
from pingpong import stroke_card

PASS, FAIL, SKIP = "  [OK]  ", "  [FAIL]", "  [skip]"
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"{PASS if ok else FAIL} {name}")
    if detail:
        print(f"          {detail}")


def skip(name, why):
    print(f"{SKIP} {name}")
    print(f"          {why}")


def section(title):
    print(f"\n=== {title} ===")


rng = np.random.default_rng(0)

# A batch of plausible arrivals at the strike plane, reused throughout
N = 64
POS, VEL, SPIN = serve.incoming_at(dataset.STRIKE_X, N, np.random.default_rng(1))


# ------------------------------------------------------------------ encoding
section("State / action / goal encoding")

st = dataset.encode_state(POS, VEL, SPIN)
check(
    "encode_state drops the strike-plane x and keeps the other eight numbers",
    st.shape == (N, 8) and np.allclose(st[:, 0], POS[:, 1]),
    f"shape {st.shape}, names {dataset.STATE_NAMES}; x is fixed at the strike "
    f"plane so carrying it would be a constant input",
)

acts = rng.uniform(dataset.ACTION_LOW, dataset.ACTION_HIGH, size=(N, 5))
normals, pvel = dataset.decode_action(acts)
check(
    "decode_action returns unit blade normals",
    np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9),
    f"max deviation {np.abs(np.linalg.norm(normals, axis=1) - 1).max():.2e}",
)
check(
    "decode_action passes the paddle velocity through untouched",
    np.allclose(pvel, acts[:, 2:5]),
)
yaw = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
pitch = np.degrees(np.arcsin(np.clip(normals[:, 2], -1, 1)))
check(
    "yaw and pitch round-trip out of the decoded normal",
    np.allclose(yaw, acts[:, 0], atol=1e-6) and np.allclose(pitch, acts[:, 1], atol=1e-6),
    "the action is recoverable from the blade normal, so the encoding loses nothing",
)

# ------------------------------------------------------------------ scoring
section("Goal scoring")

g = rng.normal(size=(N, 5)) * dataset.GOAL_SCALE
check(
    "goal_error of a goal against itself is zero",
    np.allclose(dataset.goal_error(g, g), 0.0),
)
one = np.zeros((1, 5))
for i, name in enumerate(dataset.GOAL_NAMES):
    off = np.zeros((1, 5))
    off[0, i] = dataset.GOAL_SCALE[i]          # exactly one scale unit out
    got = float(dataset.goal_error(off, one)[0])
    check(
        f"one scale unit of {name} costs its weight ({dataset.GOAL_WEIGHTS[i]})",
        abs(got - dataset.GOAL_WEIGHTS[i]) < 1e-6,
        f"got {got:.4f}; this is why placement dominates spin in every result",
    )

# ------------------------------------------------------------------ physics tie-in
section("apply_actions agrees with the simulator it is built on")

# Uniform strokes land about 0.7% of the time, so finding one that reaches
# the far half at all needs thousands of samples, not dozens
M = 8000
p1, v1, s1 = (np.repeat(POS[:1], M, 0), np.repeat(VEL[:1], M, 0),
              np.repeat(SPIN[:1], M, 0))
probe = rng.uniform(dataset.ACTION_LOW, dataset.ACTION_HIGH, size=(M, 5))
achieved, outcome, landing = dataset.apply_actions(p1, v1, s1, probe)
land_rate = float((outcome == 1).mean())
hits = np.flatnonzero(outcome == 1)

if hits.size == 0:
    skip("a single simulate() reproduces the batched landing",
         "no uniform stroke landed in this sample")
else:
    i = int(hits[0])
    n_i, u_i = dataset.decode_action(probe[i:i + 1])
    out_v, out_w = physics.collide(
        v1[i:i + 1], s1[i:i + 1], n_i,
        C.RESTITUTION_PADDLE, C.FRICTION_PADDLE, surface_vel=u_i)
    r = physics.simulate(p1[i] + n_i[0] * (C.BALL_RADIUS * 1.6), out_v[0], out_w[0])
    single = "no landing" if r.landing is None else np.round(r.landing[:2], 4)
    check(
        "a single simulate() reproduces the batched landing",
        r.landing is not None and np.allclose(r.landing[:2], landing[i][:2], atol=2e-3),
        f"batch {landing[i][:2].round(4)} vs single {single}; the live viewer "
        f"draws the single path beside batched numbers",
    )

check(
    "a shot into the net has no measurable result, and says so with NaN",
    bool(np.all(np.isnan(achieved[outcome == 3][:, 0]))) if (outcome == 3).any()
    else True,
    "zeros would quietly average into every median",
)
# What "measurable" means is narrower than "succeeded", and the two are
# easy to confuse when reading a median.
own = achieved[outcome == 0]
out_of_play = achieved[outcome == 2]
check(
    "a ball that lands on our own half is still measured",
    bool(np.all(np.isfinite(own[:, 0]))) if own.size else True,
    "it did land, just on the wrong side -- outcome, not measurability, "
    "is what says the shot failed",
)
frac = float(np.isfinite(out_of_play[:, 0]).mean()) if out_of_play.size else 0.0
check(
    "most shots that go out have nothing to measure",
    frac < 0.25,
    f"{frac * 100:.1f}% of out-of-play shots still record a landing (they "
    f"touched down past the edge). So goal_err_median is taken over shots "
    f"that landed *somewhere*, not over successful ones; success_rate is the "
    f"statistic that means 'on the table'",
)

# ------------------------------------------------------------------ achievability
section("Achievable goals")

goals, ok = dataset.achievable_goals(POS[:16], VEL[:16], SPIN[:16],
                                     np.random.default_rng(3), tries=64)
check(
    "achievable_goals only returns goals it actually produced",
    ok.any() and np.all(np.isfinite(goals[ok])),
    f"{ok.sum()}/16 solved; the rest stay NaN rather than being invented",
)
check(
    "uniform strokes land only rarely, which is why generate_mixed exists",
    land_rate < 0.05,
    f"{land_rate * 100:.2f}% of {M:,} land in; the README quotes about 0.7%",
)

# ------------------------------------------------------------------ kNN
section("Nearest neighbour")

demo_path = os.path.join("runs", "data", "expert_2000_i8p256.npz")
if not os.path.exists(demo_path):
    skip("kNN matches a brute-force search", f"{demo_path} not present")
else:
    demos = dataset.load(demo_path)
    ag = agents.KNNAgent(demos, k=5)
    q = np.concatenate([st[:8] / dataset.STATE_SCALE,
                        g[:8] / dataset.GOAL_SCALE], axis=1).astype(np.float32)
    # The reference: the (chunk, N, dim) broadcast the agent used to use,
    # before it was rewritten as a matrix product to survive 200k demos
    brute = np.empty((8, 5), np.float32)
    for j in range(8):
        d = ((q[j] - ag.q) ** 2).sum(-1)
        brute[j] = ag.a[np.argpartition(d, 5)[:5]].mean(axis=0)
    got = ag.act(st[:8], g[:8])
    check(
        "kNN matches a brute-force search exactly",
        np.allclose(got, brute, atol=1e-5),
        f"max difference {np.abs(got - brute).max():.2e}; the agent uses "
        f"|a-b|^2 = |a|^2 + |b|^2 - 2ab, which is 13x smaller in memory",
    )

# ------------------------------------------------------------------ comparison
section("Comparison layer")

check(
    "label_for keeps the demonstration budget in the name",
    compare.label_for("mdn_20000_i8p256") == "MDN 20k hq"
    and compare.label_for("policy_500") == "MLP 500",
    f"{compare.label_for('mdn_20000_i8p256')!r}; two checkpoints differing "
    f"only in budget are different experiments and must not share a label",
)
check(
    "presets expand and unknown specs pass through",
    len(compare.expand(["trainsize"])) > 1
    and compare.expand(["oracle"]) == ["oracle"],
)
check(
    "a missing checkpoint is skipped, not fatal",
    compare.build(["definitely_not_a_model"], quiet=True) == [],
    "half a ladder is still a useful comparison while a sweep is running",
)
check(
    "contenders get distinct palette slots",
    len({c.slot for c in compare.build(["oracle", "oracle:4,64"], quiet=True)}) == 2,
)

specs = [s for s in ("mdn_20000_i8p256", "policy_20000_i8p256")
         if os.path.exists(os.path.join("runs", "checkpoints", s + ".pt"))]
if not specs:
    skip("play() and evaluate() agree", "no trained checkpoints present")
else:
    cs = compare.build(specs, quiet=True)
    goal = np.array([0.9, 0.2, 6.0, 150.0, 20.0], np.float32)
    shots = compare.play(cs, POS[0], VEL[0], SPIN[0], goal)
    runs = compare.evaluate(cs, POS[:1], VEL[:1], SPIN[:1], goal[None, :])
    check(
        "play() and evaluate() give the same answer for one problem",
        all(np.allclose(s.achieved, r.achieved[0], equal_nan=True)
            for s, r in zip(shots, runs)),
        "the live overlay and the figures must never disagree",
    )
    cs[0].forced_action = np.zeros(5)
    forced = compare.play(cs[:1], POS[0], VEL[0], SPIN[0], goal, trajectory=False)
    cs[0].forced_action = None
    check(
        "a forced action is scored instead of a fresh plan",
        np.allclose(forced[0].action, 0.0),
        "the table has to describe the ball actually played; the oracle and "
        "the surrogate search both advance an RNG, so a re-plan would differ",
    )
    s0 = shots[0]
    check(
        "measured is False exactly when there is nothing to measure",
        s0.measured == bool(np.isfinite(s0.goal_error)),
    )
    card = stroke_card.describe(
        dataset.encode_state(POS[:1], VEL[:1], SPIN[:1]).astype(np.float32)[0],
        s0.action, goal=goal, verify=True)
    check(
        "the stroke card's achieved column equals requested plus error",
        (not np.isfinite(card["result"]["land_x"]))
        or abs(card["result"]["goal"][0] + card["result"]["error_per_dim"][0]
               - card["result"]["land_x"]) < 1e-6,
        "printing only two of the three columns is what made the card read "
        "as though the model had done the opposite of what it did",
    )

# ------------------------------------------------------------------ throughput
section("Throughput")

t0 = time.perf_counter()
dataset.apply_actions(np.repeat(POS[:1], 3000, 0), np.repeat(VEL[:1], 3000, 0),
                      np.repeat(SPIN[:1], 3000, 0),
                      rng.uniform(dataset.ACTION_LOW, dataset.ACTION_HIGH, (3000, 5)))
rate = 3000 / (time.perf_counter() - t0)
check(
    "batch simulation clears 1,000 trajectories per second",
    rate > 1000,
    f"{rate:,.0f}/s; the whole sweep is built on this being fast",
)


# ------------------------------------------------------------------ result
section("Result")
total, passed = len(_results), sum(_results)
print(f"\n{passed}/{total} checks passed")
if passed != total:
    raise SystemExit(1)
print("All passed.")
