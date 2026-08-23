"""Ask a trained model for a stroke, and print it in physical terms.

    python plan_stroke.py --land 0.9 0.3 --speed 6 --topspin 200

Prints a stroke card: blade orientation, paddle velocity at contact split
into drive and brush, contact impulses, and -- the number that decides
whether a machine can actually reproduce it -- the rubber friction the
stroke demands.

Options:

    --model runs/checkpoints/mdn_20000.pt   which trained model to ask
    --oracle                                use the CEM solver instead
    --incoming y z vx vy vz wx wy wz        specify the incoming ball
    --random N                              N random incoming balls
    --json out.json                         also dump machine-readable output
"""

import argparse
import json
import os

import numpy as np

from pingpong import agents, dataset, models, serve, stroke_card


def default_incoming(rng):
    pos, vel, spin = serve.incoming_at(dataset.STRIKE_X, 1, rng)
    return dataset.encode_state(pos, vel, spin)[0]


def build_agent(args):
    if args.oracle or not args.model:
        return agents.CEMOracle(iters=6, pop=128), "CEM on true physics"
    if not os.path.exists(args.model):
        raise SystemExit(f"{args.model} not found -- train first, or pass --oracle")
    model, meta = models.load(args.model)
    kind = getattr(model, "kind", "?")
    if kind == "mdn":
        return agents.MDNAgent(model), f"MDN policy ({args.model})"
    if kind == "policy":
        return agents.PolicyAgent(model), f"direct policy ({args.model})"
    if kind == "forward":
        return agents.SurrogateCEMAgent(model), f"learned forward + CEM ({args.model})"
    raise SystemExit(f"unsupported model kind: {kind}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--land", type=float, nargs=2, default=[0.9, 0.25],
                    metavar=("X", "Y"), help="where it should land, metres")
    ap.add_argument("--speed", type=float, default=5.0,
                    help="ball speed on arrival, m/s")
    ap.add_argument("--topspin", type=float, default=150.0,
                    help="topspin (+) or backspin (-) on arrival, rad/s")
    ap.add_argument("--sidespin", type=float, default=0.0, help="rad/s")
    ap.add_argument("--incoming", type=float, nargs=8, default=None,
                    metavar="V", help="y z vx vy vz wx wy wz")
    ap.add_argument("--random", type=int, default=0,
                    help="plan for N random incoming balls instead")
    ap.add_argument("--model", default="runs/checkpoints/mdn_20000.pt")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    agent, label = build_agent(args)
    goal = np.array([args.land[0], args.land[1], args.speed,
                     args.topspin, args.sidespin], dtype=np.float32)

    if args.incoming is not None:
        states = np.array([args.incoming], dtype=np.float32)
    else:
        n = max(1, args.random)
        pos, vel, spin = serve.incoming_at(dataset.STRIKE_X, n, rng)
        states = dataset.encode_state(pos, vel, spin).astype(np.float32)

    goals = np.tile(goal, (len(states), 1))
    print(f"planner: {label}")
    print(f"request: land ({goal[0]:+.2f}, {goal[1]:+.2f}) m, "
          f"speed {goal[2]:.1f} m/s, topspin {goal[3]:+.0f}, "
          f"sidespin {goal[4]:+.0f} rad/s\n")

    if isinstance(agent, agents.CEMOracle):
        pos = np.stack([np.full(len(states), dataset.STRIKE_X),
                        states[:, 0], states[:, 1]], axis=1)
        actions = agent.act(states, goals, pos, states[:, 2:5], states[:, 5:8])
    else:
        actions = agent.act(states, goals)

    cards = []
    for s, a in zip(states, actions):
        card = stroke_card.describe(s, a, goal=goal)
        cards.append(card)
        print(stroke_card.format_card(card))
        print()

    if args.json:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(clean(cards), f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
