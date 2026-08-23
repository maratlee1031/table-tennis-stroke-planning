"""Train and compare ways of returning a ball to a chosen spot.

    python train_ai.py data                 # generate the datasets (slowest step)
    python train_ai.py train                # train every model variant
    python train_ai.py eval                 # score everything, write results.csv
    python train_ai.py report               # draw the comparison figures
    python train_ai.py all                  # the whole pipeline

Useful flags:

    --sizes 2000 20000 200000     dataset sizes to sweep
    --demo-frac 1.0               demonstrations per size (default 0.1 of it)
    --only forward policy mdn     restrict which model families run
    --eval-n 3000                 size of the held-out problem set
    --workers 14                  processes generating demonstrations
    --cpu                         ignore the GPU

Everything lands in runs/: datasets, checkpoints, results.csv, figures.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from pingpong import agents, dataset, models, solver, training

RUNS = "runs"
DATA_DIR = os.path.join(RUNS, "data")
CKPT_DIR = os.path.join(RUNS, "checkpoints")
FIG_DIR = os.path.join(RUNS, "figures")
RESULTS = os.path.join(RUNS, "results.csv")

DEFAULT_SIZES = [2000, 20000, 200000]
# Expert demonstrations cost far more per sample than random ones (each is a
# CEM solve), so by default the policy sweep uses a smaller ladder: --sizes
# names the forward model's dataset and the policies get this fraction of it.
#
# That coupling is only a cost heuristic, and it silently caps the policies.
# Reaching 200k demonstrations through it means asking for 2,000,000 random
# and mixed samples, and the mixed set alone is 140,000 single-threaded CEM
# solves for data no policy ever reads. --demo-frac 1.0 breaks the link and
# makes --sizes mean demonstrations directly.
EXPERT_FRACTION = 0.1


def ensure_dirs():
    for d in (RUNS, DATA_DIR, CKPT_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)


def rand_path(n):
    return os.path.join(DATA_DIR, f"random_{n}.npz")


def mixed_path(n):
    return os.path.join(DATA_DIR, f"mixed_{n}.npz")


def expert_path(n, args=None):
    # Budget and robustness settings in the filename, so demonstrations of
    # different quality never silently overwrite each other
    return os.path.join(DATA_DIR, f"expert_{n}{demo_tag(args) if args else ''}.npz")


def demo_tag(args):
    """Suffix identifying the *demonstrations* -- their solver budget and
    whether they were chosen for robustness. Names the dataset file, so a
    better-taught run never silently overwrites the one it is meant to be
    compared against."""
    tag = ""
    if (args.expert_iters, args.expert_pop) != (5, 64):
        tag += f"_i{args.expert_iters}p{args.expert_pop}"
    if getattr(args, "robust_k", 0) > 0:
        tag += f"_r{args.robust_k}w{args.robust_weight:g}"
    return tag


def model_tag(args):
    """Suffix identifying a *trained model*: its demonstrations plus whatever
    the user is sweeping. Deliberately not part of :func:`demo_tag` -- a
    capacity sweep reuses one expensive dataset and must not go looking for a
    per-capacity copy of it."""
    return demo_tag(args) + (("_" + args.run_tag)
                             if getattr(args, "run_tag", "") else "")


def demo_label(args):
    """Human-readable version of :func:`model_tag`, for the results table."""
    bits = []
    if (args.expert_iters, args.expert_pop) != (5, 64):
        bits.append("hq")
    if getattr(args, "robust_k", 0) > 0:
        bits.append("robust")
    if getattr(args, "run_tag", ""):
        bits.append(args.run_tag)
    return "+".join(bits)


def eval_path(n):
    return os.path.join(DATA_DIR, f"evalset_{n}.npz")


# ---------------------------------------------------------------- data
def cmd_data(args):
    ensure_dirs()
    rng = np.random.default_rng(args.seed)

    for n in args.sizes:
        p = rand_path(n)
        if os.path.exists(p) and not args.force:
            print(f"[data] {p} exists, skipping")
        else:
            print(f"[data] random actions, {n:,} samples "
                  f"(for the forward model) ...")
            t0 = time.perf_counter()
            d = dataset.generate(n, rng, verbose=not args.quiet)
            dataset.save(p, d)
            print(f"       {time.perf_counter() - t0:.1f}s   "
                  f"{d['success'].mean() * 100:.1f}% landed in   -> {p}")

        p = mixed_path(n)
        if os.path.exists(p) and not args.force:
            print(f"[data] {p} exists, skipping")
        else:
            print(f"[data] mixed actions, {n:,} samples "
                  f"(uniform + solved strokes and their neighbourhood) ...")
            t0 = time.perf_counter()
            d = dataset.generate_mixed(n, rng, expert_frac=args.expert_frac,
                                       verbose=not args.quiet)
            dataset.save(p, d)
            print(f"       {time.perf_counter() - t0:.1f}s   "
                  f"{d['success'].mean() * 100:.1f}% landed in   -> {p}")

        m = max(500, int(n * args.demo_frac))
        p = expert_path(m, args)
        if os.path.exists(p) and not args.force:
            print(f"[data] {p} exists, skipping")
        else:
            print(f"[data] expert demonstrations, {m:,} CEM solves "
                  f"(for the policies) ...")
            t0 = time.perf_counter()
            d = solver.expert_dataset(m, rng, iters=args.expert_iters,
                                      pop=args.expert_pop,
                                      robust_k=args.robust_k,
                                      robust_weight=args.robust_weight,
                                      workers=args.workers,
                                      verbose=not args.quiet)
            dataset.save(p, d)
            print(f"       {time.perf_counter() - t0:.1f}s   "
                  f"{d['success'].mean() * 100:.1f}% solved   -> {p}")

    p = eval_path(args.eval_n)
    if os.path.exists(p) and not args.force:
        print(f"[data] {p} exists, skipping")
    else:
        print(f"[data] held-out evaluation set, {args.eval_n:,} problems ...")
        pos, vel, spin, tg = dataset.make_eval_set(args.eval_n)
        np.savez_compressed(p, pos=pos, vel=vel, spin=spin, targets=tg)
        print(f"       -> {p}")


def load_eval(n):
    z = np.load(eval_path(n))
    return z["pos"], z["vel"], z["spin"], z["targets"]


# ---------------------------------------------------------------- train
def cmd_train(args):
    ensure_dirs()
    dev = models.device(prefer_gpu=not args.cpu)
    print(f"[train] device: {dev}")

    for n in args.sizes:
        if "forward" in args.only:
            # Two forward models per size: one on uniform actions only, one
            # on the mixed set. The difference between them is the whole
            # covariate-shift story, so both are kept and both are scored.
            for tag, path, label in (("forward", rand_path(n), "uniform actions only"),
                                     ("forwardmix", mixed_path(n), "mixed actions")):
                if not os.path.exists(path):
                    print(f"[train] missing {path}, run `data` first")
                    continue
                print(f"\n[train] forward surrogate on {n:,} samples ({label})")
                d = dataset.load(path)
                t0 = time.perf_counter()
                model, hist = train_wrap(training.train_forward, d,
                                         hidden=args.hidden, epochs=args.epochs,
                                         dev=dev, seed=args.seed,
                                         verbose=not args.quiet)
                save_model(model, f"{tag}_{n}", n, hist,
                           time.perf_counter() - t0, args)

        m = max(500, int(n * args.demo_frac))
        src = expert_path(m, args)
        if not os.path.exists(src):
            print(f"[train] missing {src}, run `data` first")
            continue
        demos = dataset.load(src)

        if "policy" in args.only:
            print(f"\n[train] direct policy on {m:,} expert strokes")
            t0 = time.perf_counter()
            model, hist = train_wrap(training.train_policy, demos,
                                     hidden=args.hidden, epochs=args.epochs,
                                     dev=dev, seed=args.seed,
                                     verbose=not args.quiet)
            save_model(model, f"policy_{m}{model_tag(args)}", m, hist,
                       time.perf_counter() - t0, args)

        if "mdn" in args.only:
            print(f"\n[train] mixture density policy on {m:,} expert strokes")
            t0 = time.perf_counter()
            model, hist = train_wrap(training.train_mdn, demos,
                                     hidden=args.hidden, epochs=args.epochs,
                                     n_components=args.components,
                                     dev=dev, seed=args.seed,
                                     verbose=not args.quiet)
            save_model(model, f"mdn_{m}{model_tag(args)}", m, hist,
                       time.perf_counter() - t0, args)


def train_wrap(fn, data, **kw):
    return fn(data, **kw)


def save_model(model, tag, n_train, history, seconds, args):
    path = os.path.join(CKPT_DIR, f"{tag}.pt")
    models.save(path, model, meta={
        "hidden": list(args.hidden),
        "n_components": args.components,
        "n_train": int(n_train),
        "epochs": args.epochs,
        "train_seconds": round(seconds, 1),
        "history": history[-1] if history else {},
    })
    print(f"    trained in {seconds:.1f}s  -> {path}")


# ---------------------------------------------------------------- evaluate
def cmd_eval(args):
    ensure_dirs()
    dev = models.device(prefer_gpu=not args.cpu)
    ev = load_eval(args.eval_n)
    print(f"[eval] {args.eval_n:,} held-out problems on {dev}\n")

    rows = []
    tag = model_tag(args)
    label = demo_label(args)

    def run(agent, family, n_train, demo_quality=False):
        # Policies inherit their demonstrations' quality, so that has to be
        # part of their identity in the results; the forward models never
        # see demonstrations at all.
        if demo_quality and label:
            family = f"{family} ({label} demos)"
        t0 = time.perf_counter()
        r = training.evaluate(agent, ev)
        r.update(family=family, n_train=n_train, eval_n=args.eval_n,
                 demo_iters=args.expert_iters, demo_pop=args.expert_pop,
                 robust_k=getattr(args, "robust_k", 0),
                 robust_weight=getattr(args, "robust_weight", 0.0),
                 wall_seconds=round(time.perf_counter() - t0, 1))
        rows.append(r)
        print(f"  {r['agent']:<24} n={n_train:>7}  "
              f"success {r['success_rate'] * 100:5.1f}%  "
              f"place {r['place_err_median_m'] * 100:5.1f} cm  "
              f"speed {r['err_speed']:4.2f} m/s  "
              f"topspin {r['err_topspin']:5.1f}  "
              f"goal {r['goal_err_median']:.3f}  "
              f"{r['infer_ms_per_stroke']:6.2f} ms")

    # --- baselines ---
    if not args.skip_oracle:
        print("baselines")
        run(agents.CEMOracle(), "oracle", 0)

    for n in args.sizes:
        m = max(500, int(n * args.demo_frac))
        p = expert_path(m, args)
        if os.path.exists(p):
            run(agents.KNNAgent(dataset.load(p), k=args.knn_k), "knn", m,
                demo_quality=True)

    # --- learned ---
    print("\nlearned")
    for n in args.sizes:
        fwd = None
        # Not `tag`: that is the demonstration-quality suffix from the
        # enclosing scope, and shadowing it here sent the policy lookups
        # hunting for files called mdn_2000forwardmix.pt
        for kind, suffix in (("forward", "-uniform"), ("forwardmix", "")):
            fp = os.path.join(CKPT_DIR, f"{kind}_{n}.pt")
            if not os.path.exists(fp):
                continue
            m_fwd, _ = models.load(fp, map_location=dev)
            cem = agents.SurrogateCEMAgent(m_fwd, dev=dev)
            grad = agents.SurrogateGradAgent(m_fwd, dev=dev)
            cem.name += suffix
            grad.name += suffix
            run(cem, "surrogate-cem" + suffix, n)
            run(grad, "surrogate-grad" + suffix, n)
            if kind == "forwardmix":
                fwd = m_fwd
        fp = os.path.join(CKPT_DIR, f"forwardmix_{n}.pt")

        m = max(500, int(n * args.demo_frac))
        pp = os.path.join(CKPT_DIR, f"policy_{m}{tag}.pt")
        if os.path.exists(pp):
            pol, _ = models.load(pp, map_location=dev)
            run(agents.PolicyAgent(pol, dev=dev), "policy-mlp", m,
                demo_quality=True)

        mp = os.path.join(CKPT_DIR, f"mdn_{m}{tag}.pt")
        if os.path.exists(mp):
            mdn, _ = models.load(mp, map_location=dev)
            run(agents.MDNAgent(mdn, dev=dev), "mdn", m, demo_quality=True)
            if os.path.exists(fp):
                run(agents.MDNAgent(mdn, ranker=fwd, dev=dev), "mdn+rank", n,
                    demo_quality=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    if os.path.exists(RESULTS):
        # Merge rather than overwrite: a second run with different settings
        # is the whole point of the comparison, and clobbering the first one
        # would throw away hours of work.
        old = pd.read_csv(RESULTS)
        # eval_n is part of the identity: a quick 200-problem smoke run
        # must not overwrite a 3000-problem measurement of the same model.
        if "eval_n" not in old.columns:
            old["eval_n"] = 3000
        key = ["family", "n_train", "eval_n"]
        old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RESULTS, index=False)
    print(f"\n[eval] -> {RESULTS}")
    return df


# ---------------------------------------------------------------- report
def cmd_report(args):
    from ml_report import build_report
    build_report(RESULTS, FIG_DIR)


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["data", "train", "eval", "report", "all"])
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--only", nargs="+", default=["forward", "policy", "mdn"],
                    choices=["forward", "policy", "mdn"])
    ap.add_argument("--eval-n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--components", type=int, default=5)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--expert-frac", type=float, default=0.35,
                    help="share of the mixed dataset that comes from solved strokes")
    ap.add_argument("--demo-frac", type=float, default=EXPERT_FRACTION,
                    help="demonstrations per --sizes entry, as a fraction of it "
                         "(default 0.1). Use 1.0 to make --sizes mean "
                         "demonstrations directly and stop the policy ladder "
                         "being capped by the forward model's dataset")
    # Demonstration quality caps what a policy can learn. At the default
    # 5x64 the solver leaves ~18.6 cm of placement error, most of it search
    # noise: 8x256 reaches 11.2 cm and 10x512 reaches 7.8 cm. Raising these
    # costs generation time but lifts the ceiling for every policy trained
    # on the result.
    ap.add_argument("--expert-iters", type=int, default=5,
                    help="CEM iterations per demonstration")
    ap.add_argument("--expert-pop", type=int, default=64,
                    help="CEM population per demonstration")
    # Demonstrations chosen purely for accuracy sit on the edge of what is
    # feasible, and a policy imitating them inherits that fragility. This
    # re-ranks the final candidates by how well they survive being executed
    # imperfectly.
    ap.add_argument("--robust-k", type=int, default=0,
                    help="perturbations per candidate in the robustness pass "
                         "(0 disables it)")
    ap.add_argument("--robust-weight", type=float, default=solver.ROBUST_WEIGHT,
                    help="how much fragility counts against accuracy")
    # Model capacity is not in the checkpoint name, so a sweep over
    # --components or --hidden would overwrite itself. Name the run.
    # Each demonstration is single-core numpy, and the chunks are
    # independent, so this is close to a linear speed-up on the slowest
    # stage of the pipeline.
    ap.add_argument("--workers", type=int, default=None,
                    help="processes generating demonstrations "
                         "(default: cores - 2)")
    ap.add_argument("--run-tag", default="",
                    help="free-form suffix for checkpoints and result rows, "
                         "so sweeps do not overwrite each other")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate existing data")
    ap.add_argument("--skip-oracle", action="store_true",
                    help="the CEM oracle is accurate but slow to evaluate")
    ap.add_argument("--quiet", action="store_true",
                    help="hide progress bars")
    args = ap.parse_args()

    args.hidden = tuple(args.hidden)
    torch.manual_seed(args.seed)

    if args.stage in ("data", "all"):
        cmd_data(args)
    if args.stage in ("train", "all"):
        cmd_train(args)
    if args.stage in ("eval", "all"):
        cmd_eval(args)
    if args.stage in ("report", "all"):
        cmd_report(args)


if __name__ == "__main__":
    main()
