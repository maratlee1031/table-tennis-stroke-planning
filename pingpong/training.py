"""Training loops and the shared evaluation harness.

Every method is finally judged the same way: take its chosen stroke, play it
through the **real physics**, and see where the ball actually lands. No
method is ever scored on its own surrogate, which would just measure how
confidently it is wrong.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from . import agents, dataset, models, progress


def _loader(tensors, batch, shuffle=True, gen=None):
    n = tensors[0].shape[0]
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, batch):
        j = idx[i:i + batch]
        yield [t[j] for t in tensors]


def train_forward(data, hidden=(256, 256, 256), epochs=40, batch=1024,
                  lr=2e-3, val_frac=0.1, dev=None, seed=0, verbose=True):
    """Fit ``(state, action) -> (landing, success)``.

    The landing head is trained on every ball that came down anywhere,
    including the ones that sailed past the end of the table -- only strokes
    that hit the net or never resolved are excluded. Restricting it to
    successful strokes would leave the model with almost no supervision
    (uniform actions land in about 0.7% of the time) and, worse, no way to
    tell an optimiser that a stroke overshoots. Whether the ball was *in* is
    the separate job of the success head.
    """
    torch.manual_seed(seed)
    dev = dev or models.device()
    model = models.ForwardModel(hidden=hidden).to(dev)

    s = torch.as_tensor(data["state"], device=dev)
    a = torch.as_tensor(data["action"], device=dev)
    l = torch.as_tensor(data["goal"], device=dev)
    ok = torch.as_tensor(data["success"].astype(np.float32), device=dev)
    has = torch.as_tensor(
        data.get("has_landing", np.ones(len(l), bool)).astype(np.float32), device=dev)

    n_val = max(1, int(len(s) * val_frac))
    perm = torch.randperm(len(s), device=dev)
    va, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    history = []
    pbar = progress.bar(epochs, "forward surrogate", "epoch", enabled=verbose)
    for ep in range(epochs):
        model.train()
        for bs, ba, bl, bo, bh in _loader([s[tr], a[tr], l[tr], ok[tr], has[tr]], batch):
            opt.zero_grad(set_to_none=True)
            pl, plog = model(bs, ba)
            land_loss = (F.smooth_l1_loss(pl, bl, reduction="none").mean(-1) * bh).sum()
            land_loss = land_loss / bh.sum().clamp(min=1.0)
            ok_loss = F.binary_cross_entropy_with_logits(plog, bo)
            (land_loss + ok_loss).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pl, plog = model(s[va], a[va])
            m = has[va] > 0.5
            err = ((pl[m] - l[va][m]).norm(dim=-1).mean().item()
                   if m.any() else float("nan"))
            in_m = ok[va] > 0.5
            err_in = ((pl[in_m] - l[va][in_m]).norm(dim=-1).mean().item()
                      if in_m.any() else float("nan"))
            acc = (((plog > 0) == in_m).float().mean().item())
        history.append({"epoch": ep, "landing_mae_m": err,
                        "landing_mae_in_m": err_in, "success_acc": acc})
        pbar.update(1)
        pbar.set_postfix_str(f", goal err {err:.3f}, ok acc {acc * 100:.1f}%")
    pbar.close()
    return model, history


def train_policy(demos, hidden=(256, 256, 256), epochs=60, batch=1024,
                 lr=2e-3, val_frac=0.1, dev=None, seed=0, verbose=True):
    """Fit ``(state, target) -> action`` by regression on expert strokes."""
    torch.manual_seed(seed)
    dev = dev or models.device()
    model = models.PolicyMLP(hidden=hidden).to(dev)

    ok = demos["success"]
    s = torch.as_tensor(demos["state"][ok], device=dev)
    t = torch.as_tensor(demos["goal"][ok], device=dev)
    a = torch.as_tensor(demos["action"][ok], device=dev)
    scale = torch.as_tensor(dataset.ACTION_SCALE.astype(np.float32), device=dev)

    n_val = max(1, int(len(s) * val_frac))
    perm = torch.randperm(len(s), device=dev)
    va, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    history = []
    pbar = progress.bar(epochs, "direct policy", "epoch", enabled=verbose)
    for ep in range(epochs):
        model.train()
        for bs, bt, ba in _loader([s[tr], t[tr], a[tr]], batch):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(bs, bt) / scale, ba / scale)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(s[va], t[va]) / scale, a[va] / scale).item()
        history.append({"epoch": ep, "val_mse": v})
        pbar.update(1)
        pbar.set_postfix_str(f", val MSE {v:.4f}")
    pbar.close()
    return model, history


def train_mdn(demos, hidden=(256, 256, 256), n_components=5, epochs=60,
              batch=1024, lr=2e-3, val_frac=0.1, dev=None, seed=0, verbose=True):
    """Fit a mixture density policy by maximum likelihood."""
    torch.manual_seed(seed)
    dev = dev or models.device()
    model = models.MDNPolicy(hidden=hidden, n_components=n_components).to(dev)

    ok = demos["success"]
    s = torch.as_tensor(demos["state"][ok], device=dev)
    t = torch.as_tensor(demos["goal"][ok], device=dev)
    a = torch.as_tensor(demos["action"][ok], device=dev)

    n_val = max(1, int(len(s) * val_frac))
    perm = torch.randperm(len(s), device=dev)
    va, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    history = []
    pbar = progress.bar(epochs, "mixture density policy", "epoch", enabled=verbose)
    for ep in range(epochs):
        model.train()
        for bs, bt, ba in _loader([s[tr], t[tr], a[tr]], batch):
            opt.zero_grad(set_to_none=True)
            loss = model.nll(bs, bt, ba)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = model.nll(s[va], t[va], a[va]).item()
        history.append({"epoch": ep, "val_nll": v})
        pbar.update(1)
        pbar.set_postfix_str(f", val NLL {v:.2f}")
    pbar.close()
    return model, history


# ---------------------------------------------------------------- evaluation
def evaluate(agent, eval_set, verify_dt=1.0 / 480.0, chunk=2000):
    """Score an agent by playing its strokes through the true physics.

    The goal has five parts -- where, how fast, how much topspin, how much
    sidespin -- so the report keeps them separate as well as combined. A
    method can be excellent at placement and useless at spin, and a single
    number would hide that.

    Errors are over *successful* returns only: how far from the requested
    spin a ball that went into the net ended up is not a meaningful number,
    and the success rate already accounts for those.
    """
    pos, vel, spin, goals = eval_set
    state = dataset.encode_state(pos, vel, spin)

    actions = np.empty((len(state), dataset.ACTION_DIM), np.float32)
    pbar = progress.bar(len(state), f"eval {agent.name}", "strokes", leave=False)
    t0 = time.perf_counter()
    for i in range(0, len(state), chunk):
        sl = slice(i, i + chunk)
        if isinstance(agent, agents.CEMOracle):
            actions[sl] = agent.act(state[sl], goals[sl],
                                    pos[sl], vel[sl], spin[sl])
        else:
            actions[sl] = agent.act(state[sl], goals[sl])
        pbar.update(len(actions[sl]))
    pbar.close()
    infer_s = time.perf_counter() - t0

    achieved, outcome, _ = dataset.apply_actions(pos, vel, spin, actions,
                                                 dt=verify_dt, max_time=3.0)
    ok = outcome == 1
    if not ok.any():
        per_dim = {f"err_{k}": float("nan") for k in dataset.GOAL_NAMES}
        combined = np.array([np.nan])
        place = np.array([np.nan])
    else:
        d = np.abs(achieved[ok] - goals[ok])
        per_dim = {f"err_{k}": float(np.median(d[:, i]))
                   for i, k in enumerate(dataset.GOAL_NAMES)}
        combined = dataset.goal_error(achieved[ok], goals[ok])
        place = np.linalg.norm(achieved[ok][:, :2] - goals[ok][:, :2], axis=1)

    out = {
        "agent": agent.name,
        "n": int(len(state)),
        "success_rate": float(ok.mean()),
        # Combined score in normalised, weighted units -- comparable across
        # dimensions that are metres, m/s and rad/s
        "goal_err_median": float(np.median(combined)),
        "goal_err_p90": float(np.percentile(combined, 90)),
        # Placement alone, in metres, so it stays comparable with the
        # earlier landing-only experiments
        "place_err_median_m": float(np.median(place)),
        "within_10cm": float(np.mean(place < 0.10)) if ok.any() else 0.0,
        "infer_ms_per_stroke": float(infer_s / len(state) * 1000),
        "net_rate": float((outcome == 3).mean()),
        "out_rate": float((outcome == 2).mean()),
    }
    out.update(per_dim)
    return out
