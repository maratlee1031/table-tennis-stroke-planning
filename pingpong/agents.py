"""Every way of choosing a stroke, behind one interface.

    agent.act(state, target) -> action        both batched, numpy in / numpy out

Having them share an interface is the whole point: the comparison then
measures the method rather than differences in how each was wired up. All of
them are finally scored the same way, by playing their chosen stroke through
the real physics.
"""

import numpy as np
import torch

from . import dataset, models, solver


class Agent:
    name = "agent"

    def act(self, state, target):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


# ---------------------------------------------------------------- baselines
class CEMOracle(Agent):
    """CEM run against the true physics. Not learned -- the upper bound.

    Anything learned is trying to approximate this while being faster.
    """

    name = "cem-oracle"

    def __init__(self, iters=5, pop=64, sim_dt=solver.SOLVER_DT, seed=0):
        self.iters, self.pop, self.sim_dt = iters, pop, sim_dt
        self.rng = np.random.default_rng(seed)

    def act(self, state, target, pos=None, vel=None, spin=None):
        if pos is None:
            pos, vel, spin = state_to_ball(state)
        a, _, _, _ = solver.solve_batch(pos, vel, spin, target,
                                        iters=self.iters, pop=self.pop,
                                        rng=self.rng, sim_dt=self.sim_dt)
        return a


class KNNAgent(Agent):
    """Nearest-neighbour retrieval over expert demonstrations.

    This is what the original project did: find past strokes whose landing
    was closest to the target and copy them. It cannot generalise beyond the
    stored examples, which is exactly what the comparison should show.
    """

    name = "knn"

    def __init__(self, demos, k=5):
        self.k = k
        q = np.concatenate([demos["state"] / dataset.STATE_SCALE,
                            demos["goal"] / dataset.GOAL_SCALE], axis=1)
        self.q = q.astype(np.float32)
        self.a = demos["action"].astype(np.float32)
        self.qn = (self.q ** 2).sum(1)               # reference norms, for the expansion below

    def act(self, state, target):
        query = np.concatenate([state / dataset.STATE_SCALE,
                                target / dataset.GOAL_SCALE], axis=1).astype(np.float32)
        out = np.empty((len(query), dataset.ACTION_DIM), np.float32)
        step = 512
        for i in range(0, len(query), step):
            blk = query[i:i + step]
            # |a-b|^2 = |a|^2 + |b|^2 - 2a.b, so the distances are one matrix
            # product instead of a (step, N, dim) broadcast. That broadcast is
            # 5 GB per chunk at 200k demonstrations, where this is 0.4 GB and
            # runs in BLAS. |a|^2 is constant along a row and cannot change
            # which neighbours are nearest, so it is left out.
            d = self.qn[None, :] - 2.0 * (blk @ self.q.T)
            idx = np.argpartition(d, self.k, axis=1)[:, :self.k]
            out[i:i + step] = self.a[idx].mean(axis=1)
        return out


# ---------------------------------------------------------------- learned
class PolicyAgent(Agent):
    """Direct policy network: one forward pass, no search."""

    name = "policy-mlp"

    def __init__(self, model, dev=None):
        self.dev = dev or models.device()
        self.model = model.to(self.dev).eval()

    @torch.no_grad()
    def act(self, state, target):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.dev)
        t = torch.as_tensor(target, dtype=torch.float32, device=self.dev)
        return self.model(s, t).cpu().numpy()


class MDNAgent(Agent):
    """Mixture policy. Optionally re-ranks its own modes with a forward model.

    Without the ranker it simply takes the highest-weight component; with one
    it evaluates every mode and keeps whichever the surrogate says lands
    closest. The mixture proposes, the forward model disposes.
    """

    name = "mdn"

    def __init__(self, model, ranker=None, dev=None):
        self.dev = dev or models.device()
        self.model = model.to(self.dev).eval()
        self.ranker = ranker.to(self.dev).eval() if ranker is not None else None
        if ranker is not None:
            self.name = "mdn+rank"

    @torch.no_grad()
    def act(self, state, target):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.dev)
        t = torch.as_tensor(target, dtype=torch.float32, device=self.dev)
        if self.ranker is None:
            return self.model.best_mode(s, t).cpu().numpy()

        modes = self.model.sample_modes(s, t)          # (N,K,D)
        n, k, d = modes.shape
        flat = modes.reshape(n * k, d)
        s_rep = s.repeat_interleave(k, 0)
        t_rep = t.repeat_interleave(k, 0)
        got, ok = self.ranker(s_rep, flat)
        cost = goal_cost(got, t_rep, self.dev) + 3.0 * (1.0 - torch.sigmoid(ok))
        best = cost.reshape(n, k).argmin(dim=1)
        return modes[torch.arange(n, device=self.dev), best].cpu().numpy()


class SurrogateCEMAgent(Agent):
    """CEM, but the rollouts run on the learned forward model.

    Same search as the oracle with the physics swapped for a network, so the
    entire population evaluates as one batched GPU forward pass instead of
    thousands of integration steps.
    """

    name = "surrogate-cem"

    def __init__(self, model, iters=4, pop=256, elite_frac=0.15, dev=None, seed=0):
        self.dev = dev or models.device()
        self.model = model.to(self.dev).eval()
        self.iters, self.pop, self.elite_frac = iters, pop, elite_frac
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

    @torch.no_grad()
    def act(self, state, target):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.dev)
        t = torch.as_tensor(target, dtype=torch.float32, device=self.dev)
        n, k = s.shape[0], self.pop
        low = torch.as_tensor(dataset.ACTION_LOW, dtype=torch.float32, device=self.dev)
        high = torch.as_tensor(dataset.ACTION_HIGH, dtype=torch.float32, device=self.dev)

        mean = torch.as_tensor(solver.INIT_MEAN, dtype=torch.float32,
                               device=self.dev).repeat(n, 1)
        std = torch.as_tensor(solver.INIT_STD, dtype=torch.float32,
                              device=self.dev).repeat(n, 1)
        floor = torch.as_tensor(solver.FLOOR_STD, dtype=torch.float32, device=self.dev)
        n_elite = max(4, int(k * self.elite_frac))

        s_rep = s.repeat_interleave(k, 0)
        t_rep = t.repeat_interleave(k, 0)
        best_a = mean.clone()
        best_c = torch.full((n,), float("inf"), device=self.dev)

        for _ in range(self.iters):
            eps = torch.randn(n, k, dataset.ACTION_DIM, device=self.dev,
                              generator=self.gen)
            cand = (mean[:, None, :] + std[:, None, :] * eps).clamp(low, high)
            got, ok = self.model(s_rep, cand.reshape(-1, dataset.ACTION_DIM))
            cost = (goal_cost(got, t_rep, self.dev)
                    + 3.0 * (1.0 - torch.sigmoid(ok))).reshape(n, k)

            order = cost.argsort(dim=1)
            top = order[:, 0]
            rows = torch.arange(n, device=self.dev)
            better = cost[rows, top] < best_c
            best_c = torch.where(better, cost[rows, top], best_c)
            best_a = torch.where(better[:, None], cand[rows, top], best_a)

            elites = torch.gather(
                cand, 1, order[:, :n_elite, None].expand(-1, -1, dataset.ACTION_DIM))
            mean = elites.mean(1)
            std = elites.std(1) + floor

        return best_a.cpu().numpy()


class SurrogateGradAgent(Agent):
    """Invert the forward model by gradient descent through the network.

    The surrogate is differentiable, so instead of sampling the action space
    the action itself can be optimised with autograd. Cheapest of the
    search-based methods, and it is only possible because the forward model
    is a network rather than a simulator.
    """

    name = "surrogate-grad"

    def __init__(self, model, steps=60, lr=0.08, restarts=8, dev=None, seed=0):
        self.dev = dev or models.device()
        self.model = model.to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.steps, self.lr, self.restarts = steps, lr, restarts
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

    def act(self, state, target):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.dev)
        t = torch.as_tensor(target, dtype=torch.float32, device=self.dev)
        n, r = s.shape[0], self.restarts
        low = torch.as_tensor(dataset.ACTION_LOW, dtype=torch.float32, device=self.dev)
        high = torch.as_tensor(dataset.ACTION_HIGH, dtype=torch.float32, device=self.dev)

        # Several restarts, because the surrogate's landscape is not convex
        # and a single descent lands in whatever basin it started in
        init = torch.as_tensor(solver.INIT_MEAN, dtype=torch.float32, device=self.dev)
        spread = torch.as_tensor(solver.INIT_STD, dtype=torch.float32, device=self.dev)
        a = (init + spread * torch.randn(n * r, dataset.ACTION_DIM,
                                         device=self.dev, generator=self.gen))
        a = a.clamp(low, high).requires_grad_(True)

        s_rep = s.repeat_interleave(r, 0)
        t_rep = t.repeat_interleave(r, 0)
        opt = torch.optim.Adam([a], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            got, ok = self.model(s_rep, a)
            loss = (goal_cost(got, t_rep, self.dev)
                    + 3.0 * torch.nn.functional.softplus(-ok)).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(low, high)

        with torch.no_grad():
            got, ok = self.model(s_rep, a)
            cost = (goal_cost(got, t_rep, self.dev)
                    + 3.0 * (1.0 - torch.sigmoid(ok))).reshape(n, r)
            best = cost.argmin(dim=1)
            chosen = a.reshape(n, r, -1)[torch.arange(n, device=self.dev), best]
        return chosen.detach().cpu().numpy()


# ---------------------------------------------------------------- helpers
_GOAL_W = {}


def goal_cost(achieved, goal, dev):
    """Weighted goal error in normalised units -- the torch twin of
    dataset.goal_error. Raw metres would let a 300 rad/s spin miss dominate
    a 30 cm landing miss purely through unit choice."""
    key = str(dev)
    if key not in _GOAL_W:
        _GOAL_W[key] = (
            torch.as_tensor((dataset.GOAL_WEIGHTS / dataset.GOAL_SCALE).astype(np.float32),
                            device=dev))
    return ((achieved - goal) * _GOAL_W[key]).norm(dim=-1)
def state_to_ball(state):
    """Rebuild (pos, vel, spin) from the packed state, for the true physics."""
    state = np.atleast_2d(state)
    pos = np.stack([np.full(len(state), dataset.STRIKE_X),
                    state[:, 0], state[:, 1]], axis=1)
    return pos, state[:, 2:5].copy(), state[:, 5:8].copy()
