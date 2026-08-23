"""Neural models for the return-the-ball problem.

Three, deliberately, because they represent three different answers to the
same question and the point of the project is to compare them.

**ForwardModel** -- learns ``(state, action) -> (landing, success)``. This
mapping is single-valued, so plain regression is well-posed. It is not a
policy: to get a stroke out of it you invert it at query time, either by CEM
or by gradient descent through the network.

**PolicyMLP** -- learns ``(state, target) -> action`` directly, with MSE.
This mapping is **one-to-many**: many different strokes put the ball on the
same spot, and MSE regression converges to their mean, which is usually not
a valid stroke at all. It is included precisely to show that failure.

**MDNPolicy** -- same inputs, but predicts a mixture of Gaussians over
actions instead of a point. That is the standard fix for the one-to-many
problem: the model can keep several distinct strokes alive rather than
averaging them into something that works for neither.
"""

import numpy as np
import torch
import torch.nn as nn

from . import dataset


def device(prefer_gpu=True):
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")


def mlp(sizes, act=nn.SiLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class Normaliser(nn.Module):
    """Fixed-scale normalisation, registered as buffers so it travels with
    the checkpoint. Fixed rather than fitted, so a model trained on one
    dataset stays valid on another."""

    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(np.asarray(scale, np.float32)))

    def forward(self, x):
        return x / self.scale


class ForwardModel(nn.Module):
    """(state, action) -> (achieved goal, success logit)."""

    kind = "forward"

    def __init__(self, hidden=(256, 256, 256)):
        super().__init__()
        self.ns = Normaliser(dataset.STATE_SCALE)
        self.na = Normaliser(dataset.ACTION_SCALE)
        n_in = dataset.STATE_DIM + dataset.ACTION_DIM
        self.body = mlp([n_in, *hidden])
        self.head_land = nn.Linear(hidden[-1], dataset.GOAL_DIM)
        self.head_ok = nn.Linear(hidden[-1], 1)
        self.register_buffer("tscale",
                             torch.as_tensor(dataset.GOAL_SCALE.astype(np.float32)))

    def forward(self, state, action):
        h = self.body(torch.cat([self.ns(state), self.na(action)], dim=-1))
        return self.head_land(h) * self.tscale, self.head_ok(h).squeeze(-1)


class PolicyMLP(nn.Module):
    """(state, goal) -> action, as a point estimate."""

    kind = "policy"

    def __init__(self, hidden=(256, 256, 256)):
        super().__init__()
        self.ns = Normaliser(dataset.STATE_SCALE)
        self.nt = Normaliser(dataset.GOAL_SCALE)
        n_in = dataset.STATE_DIM + dataset.GOAL_DIM
        self.net = mlp([n_in, *hidden, dataset.ACTION_DIM])
        self.register_buffer("ascale",
                             torch.as_tensor(dataset.ACTION_SCALE.astype(np.float32)))

    def forward(self, state, target):
        return self.net(torch.cat([self.ns(state), self.nt(target)], dim=-1)) * self.ascale


class MDNPolicy(nn.Module):
    """(state, goal) -> mixture of Gaussians over actions.

    Keeps several candidate strokes alive instead of averaging them, which
    is what makes it viable where the plain MLP is not.
    """

    kind = "mdn"

    def __init__(self, hidden=(256, 256, 256), n_components=5):
        super().__init__()
        self.k = n_components
        self.d = dataset.ACTION_DIM
        self.ns = Normaliser(dataset.STATE_SCALE)
        self.nt = Normaliser(dataset.GOAL_SCALE)
        n_in = dataset.STATE_DIM + dataset.GOAL_DIM
        self.body = mlp([n_in, *hidden])
        self.head = nn.Linear(hidden[-1], self.k * (1 + 2 * self.d))
        self.register_buffer("ascale",
                             torch.as_tensor(dataset.ACTION_SCALE.astype(np.float32)))

    def forward(self, state, target):
        """Returns (log_weights (N,K), means (N,K,D), log_sigma (N,K,D))."""
        h = self.head(self.body(torch.cat([self.ns(state), self.nt(target)], dim=-1)))
        k, d = self.k, self.d
        logit_w = h[:, :k]
        mu = h[:, k:k + k * d].reshape(-1, k, d)
        log_sig = h[:, k + k * d:].reshape(-1, k, d).clamp(-6.0, 2.0)
        return torch.log_softmax(logit_w, dim=-1), mu * self.ascale, log_sig

    def nll(self, state, target, action):
        """Mixture negative log likelihood of the demonstrated action."""
        logw, mu, log_sig = self(state, target)
        a = action.unsqueeze(1)                       # (N,1,D)
        sig = log_sig.exp() * self.ascale             # scale back to action units
        z = (a - mu) / sig
        log_p = (-0.5 * (z ** 2) - log_sig - 0.5 * np.log(2 * np.pi)
                 - torch.log(self.ascale)).sum(-1)     # (N,K)
        return -torch.logsumexp(logw + log_p, dim=-1).mean()

    def best_mode(self, state, target):
        """The mean of the highest-weight component -- a usable single stroke."""
        logw, mu, _ = self(state, target)
        idx = logw.argmax(dim=-1)
        return mu[torch.arange(mu.shape[0], device=mu.device), idx]

    def sample_modes(self, state, target):
        """All component means, (N, K, D), for picking with a forward model."""
        _, mu, _ = self(state, target)
        return mu


REGISTRY = {
    "forward": ForwardModel,
    "policy": PolicyMLP,
    "mdn": MDNPolicy,
}


def save(path, model, meta=None):
    torch.save({"kind": model.kind,
                "state_dict": model.state_dict(),
                "meta": meta or {}}, path)


def load(path, map_location=None):
    blob = torch.load(path, map_location=map_location or device(), weights_only=False)
    meta = blob.get("meta", {})
    cls = REGISTRY[blob["kind"]]
    kwargs = {}
    if "hidden" in meta:
        kwargs["hidden"] = tuple(meta["hidden"])
    if blob["kind"] == "mdn" and "n_components" in meta:
        kwargs["n_components"] = meta["n_components"]
    model = cls(**kwargs)
    model.load_state_dict(blob["state_dict"])
    return model, meta
