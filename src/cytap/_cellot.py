"""Neural optimal transport (CellOT) components used by :mod:`cytap.counterfactual`.

Adapted from CellOT (https://github.com/bunnech/cellot): cellot/networks/icnns.py,
cellot/models/cellot.py, cellot/losses/mmd.py and the CellOT loop of cellot/train/train.py.
Bunne, Stark et al., Learning single-cell perturbation responses using neural optimal
transport. Nature Methods (2023).

Copyright (c) 2022, Stefan G. Stark and Charlotte Bunne. All rights reserved.
Licensed under the BSD 3-Clause License; see LICENSE-CELLOT in this directory.

Changes: configuration and data loading rewritten without ml_collections/absl; tensors are
placed on the model's device; no file logging.
"""
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from torch import autograd, nn

ACTIVATIONS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}


class NonNegativeLinear(nn.Linear):
    def __init__(self, *args, beta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def forward(self, x):
        return nn.functional.linear(x, self.kernel(), self.bias)

    def kernel(self):
        return nn.functional.softplus(self.weight, beta=self.beta)


class ICNN(nn.Module):
    """Input-convex neural network; transport map = gradient of the potential."""

    def __init__(self, input_dim, hidden_units, activation="LeakyReLU", softplus_W_kernels=False,
                 softplus_beta=1, std=0.1, fnorm_penalty=0, kernel_init_fxn=None):
        super().__init__()
        self.fnorm_penalty = fnorm_penalty
        self.softplus_W_kernels = softplus_W_kernels
        if isinstance(activation, str):
            activation = ACTIVATIONS[activation.lower().replace("_", "")]
        self.sigma = activation
        units = list(hidden_units) + [1]

        if softplus_W_kernels:
            def WLinear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)
        else:
            WLinear = nn.Linear

        self.W = nn.ModuleList([WLinear(i, o, bias=False) for i, o in zip(units[:-1], units[1:])])
        self.A = nn.ModuleList([nn.Linear(input_dim, o, bias=True) for o in units])
        if kernel_init_fxn is not None:
            for layer in self.A:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)
            for layer in self.W:
                kernel_init_fxn(layer.weight)

    def forward(self, x):
        z = self.sigma(0.2)(self.A[0](x))
        z = z * z
        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = self.sigma(0.2)(W(z) + A(x))
        return self.W[-1](z) + self.A[-1](x)

    def transport(self, x):
        assert x.requires_grad
        (out,) = autograd.grad(self.forward(x), x, create_graph=True, only_inputs=True,
                               grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float())
        return out

    def clamp_w(self):
        if self.softplus_W_kernels:
            return
        for w in self.W:
            w.weight.data = w.weight.data.clamp(min=0)

    def penalize_w(self):
        return self.fnorm_penalty * sum(map(lambda x: torch.nn.functional.relu(-x.weight).norm(), self.W))


def uniform_init(b=0.1, a=0.0):
    def init(*args):
        return torch.nn.init.uniform_(*args, a=a, b=b)
    return init


def compute_loss_g(f, g, source, transport=None):
    if transport is None:
        transport = g.transport(source)
    return f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)


def compute_loss_f(f, g, source, target, transport=None):
    if transport is None:
        transport = g.transport(source)
    return -f(transport) + f(target)


def mmd_distance(x, y, gamma):
    return rbf_kernel(x, x, gamma).mean() + rbf_kernel(y, y, gamma).mean() - 2 * rbf_kernel(x, y, gamma).mean()


def compute_scalar_mmd(target, transport, gammas=(2, 1, 0.5, 0.1, 0.01, 0.005)):
    def safe(g):
        try:
            return mmd_distance(target, transport, g)
        except ValueError:
            return np.nan
    return np.mean([safe(g) for g in gammas])
