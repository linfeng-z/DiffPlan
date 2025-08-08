import torch
from torch import nn, autograd

from diffplan.utils.deq_jacobian import jac_loss_estimate


class DEQLayer(nn.Module):
    """
    DEQ Fixed-Point Layer
    """

    def __init__(
        self, args, layer, solver_fwd, solver_bwd, jac_reg=False, jac_log=False
    ):
        super().__init__()
        self.args = args

        self.f = layer

        # > separate forward and backward solver
        self.solver_fwd = solver_fwd
        self.solver_bwd = solver_bwd

        self.jac_reg = jac_reg
        self.jac_log = jac_log

        self.residuals_forward = None
        self.residuals_backward = None

    def forward(self, x, z0=None):
        # > input init z if necessary; otherwise use zeros
        z_init = torch.zeros_like(x) if z0 is None else z0

        # > compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.residuals_forward = self.solver_fwd(
                f=lambda _z: self.f(_z, x),
                x0=z_init,
                # (forward fixed-point iteration args)
                tol=self.args.deq_fwd_tol,
                max_iter=self.args.deq_fwd_max_iter,
                # (only useful for Anderson solver)
                m=self.args.deq_fwd_anderson_m,
                beta=self.args.deq_fwd_anderson_beta,
            )
        z = self.f(z, x)

        # > set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        # > Optional: add Jacobian regularization, or only logging
        # jac_loss = jac_loss_estimate(f0, z0, vecs=1) if self.jac_reg else None
        if (
            self.training
        ):  # > only compute in training mode; or some grad may be missing
            if self.jac_reg:
                jac_loss = jac_loss_estimate(f0, z0, vecs=1)
            elif (
                self.jac_log
            ):  # only logging; use no_grad to save memory from tracking grads
                with torch.no_grad():
                    jac_loss = jac_loss_estimate(f0, z0, vecs=1)
            else:
                jac_loss = None
        else:
            jac_loss = None

        def backward_hook(grad):
            g, self.residuals_backward = self.solver_bwd(
                f=lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                x0=grad,
                # (backward fixed-point iteration args)
                tol=self.args.deq_bwd_tol,
                max_iter=self.args.deq_bwd_max_iter,
                # only useful for Anderson solver
                m=self.args.deq_bwd_anderson_m,
                beta=self.args.deq_bwd_anderson_beta,
            )
            return g

        # > check if requiring grad since for inference `torch.no_grad()` is used
        if z.requires_grad:
            z.register_hook(backward_hook)

        return z, jac_loss  # noqa


def forward_iteration(f, x0, max_iter=50, tol=1e-2, m=None, beta=None):
    """Naive forward fixed-point iteration"""
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if res[-1] < tol:
            break
    return f0, res


def anderson(f, x0, max_iter=50, tol=1e-2, beta=1.0, m=5, lam=1e-4):
    """Anderson acceleration for fixed point iteration."""
    # TODO change to support both 2D and 1D spatial dimensions
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )

        # > Note (torch.solve deprecated): (1) only one return (no LU), (2) swap A and B input
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
            :, 1 : n + 1, 0
        ]  # (bsz x n)

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append(
            (F[:, k % m] - X[:, k % m]).norm().item()
            / (1e-5 + F[:, k % m].norm().item())
        )
        if res[-1] < tol:
            break

    return X[:, k % m].view_as(x0), res


def anderson_nd(f, x0, max_iter=50, tol=1e-2, beta=1.0, m=5, lam=1e-4):
    """
    Anderson acceleration for fixed point iteration.
    Note: compatible with any dimensional tensor input
    TODO implement equivariance-compatible version!
    """
    bsz = x0.size(0)
    x_flat, f_flat = x0.view(bsz, -1), f(x0).view(bsz, -1)
    x_size = list(x_flat.size())
    x_size.insert(1, m)
    f_size = list(f_flat.size())
    f_size.insert(1, m)
    X = torch.zeros(*x_size, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(*f_size, dtype=x0.dtype, device=x0.device)

    X[:, 0], F[:, 0] = x_flat, f_flat
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )

        # > Note (torch.solve deprecated): (1) only one return (no LU), (2) swap A and B input
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
            :, 1 : n + 1, 0
        ]  # (bsz x n)

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append(
            (F[:, k % m] - X[:, k % m]).norm().item()
            / (1e-5 + F[:, k % m].norm().item())
        )
        if res[-1] < tol:
            break

    return X[:, k % m].view_as(x0), res


def anderson_1d(f, x0, max_iter=50, tol=1e-2, beta=1.0, m=5, lam=1e-4):
    """Anderson acceleration for fixed point iteration."""
    bsz, d, L = x0.shape
    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )

        # > Note (torch.solve deprecated): (1) only one return (no LU), (2) swap A and B input
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
            :, 1 : n + 1, 0
        ]  # (bsz x n)

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append(
            (F[:, k % m] - X[:, k % m]).norm().item()
            / (1e-5 + F[:, k % m].norm().item())
        )
        if res[-1] < tol:
            break

    return X[:, k % m].view_as(x0), res
