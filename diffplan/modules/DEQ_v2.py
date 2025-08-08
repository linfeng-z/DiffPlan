import torch
from torch import autograd
from torch import nn

from diffplan.utils.deq_jacobian import jac_loss_estimate


class DEQModel(nn.Module):
    def __init__(self, layer, solver, **kwargs):
        super().__init__()

        self.f = layer
        self.solver = solver
        self.solver_kwargs = kwargs

        self.residuals_forward = None
        self.residuals_backward = None
        self.jac_loss = None

    def forward(self, x, **f_kwargs):
        # Forward pass
        with torch.no_grad():
            z0 = torch.zeros_like(x)

            # See step 2 above
            solved_forward = self.solver(
                f=lambda z: self.f(z, x, **f_kwargs), x0=z0, **self.solver_kwargs
            )
            z_star = solved_forward["result"]
            self.residuals_forward = solved_forward["rel_trace"]

            new_z_star = z_star

        # (Prepare for) Backward pass, see step 3 above
        if self.training:
            new_z_star = self.f(z_star.requires_grad_(), x, **f_kwargs)

            # Jacobian-related computations, see additional step above. For instance:
            jac_loss = jac_loss_estimate(new_z_star, z_star, vecs=1)

            def backward_hook(grad):
                # if self.hook is not None:
                #     self.hook.remove()
                #     torch.cuda.synchronize()  # To avoid infinite recursion

                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                solved_backward = self.solver(
                    f=lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[
                        0
                    ]
                    + grad,
                    x0=torch.zeros_like(grad),
                    **self.solver_kwargs
                )
                new_grad = solved_backward["result"]
                self.residuals_backward = solved_backward["rel_trace"]

                return new_grad

            new_z_star.register_hook(backward_hook)
            # self.hook = new_z_star.register_hook(backward_hook)

        else:
            jac_loss = None

        return new_z_star, jac_loss
