import time
from collections import namedtuple

from diffplan.modules.DEQ import forward_iteration, DEQLayer, anderson_1d, anderson_nd

StandardReturn = namedtuple(typename="StandardReturn", field_names=["logits", "probs"])

StandardReturnWithAuxInfo = namedtuple(
    typename="StandardReturnWithAuxLoss", field_names=["logits", "probs", "info"]
)

NormalDebugReturn = namedtuple(
    typename="NormalDebugReturn", field_names=["logits", "probs", "q", "v", "r"]
)

EquivariantDebugReturn = namedtuple(
    typename="EquivariantDebugReturn",
    field_names=["logits", "probs", "logits_geo", "q_geo", "v_geo", "r_geo"],
)

TransformedOutput = namedtuple(
    typename="TransformedOutput",
    field_names=[
        "x_geo",
        "x_geo_gx",
        "q_geo",
        "r_geo",
        "v_geo",
        "logits_geo",
        "logits_raw",
        "q_geo_fgx",
        "r_geo_fgx",
        "v_geo_fgx",
        "logits_geo_fgx",
        "logits_fgx",
        "q_geo_gfx",
        "r_geo_gfx",
        "v_geo_gfx",
        "logits_geo_gfx",
        "logits_gfx",
    ],
)


def get_solver(solver_name, one_d=False):
    # > Define solver
    if solver_name == "anderson":
        # solver = anderson
        if not one_d:
            # solver = anderson
            solver = anderson_nd
        else:
            solver = anderson_1d

    elif solver_name == "forward":
        solver = forward_iteration

    # Note: Broyden solver from DEQ library could be integrated in future
    # elif self.args.solver == 'broyden':
    #     solver = broyden

    else:
        raise ValueError

    return solver


def get_deq_layer(args, vi_layer):
    """get DEQ layer, with separate forward and backward solvers"""
    solver_fwd = get_solver(args.solver_fwd, one_d="SPT" in args.model)
    solver_bwd = get_solver(args.solver_bwd, one_d="SPT" in args.model)

    deq_layer = DEQLayer(
        args=args,
        layer=vi_layer,
        solver_fwd=solver_fwd,
        solver_bwd=solver_bwd,
        jac_reg=args.jacobian_reg,
        jac_log=args.jacobian_log,
    )

    return deq_layer


class Timer:
    def __init__(self, name=None, verbose=False):
        self.name = name if name is not None else "interval"
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.verbose:
            print(f"> ({self.name}) time: {self.interval:.3f}")
