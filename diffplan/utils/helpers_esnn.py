import escnn.group
import numpy as np
import torch
from escnn import nn as esnn
from torch import nn


def get_repr(var, g_space: escnn.gspaces.GSpace, cfg, add_spatial=False):
    """
    Get repr for all spaces
    """

    group = g_space.fibergroup

    if cfg.group.startswith("c"):
        std_repr = g_space.irrep(1)
    elif cfg.group.startswith("d"):
        std_repr = g_space.irrep(1, 1)
    else:
        raise ValueError

    if var == "location":
        reprs = [std_repr]

    elif var == "goal":
        reprs = [g_space.trivial_repr]

    elif var == "obstacle":
        reprs = [g_space.trivial_repr]

    elif var == "value":
        # Note: Representation types could be made configurable in future
        if cfg.v_repr == "regular":
            reprs = [g_space.regular_repr]
        elif cfg.v_repr == "trivial":
            reprs = [g_space.trivial_repr]
        else:
            raise ValueError

    elif var == "latent":
        latent_num = get_latent_num(cfg, g_space=g_space, h_dim=cfg.l_h)
        reprs = [g_space.regular_repr] * latent_num

    elif var == "q_map":
        latent_num = get_latent_num(cfg, g_space=g_space, h_dim=cfg.l_q)
        reprs = [g_space.regular_repr] * latent_num
        if add_spatial:
            reprs = [std_repr] + reprs

    elif var == "trivial":
        reprs = [g_space.trivial_repr]

    elif var == "regular":
        reprs = [g_space.regular_repr]

    elif var == "img-encoded":
        # if cfg.group == 'c4':
        #     _repr = g_space.regular_repr
        # elif cfg.group == 'd4':
        #     _repr = g_space.quotient_repr((0, 1))
        # elif cfg.group == 'c8':
        #     # _repr = g_space.quotient_repr(2)
        #     raise NotImplementedError
        # elif cfg.group == 'd8':
        #     # _repr = g_space.quotient_repr((0, 2))
        #     # TODO experiment quotient + restrict representation
        #     d8_group = g_space.fibergroup
        #     _repr = d8_group.restrict_representation((None, 4), d8_group.quotient_representation((0, 2)))
        # else:
        #     raise ValueError

        # TODO hardcode regular repr - after using new sym induction encoder
        _repr = g_space.regular_repr

        reprs = [
            _repr
        ] * cfg.visual_feat  # TODO current default = 128, that's probably too much

    else:
        raise ValueError

    return reprs


def get_group(g_name):
    # 2D discrete subgroups
    if g_name.startswith("c") or g_name.startswith("d"):
        # dimensionality = 2
        rot_num = int(g_name[1:])
        enable_reflection = "d" in g_name  # for dihedral group
        group_size = rot_num if not enable_reflection else (rot_num * 2)

        if not enable_reflection:
            group = escnn.group.cyclic_group(N=rot_num)
        else:
            group = escnn.group.dihedral_group(N=rot_num)

    # 3D discrete subgroups
    elif g_name in ["ico", "full_ico", "octa", "full_octa"]:
        dimensionality = 3
        enable_reflection = g_name.startswith("full")

        name2group = {
            "ico": escnn.group.ico_group(),
            "full_ico": escnn.group.full_ico_group(),
            "octa": escnn.group.octa_group(),
            "full_octa": escnn.group.full_octa_group(),
        }
        group = name2group[g_name]

    else:
        raise ValueError

    return group


def get_latent_num(cfg, g_space, h_dim, h_repr=None, multiply_repr_size=False):
    if h_repr is None:
        h_repr = cfg.latent_repr

    if h_repr == "regular":
        if cfg.latent_dim_factor == "linear":
            # This keeps the same latent size, but equivariant methods have less learnable parameters
            h_dim = h_dim // g_space.regular_repr.size

        elif cfg.latent_dim_factor == "sqrt":
            # TODO This option uses sqrt(size) to keep same # of free parameters; fixed: divided then round
            # h_dim = h_dim // int(np.sqrt(g_space.regular_repr.size))
            h_dim = int(h_dim / np.sqrt(g_space.regular_repr.size)) + 1

        elif cfg.latent_dim_factor == "sqrt-1.2x":
            h_dim = int(1.2 * h_dim / np.sqrt(g_space.regular_repr.size))

        elif cfg.latent_dim_factor == "sqrt-1.5x":
            h_dim = int(1.5 * h_dim / np.sqrt(g_space.regular_repr.size))

        elif cfg.latent_dim_factor == "const":
            h_dim = h_dim

        else:
            raise ValueError

        repr_size = g_space.regular_repr.size

    elif h_repr == "trivial":
        h_dim = h_dim
        repr_size = 1

    else:
        raise NotImplementedError("Unsupported latent space representation")

    return h_dim if not multiply_repr_size else h_dim * repr_size


def sym_mlp(g_space, in_field, out_field, h_num, act_fn=esnn.ELU):
    """
    Return an equivariant MLP using equivariant 1x1 convolution

    """
    if isinstance(h_num, int):
        h_num = [h_num, h_num]

    # Hidden space
    h_reprs = [d * [g_space.regular_repr] for d in h_num]
    h_field = [g_space.type(*h_repr) for h_repr in h_reprs]

    # NOTE: use one hidden layer
    return esnn.SequentialModule(
        esnn.Linear(in_field, h_field[0]),
        esnn.IIDBatchNorm1d(h_field[0]),
        # act_fn(h_field[0]),
        # esnn.Linear(h_field[0], h_field[0]),
        # esnn.IIDBatchNorm1d(h_field[1]),
        # act_fn(h_field[1]),
        # esnn.Linear(h_field[1], out_field),
        # esnn.IIDBatchNorm1d(h_field[0]),
        act_fn(h_field[0]),
        esnn.Linear(h_field[0], out_field),
    )


def sym_enc(cfg, g_space, in_field, out_field, use_state=True):
    if use_state:
        h_num = get_latent_num(cfg, g_space=g_space, h_dim=cfg.enc_dim)
        h_repr = h_num * [g_space.regular_repr]
        h_field = g_space.type(*h_repr)

        layers = [
            esnn.Linear(in_field, h_field),
            esnn.ELU(h_field),
            esnn.Linear(h_field, out_field),
        ]

    else:
        raise ValueError

    return esnn.SequentialModule(*layers)


class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0)


def _get_sym_out_shape(in_shape, layers, in_field):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    x = esnn.GeometricTensor(x, in_field)
    return (
        (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x)
        .tensor.squeeze(0)
        .shape
    )
