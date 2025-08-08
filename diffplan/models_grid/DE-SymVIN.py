import numpy as np
import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as e2nn
from e2cnn.nn import GeometricTensor

from diffplan.modules.helpers import (
    EquivariantDebugReturn,
    StandardReturn,
    StandardReturnWithAuxInfo,
    get_deq_layer,
)
from diffplan.modules.pooling import GroupReducedMaxPooling


class Planner(nn.Module):
    """
    Implementation of the Symmetric Value Iteration Network, with DEQ layer
    """

    def __init__(self, num_orient, num_actions, args):
        super(Planner, self).__init__()
        self.args = args

        self.num_orient = num_orient
        self.num_actions = num_actions
        self.enable_e2_pi = args.enable_e2_pi

        self.l_q = args.l_q
        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f
        self.padding_mode = "circular" if "wrap" in args.mechanism else "zeros"

        self.group = args.group
        # > Init symmetry group space
        assert self.group in ["c2", "c4", "c8", "c16", "d2", "d4", "d8"]
        self.rot_num = int(self.group[1:])
        self.enable_reflection = "d" in self.group  # for dihedral group
        self.group_size = (
            self.rot_num if not self.enable_reflection else (self.rot_num * 2)
        )

        if not self.enable_reflection:
            self.r2_act = gspaces.Rot2dOnR2(N=self.rot_num)
        else:
            self.r2_act = gspaces.FlipRot2dOnR2(N=self.rot_num)

        print("> Group:", self.group)
        print("> Group space:", self.r2_act)

        self._init_layers(args, num_actions, num_orient)

    @property
    def residuals_forward(self):
        return self.deq_layer.residuals_forward

    @property
    def residuals_backward(self):
        return self.deq_layer.residuals_backward

    def _get_repr(self, name):
        name2repr = {
            "trivial": self.r2_act.trivial_repr,
            "regular": self.r2_act.regular_repr,
        }

        # > may also quotient repr for latent layer (same as action output); not verified
        if name == "quotient":
            _, repr_out_pi, _ = self._get_action_repr()
            name2repr.update({"quotient": repr_out_pi})

        return name2repr[name], name2repr[name].size

    def _get_action_repr(self):
        if self.num_actions == 4 and self.rot_num == 4:
            if self.enable_reflection:
                # > quotient out reflections and keep rotations
                repr_out_pi = self.r2_act.quotient_repr(subgroup_id=(0, 1))
            else:
                repr_out_pi = self.r2_act.regular_repr
            reprs_out_pi = 1 * [repr_out_pi]

        elif self.num_actions == 4 and self.rot_num in [8, 16]:
            if self.enable_reflection:
                repr_out_pi = self.r2_act.quotient_repr(
                    subgroup_id=(0, self.rot_num // self.num_actions)
                )
            else:
                repr_out_pi = self.r2_act.quotient_repr(
                    subgroup_id=self.rot_num // self.num_actions
                )
            reprs_out_pi = 1 * [repr_out_pi]

        elif self.num_actions == 8 and self.rot_num == 4:
            # > TODO e.g., C4/D8 for 8 actions
            raise NotImplementedError

        elif self.num_actions == 8 and self.rot_num == 8:
            raise NotImplementedError

        else:
            raise ValueError

        field_type = e2nn.FieldType(self.r2_act, reprs_out_pi)

        return reprs_out_pi, repr_out_pi, field_type

    def _init_layers(self, args, num_actions, num_orient):
        """
        1. define fiber repr types
        2. decide sizes of fiber reprs
        3. update conv layer sizes accordingly
        4. define steerable conv layers
        """

        num_in_h = num_orient + 1
        num_out_h = args.l_h  # trivial repr, no division by self.group_size
        num_out_r = num_orient  # reward per orientation
        num_in_q = num_out_r * 2  # concat input for q and r

        if args.divide_by_size:
            # > for regular repr - can choose to divide by group size, to keep intermediate embedding sizes the same
            num_out_q = args.l_q * num_orient // self.group_size
        else:
            # > or don't divide, to keep layers' parameter number the same
            num_out_q = args.l_q * num_orient

        # > Define repr type for steerable conv layers
        # > + Store repr size
        repr_in_h, self.size_repr_in_h = self._get_repr(args.repr_in_h)
        repr_out_h, self.size_repr_out_h = self._get_repr(args.repr_out_h)
        repr_out_r, self.size_repr_out_r = self._get_repr(args.repr_out_r)
        repr_in_q, self.size_repr_in_q = (
            repr_out_r,
            self.size_repr_out_r,
        )  # R out = Q in
        repr_out_q, self.size_repr_out_q = self._get_repr(args.repr_out_q)

        # > only generate repr for action space when
        if self.enable_e2_pi:
            reprs_out_pi, repr_out_pi, self.feat_out_pi = self._get_action_repr()

        self.repr_out_r = repr_out_r

        # > Provide output size
        self.size_out_q = num_out_q * self.group_size
        # TODO fix: no another group size for Q, replace with num_orient instead

        # > Define feature (Note: use trivial repr for input space)
        self.feat_in_h = e2nn.FieldType(self.r2_act, num_in_h * [repr_in_h])
        self.feat_out_h = e2nn.FieldType(self.r2_act, num_out_h * [repr_out_h])
        self.feat_out_r = e2nn.FieldType(self.r2_act, num_out_r * [repr_out_r])
        self.feat_in_q = e2nn.FieldType(self.r2_act, num_in_q * [repr_in_q])
        self.feat_out_q = e2nn.FieldType(self.r2_act, num_out_q * [repr_out_q])

        # > Define E(2) Conv
        self.h_conv = e2nn.R2Conv(
            self.feat_in_h,
            self.feat_out_h,
            kernel_size=3,
            padding=1,
            padding_mode=self.padding_mode,
            bias=True,
        )
        self.r_conv = e2nn.R2Conv(
            self.feat_out_h, self.feat_out_r, kernel_size=1, padding=0, bias=False
        )
        self.q_conv = e2nn.R2Conv(
            self.feat_in_q,
            self.feat_out_q,
            kernel_size=self.f,
            padding=int((self.f - 1) // 2),
            padding_mode=self.padding_mode,
            bias=False,
        )

        # > Use customized (group channel wise) max pooling
        self.max_pool = GroupReducedMaxPooling(
            in_type=self.feat_out_q, out_repr=repr_out_r
        )

        # > Output policy layer
        if self.enable_e2_pi:
            self.pi_r2conv = e2nn.R2Conv(
                self.feat_out_q, self.feat_out_pi, kernel_size=1, padding=0, bias=False
            )
            print(
                f"> Enable E2 policy! Feature type: {self.feat_out_pi}, fiber group: {self.feat_out_pi.fibergroup}"
            )
        else:
            self.pi_conv2d = nn.Conv2d(
                self.size_out_q,
                num_actions,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=False,
            )

        self.sm = nn.Softmax2d()  # nn.Softmax(dim=1)

        # > Define VI layer
        self.vi_layer = VILayer(
            args=self.args,
            feat_out_r=self.feat_out_r,
            q_conv=self.q_conv,
            max_pool=self.max_pool,
        )

        # > Define Deep Equilibrium Model for solving fixed point - fixed-point layer
        self.deq_layer = get_deq_layer(args, self.vi_layer)

    def deq_raw(self, x):
        deq = self.deq_layer.double()
        z, _ = deq(x)
        return z

    def forward(self, map_design, goal_map, debug=False):
        batch_size = map_design.size(0)
        maze_size = map_design.size(-1)

        x = torch.cat([map_design, goal_map], 1)
        device = x.device

        x_geo = e2nn.GeometricTensor(x, self.feat_in_h)

        # > value iteration
        q_geo, r_geo, v_geo, jac_loss = self._value_iterate(x_geo, device)
        info = {"jac_loss": jac_loss}

        # > extract action from policy
        logits, logits_geo = self._value2logits(q_geo)
        logits, probs = self._process_logits(logits, batch_size, maze_size)

        if not debug:
            return StandardReturnWithAuxInfo(logits, probs, info)
        else:
            return EquivariantDebugReturn(
                logits, probs, logits_geo, q_geo, v_geo, r_geo
            )

    def _value_iterate(self, x_geo, device):
        h_trivial = self.h_conv(x_geo)
        r_geo = self.r_conv(h_trivial)

        # > put unwrapped PyTorch tensor of r; v initialized inside
        r = r_geo.tensor
        # > DEQ layer
        v_out, jac_loss = self.deq_layer(r)

        # > wrap again
        v_geo = e2nn.GeometricTensor(v_out, self.feat_out_r)
        # > additional execution
        rv_geo = e2nn.tensor_directsum([r_geo, v_geo])
        q_geo = self.q_conv(rv_geo)

        return q_geo, r_geo, v_geo, jac_loss

    def _value2logits(self, q_geo):
        # > Use equivariant policy or not (normal 2D conv)
        if self.enable_e2_pi:
            logits_geo = self.pi_r2conv(q_geo)
            logits = logits_geo.tensor
        else:
            logits = self.pi_conv2d(q_geo.tensor)
            logits_geo = None  # `e2nn.GeometricTensor(logits, self.feat_out_pi)`

        return logits, logits_geo

    def _process_logits(self, logits, batch_size, maze_size):
        logits = logits.view(
            batch_size, self.num_orient, self.num_actions, maze_size, maze_size
        )

        # > Reshape for probs & Normalize over actions
        logits_reshape = logits.view(-1, self.num_actions, maze_size, maze_size)
        probs = self.sm(logits_reshape)

        # > Note: group repr & action space need to match (be compatible group action: G x A -> A)
        # > Reshape to output dimensions
        probs = probs.view(
            batch_size, self.num_orient, self.num_actions, maze_size, maze_size
        )
        logits = torch.transpose(logits, 1, 2).contiguous()
        probs = torch.transpose(probs, 1, 2).contiguous()

        return StandardReturn(logits, probs)

    def get_equivariance_error(
        self,
        map_design,
        goal_map,
        rand_input=False,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ):
        batch_size = map_design.size(0)
        maze_size = map_design.size(-1)
        device = map_design.device

        if not rand_input:
            x = torch.cat([map_design, goal_map], 1)
        else:
            x = torch.randn(batch_size, 2, maze_size, maze_size)

        x_geo = e2nn.GeometricTensor(x, self.feat_in_h)

        # > forward f(x)
        q_geo, r_geo, v_geo, _ = self._value_iterate(x_geo, device)
        _, logits_geo = self._value2logits(q_geo)

        # > compute f(g.x) and g.f(x)
        # > Note: e2nn.GroupPooling uses .transform_fibers(e), while .transform(e) should be used here
        ee_dict = {}
        for element in self.r2_act.fibergroup.testing_elements():
            # > f(g.x)
            x_geo_gx = x_geo.transform(element)
            q_geo_fgx, r_geo_fgx, v_geo_fgx, _ = self._value_iterate(x_geo_gx, device)
            _, logits_geo_fgx = self._value2logits(q_geo_fgx)

            # > g.f(x)
            q_geo_gfx, r_geo_gfx, v_geo_gfx, logits_geo_gfx = (
                q_geo.transform(element),
                r_geo.transform(element),
                v_geo.transform(element),
                logits_geo.transform(element),
            )

            q_err = (q_geo_fgx.tensor - q_geo_gfx.tensor).detach().numpy()
            r_err = (r_geo_fgx.tensor - r_geo_gfx.tensor).detach().numpy()
            v_err = (v_geo_fgx.tensor - v_geo_gfx.tensor).detach().numpy()
            logits_err = (
                (logits_geo_fgx.tensor - logits_geo_gfx.tensor).detach().numpy()
            )

            q_err = np.abs(q_err).reshape(-1)
            r_err = np.abs(r_err).reshape(-1)
            v_err = np.abs(v_err).reshape(-1)
            logits_err = np.abs(logits_err).reshape(-1)

            print(
                f"EEs of element {element}:",
                q_err.mean(),
                r_err.mean(),
                v_err.mean(),
                logits_err.mean(),
            )

            ee_dict[element] = {
                "q": q_err.mean(),
                "r": r_err.mean(),
                "v": v_err.mean(),
                "logits": logits_err.mean(),
            }

            assert torch.allclose(
                logits_geo_fgx.tensor, logits_geo_gfx.tensor, atol=atol, rtol=rtol
            ), f"EE of element {element} is too high: {logits_err.mean()}"

        return ee_dict


class VILayer(nn.Module):
    def __init__(self, args, feat_out_r, q_conv, max_pool):
        super().__init__()
        self.args = args

        self.feat_out_r = feat_out_r
        self.q_conv = q_conv
        self.max_pool = max_pool

    def forward(self, v, r):
        """
        Note: convert to GeometricTensor internally
        """

        v_geo = GeometricTensor(v, type=self.feat_out_r)
        r_geo = GeometricTensor(r, type=self.feat_out_r)

        # > concat and convolve with "transition probability"
        rv_geo = e2nn.tensor_directsum([r_geo, v_geo])
        q_geo = self.q_conv(rv_geo)

        # > max over group channel
        # > Q: batch_size x (|G| * #repr) x width x height
        # > V: batch_size x (|G| * 1) x width x height
        v_geo = self.max_pool(q_geo)

        v = v_geo.tensor

        return v


class GeoVILayer(nn.Module):
    def __init__(self, args, repr_out_r, q_conv, max_pool):
        super().__init__()
        self.args = args

        self.repr_out_r = repr_out_r
        self.q_conv = q_conv
        self.max_pool = max_pool

    def forward(self, v_geo, r_geo):
        """
        Note: take geometric tensors as input and output
        """

        # > concat and convolve with "transition probability"
        rv_geo = e2nn.tensor_directsum([r_geo, v_geo])
        q_geo = self.q_conv(rv_geo)

        # > max over group channel
        # > Q: batch_size x (|G| * #repr) x width x height
        # > V: batch_size x (|G| * 1) x width x height
        v_geo = self.max_pool(q_geo)

        return v_geo
