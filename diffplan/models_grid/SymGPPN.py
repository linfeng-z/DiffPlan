import e2cnn
import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as e2nn

from diffplan.modules.helpers import EquivariantDebugReturn, StandardReturn


class Planner(nn.Module):
    """
    Implementation of the Value Iteration Network.
    """

    def __init__(self, num_orient, num_actions, args):
        super(Planner, self).__init__()

        self.num_orient = num_orient
        self.num_actions = num_actions

        self.l_q = args.l_q
        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f
        self.group = args.group
        self.padding_mode = "circular" if "wrap" in args.mechanism else "zeros"

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

        #     self._init_layers(args, num_actions, num_orient)
        #
        # def _init_layers(self, args, num_actions, num_orient):
        filter_size = 3
        num_input_channel_h = 2

        # num_output_channel_h = self.l_h // self.group_size
        num_output_channel_h = self.l_h  # fixme check divid by group or not?

        self.feat_type_in_h = e2cnn.nn.FieldType(
            self.r2_act, num_input_channel_h * [self.r2_act.trivial_repr]
        )
        self.feat_type_out_h = e2cnn.nn.FieldType(
            self.r2_act, num_output_channel_h * [self.r2_act.regular_repr]
        )
        self.h_conv = e2cnn.nn.R2Conv(
            self.feat_type_in_h,
            self.feat_type_out_h,
            kernel_size=filter_size,
            padding=1,
            padding_mode=self.padding_mode,
            bias=True,
        )

        num_output_channel_r = 1
        filter_size_r = 1
        self.feat_type_out_r = e2cnn.nn.FieldType(
            self.r2_act, num_output_channel_r * [self.r2_act.regular_repr]
        )
        self.r_conv = e2cnn.nn.R2Conv(
            self.feat_type_out_h,
            self.feat_type_out_r,
            kernel_size=filter_size_r,
            padding=0,
            bias=False,
        )

        filter_size_q = self.f

        # num_output_channel_q = self.l_q // self.group_size  # FIXME divide by group size!
        if args.divide_by_size:
            # > for regular repr - can choose to divide by group size, to keep intermediate embedding sizes the same
            num_output_channel_q = args.l_q // self.group_size
        else:
            # > or don't divide, to keep layers' parameter number the same
            num_output_channel_q = args.l_q

        self.feat_type_out_q = e2cnn.nn.FieldType(
            self.r2_act, num_output_channel_q * [self.r2_act.regular_repr]
        )
        # self.feat_type_in_q = e2cnn.nn.FieldType(self.r2_act, num_output_channel_r * 2 * [self.r2_act.regular_repr])

        self.q_conv = e2cnn.nn.R2Conv(
            self.feat_type_out_r,
            self.feat_type_out_q,
            kernel_size=filter_size_q,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )

        # TODO add option to enable or disable equivariant policy layer
        self.enable_e2_pi = args.enable_e2_pi
        # > init repr
        if self.enable_e2_pi:
            reprs_out_pi, repr_out_pi, self.feat_out_pi = self._get_action_repr()
        # > init layer
        if self.enable_e2_pi:
            self.pi_r2conv = e2nn.R2Conv(
                self.feat_type_out_q,
                self.feat_out_pi,
                kernel_size=1,
                padding=0,
                bias=False,
            )
            print(
                f"> Enable E2 policy! Feature type: {self.feat_out_pi}, fiber group: {self.feat_out_pi.fibergroup}"
            )
        else:
            self.size_out_q = num_output_channel_q * self.group_size
            self.pi_conv2d = nn.Conv2d(
                self.size_out_q,
                num_actions,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=False,
            )

        self.LSTM_cnn = E2ConvLSTM(args)

        self.sm = nn.Softmax2d()

    def _get_repr(self, name):
        name2repr = {
            "trivial": self.r2_act.trivial_repr,
            "regular": self.r2_act.regular_repr,
        }

        # > may also quotient repr for latent layer (same as action output); not verified
        if name == "quotient":
            _, repr_out_pi, _ = self._get_action_repr()
            name2repr.update({"quotient": repr_out_pi})

        return name2repr[name]

    def _get_action_repr(self):
        if self.num_actions == 4 and self.rot_num == 4:
            if self.enable_reflection:
                # > quotient out reflections and keep rotations
                repr_out_pi = self.r2_act.quotient_repr(subgroup_id=(0, 1))
            else:
                repr_out_pi = self.r2_act.regular_repr
            reprs_out_pi = 1 * [repr_out_pi]

        elif self.num_actions == 4 and self.rot_num in [8, 16]:
            # > TODO e.g., C8 and D8
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

    def forward(self, map_design, goal_map, debug=False):
        maze_size = map_design.size()[-1]
        x = torch.cat([map_design, goal_map], 1)

        x_geo = e2cnn.nn.GeometricTensor(x, self.feat_type_in_h)
        h_geo = self.h_conv(x_geo)  # [b, 160*4, 15, 15]
        r_geo = self.r_conv(h_geo)  # [b, 1*4, 15, 15]
        q_geo = self.q_conv(r_geo)  # [b, 40(l_q), 15, 15]

        q_rnn = q_geo
        q_cell = q_geo

        for _ in range(0, self.k - 1):
            q_rnn, q_cell = self.LSTM_cnn(r_geo, q_rnn, q_cell)

        q_rnn, q_cell = self.LSTM_cnn(r_geo, q_rnn, q_cell)

        # TODO use equivariant policy layer
        # logits = self.policy_c4(q_rnn.tensor)  # [20, 4, 15, 15]
        logits, logits_geo = self._value2logits(q_geo=q_rnn)

        # Normalize over actions
        logits = logits.view(-1, self.num_actions, maze_size, maze_size)
        probs = self.sm(logits)  # [20, 4, 15, 15]

        # Reshape to output dimensions
        logits = logits.view(
            -1, self.num_orient, self.num_actions, maze_size, maze_size
        )
        probs = probs.view(-1, self.num_orient, self.num_actions, maze_size, maze_size)
        logits = torch.transpose(logits, 1, 2).contiguous()
        probs = torch.transpose(probs, 1, 2).contiguous()

        if not debug:
            return StandardReturn(logits, probs)
        else:
            return EquivariantDebugReturn(logits, probs, logits_geo, q_rnn, None, r_geo)

    def _value2logits(self, q_geo):
        # > Use equivariant policy or not (normal 2D conv)
        if self.enable_e2_pi:
            logits_geo = self.pi_r2conv(q_geo)
            logits = logits_geo.tensor
        else:
            logits = self.pi_conv2d(q_geo.tensor)
            logits_geo = None

        return logits, logits_geo


class E2ConvLSTM(nn.Module):
    """
    Using pytorch, implement a customized LSTM
    """

    def __init__(self, args):
        super(E2ConvLSTM, self).__init__()

        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f
        self.l_q = args.l_q
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

        # input content
        filter_size_q_hid = self.f

        # num_output_channel_hid = self.l_q
        # num_output_channel_hid = self.l_q // self.group_size  # FIXME divide by group size!
        if args.divide_by_size:
            # > for regular repr - can choose to divide by group size, to keep intermediate embedding sizes the same
            num_output_channel_hid = args.l_q // self.group_size
        else:
            # > or don't divide, to keep layers' parameter number the same
            num_output_channel_hid = args.l_q

        self.feat_type_in_out_q_hid = e2cnn.nn.FieldType(
            self.r2_act, num_output_channel_hid * [self.r2_act.regular_repr]
        )
        self.cnn_q_hid_tanh_c4 = e2cnn.nn.R2Conv(
            self.feat_type_in_out_q_hid,
            self.feat_type_in_out_q_hid,
            kernel_size=filter_size_q_hid,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )

        filter_size_q = self.f
        # num_output_channel_q = self.l_q
        # num_output_channel_q = self.l_q // self.group_size  # FIXME divide by group size!
        if args.divide_by_size:
            # > for regular repr - can choose to divide by group size, to keep intermediate embedding sizes the same
            num_output_channel_q = args.l_q // self.group_size
        else:
            # > or don't divide, to keep layers' parameter number the same
            num_output_channel_q = args.l_q

        self.feat_type_out_q = e2cnn.nn.FieldType(
            self.r2_act, num_output_channel_q * [self.r2_act.regular_repr]
        )
        num_output_channel_r = 1
        self.feat_type_out_r = e2cnn.nn.FieldType(
            self.r2_act, num_output_channel_r * [self.r2_act.regular_repr]
        )

        self.cnn_r_hid_tanh_c4 = e2cnn.nn.R2Conv(
            self.feat_type_out_r,
            self.feat_type_out_q,
            kernel_size=filter_size_q,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )

        self.tanh_input = nn.Tanh()

        # input gate
        self.cnn_q_hid_sig = e2cnn.nn.R2Conv(
            self.feat_type_in_out_q_hid,
            self.feat_type_in_out_q_hid,
            kernel_size=filter_size_q_hid,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )
        self.cnn_r_sig = e2cnn.nn.R2Conv(
            self.feat_type_out_r,
            self.feat_type_out_q,
            kernel_size=filter_size_q,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )
        self.sigmoid_input = nn.Sigmoid()

        # forget gate
        self.cnn_q_hid_forget = e2cnn.nn.R2Conv(
            self.feat_type_in_out_q_hid,
            self.feat_type_in_out_q_hid,
            kernel_size=filter_size_q_hid,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )
        self.cnn_r_forget = e2cnn.nn.R2Conv(
            self.feat_type_out_r,
            self.feat_type_out_q,
            kernel_size=filter_size_q,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )
        self.sigmoid_forget = nn.Sigmoid()

        # output gate
        self.cnn_q_hid_out = e2cnn.nn.R2Conv(
            self.feat_type_in_out_q_hid,
            self.feat_type_in_out_q_hid,
            kernel_size=filter_size_q_hid,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )
        self.cnn_r_out = e2cnn.nn.R2Conv(
            self.feat_type_out_r,
            self.feat_type_out_q,
            kernel_size=filter_size_q,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )

        self.sigmoid_output = nn.Sigmoid()

        self.tanh_output = nn.Tanh()

    def input_content(self, r, q):
        r_hid = self.cnn_r_hid_tanh_c4(r)
        q_hid = self.cnn_q_hid_tanh_c4(q)
        hid_plus = r_hid + q_hid
        q_rnn = self.tanh_input(hid_plus.tensor)

        return q_rnn

    def input_gate(self, r, q):
        r_hid = self.cnn_r_sig(r)
        q_hid = self.cnn_q_hid_sig(q)
        hid_plus = r_hid + q_hid
        q_rnn = self.sigmoid_input(hid_plus.tensor)

        return q_rnn

    def forget(self, r, q):
        r_hid = self.cnn_r_forget(r)
        q_hid = self.cnn_q_hid_forget(q)
        hid_plus = r_hid + q_hid
        q_rnn = self.sigmoid_forget(hid_plus.tensor)

        return q_rnn

    def output_gate(self, r, q):
        r_hid = self.cnn_r_out(r)
        q_hid = self.cnn_q_hid_out(q)
        hid_plus = r_hid + q_hid
        q_rnn = self.sigmoid_output(hid_plus.tensor)

        return q_rnn

    def cell_memory_update(self, q_cell, forget_gate, input_cont, input_gate):
        cell_new = forget_gate * q_cell + input_cont * input_gate

        return cell_new

    def forward(self, r, q, q_cell):
        """
        input: r: [1, maze_size, maze_size]
                q: [hid_dim, maze_size, maze_size]
                q_cell: [hid_dim, maze_size, maze_size]
        output: same as q: [hid_dim, maze_size, maze_size]
        """

        # equation 1: forget gate
        forget_gate = self.forget(r, q)

        # equation 2: input gate
        input_cont = self.input_content(r, q)
        input_gate = self.input_gate(r, q)

        # equation 3: cell update
        q_cell_new = self.cell_memory_update(
            q_cell.tensor, forget_gate, input_cont, input_gate
        )

        # equation 4: output gate
        output_g = self.output_gate(r, q)

        # equation 5: hidden state q
        q_new = output_g * self.tanh_output(q_cell_new)

        q_new = e2cnn.nn.GeometricTensor(q_new, self.feat_type_in_out_q_hid)
        q_cell_new = e2cnn.nn.GeometricTensor(q_cell_new, self.feat_type_in_out_q_hid)

        return q_new, q_cell_new
