import torch
import torch.nn as nn

from diffplan.modules.helpers import (
    NormalDebugReturn,
    get_deq_layer,
    StandardReturnWithAuxInfo,
)


class Planner(nn.Module):
    """
    Implementation of the Gated Path Planning Network.
    """

    def __init__(self, num_orient, num_actions, args):
        super(Planner, self).__init__()

        self.num_orient = num_orient
        self.num_actions = num_actions

        # Note: l_q was not used in this official code! - all hidden LSTM layers use l_h
        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f
        self.padding_mode = "circular" if "wrap" in args.mechanism else "zeros"

        self.input_inject = args.input_inject
        print(f"> Note: input inject = {args.input_inject}")

        self.hid = nn.Conv2d(
            in_channels=(num_orient + 1),  # maze map + goal location
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.h0 = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.c0 = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.conv = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=1,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=True,
        )

        # > for input injection, as in VIN
        self.r = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=num_orient,  # reward per orientation
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            padding_mode=self.padding_mode,
            bias=False,
        )

        self.gru = nn.GRUCell(1, self.l_h)

        self.policy = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=num_actions * num_orient,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.sm = nn.Softmax2d()

        # > init the VI layer and DEQ layer
        self.vi_layer = GruVILayer(
            args=args, num_orient=num_orient, conv=self.conv, gru=self.gru
        )
        self.deq_layer = get_deq_layer(args, self.vi_layer)

    @property
    def residuals_forward(self):
        return self.deq_layer.residuals_forward

    @property
    def residuals_backward(self):
        return self.deq_layer.residuals_backward

    def forward(self, map_design, goal_map, debug=False):
        maze_size = map_design.size()[-1]
        x = torch.cat([map_design, goal_map], 1)

        hid = self.hid(x)
        h0 = self.h0(hid).transpose(1, 3).contiguous().view(-1, self.l_h)

        # > prepare for input inject for DEQ layer
        h_inject = self.r(hid)
        h_inject = h_inject.transpose(1, 3).contiguous().view(-1, 1)

        # > run DEQ layer
        h_out, jac_loss = self.deq_layer(x=h_inject, z0=h0)
        info = {"jac_loss": jac_loss}

        # FIXME check this reshaping
        # hk = h_out.view(-1, maze_size, maze_size, self.l_h).transpose(3, 1)
        hk = h_out.view(-1, self.l_h, maze_size, maze_size)
        logits = self.policy(hk)

        # Normalize over actions
        logits = logits.view(-1, self.num_actions, maze_size, maze_size)
        probs = self.sm(logits)

        # Reshape to output dimensions
        logits = logits.view(
            -1, self.num_orient, self.num_actions, maze_size, maze_size
        )
        probs = probs.view(-1, self.num_orient, self.num_actions, maze_size, maze_size)
        logits = torch.transpose(logits, 1, 2).contiguous()
        probs = torch.transpose(probs, 1, 2).contiguous()

        if not debug:
            return StandardReturnWithAuxInfo(logits, probs, info)
        else:
            return NormalDebugReturn(logits, probs, hk, None, None)


class GruVILayer(nn.Module):
    def __init__(self, args, num_orient, conv, gru, input_inject=True):
        super().__init__()
        self.args = args
        self.num_orient = num_orient

        self.l_h = args.l_h

        # > Init network
        self.conv = conv
        self.gru = gru
        self.input_inject = input_inject

    def forward(self, z_iterate, x_inject):
        """
        A single step of value iteration (Q = R + P * V, V = max_a Q)
        Note: must take input (input inject), to make the fixed-point input-dependent
        Args:
            z_iterate: (intermediate vars) h & c from RNN/LSTM
            x_inject: (fixed input) from reward
        """

        if self.input_inject:
            h_in = x_inject
        else:
            # > use previous h as input
            maze_size = self.args.maze_size
            h_map = z_iterate.view(-1, maze_size, maze_size, self.l_h)
            h_map = h_map.transpose(3, 1)
            h_in = self.conv(h_map).transpose(1, 3).contiguous().view(-1, 1)

        # z_next = self.gru(x_inject, z_iterate)
        z_next = self.gru(h_in, z_iterate)
        # > input (x_inject): (batch_size * m * m, 1), e.g.: (32*15*15=7200, 1)
        # > hx (z_iterate): (batch_size * m * m, 150), e.g.: (32*15*15=7200, 150)

        return z_next
