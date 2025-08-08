import torch
import torch.nn as nn

from diffplan.modules.helpers import NormalDebugReturn, StandardReturn
from diffplan.modules.ConvRNN_modules import ConvGRUCell


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
        self.l_q = args.l_q
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

        # TODO for conv version - same as self.conv
        self.q = nn.Conv2d(
            in_channels=num_orient,
            # out_channels=self.l_q * num_orient,
            out_channels=self.l_h * num_orient,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )

        # self.gru = nn.GRUCell(1, self.l_h)
        # TODO ConvGRU version - replace with fully conv version
        self.conv_gru = ConvGRUCell(
            input_size=(args.maze_size, args.maze_size),
            input_dim=1,
            # hidden_dim=self.l_q,
            hidden_dim=self.l_h,
            kernel_size=(self.f, self.f),
            bias=True,
        )

        self.policy = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=num_actions * num_orient,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.sm = nn.Softmax2d()

        # > store forward fixed-point iteration loss
        self.residuals_forward = None

    def forward(self, map_design, goal_map, debug=False):
        maze_size = map_design.size()[-1]
        x = torch.cat([map_design, goal_map], 1)

        # > VIN style preprocessing
        h = self.hid(x)  # [150, 15, 15]
        r = self.r(h)  # [1, 15, 15]
        q = self.q(r)  # [600, 15, 15]

        # > run value iteration computation
        final_h = self.value_iterate(r=r, init_h=q, maze_size=maze_size)
        logits = self.policy(final_h)

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
            return StandardReturn(logits, probs)
        else:
            return NormalDebugReturn(logits, probs, final_h, None, None)

    def value_iterate(self, r, init_h, maze_size):
        if self.input_inject:
            # > with input inject: preprocess using the "reward" conv (as in VIN; note different filter size)
            h_inject = r
        else:
            # > no input inject (GPPN default): go through conv every time
            h_inject = None

        last_h = init_h
        self.residuals_forward = []

        for _ in range(0, self.k - 1):
            prev_h = last_h.detach().clone()

            last_h = self._iterate(h_inject, last_h, maze_size)

            res = (last_h - prev_h).norm().item() / (1e-5 + last_h.norm().item())
            self.residuals_forward.append(res)

        return last_h

    def _iterate(self, h_inject, last_h, maze_size):
        """
        A step of value iteration using recurrent units
        Input injection is used to inject to every step
        """

        if self.input_inject:
            h_in = h_inject
        else:
            # > use previous h as input
            h_in = self.conv(last_h)  # for conv version, no need to flatten

        last_h = self.conv_gru(h_in, last_h)
        # > input (h_in): (batch_size * m * m, 1), e.g.: (32*15*15=7200, 1)
        # > hx (last_h): (batch_size * m * m, 150), e.g.: (32*15*15=7200, 150)

        return last_h
