import torch
import torch.nn as nn

from diffplan.modules.helpers import NormalDebugReturn, StandardReturn


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

        self.lstm = nn.LSTMCell(1, self.l_h)

        self.policy = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=num_actions * num_orient,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.sm = nn.Softmax2d()

    def forward(self, map_design, goal_map, debug=False):
        maze_size = map_design.size()[-1]
        X = torch.cat([map_design, goal_map], 1)

        hid = self.hid(X)
        h0 = self.h0(hid).transpose(1, 3).contiguous().view(-1, self.l_h)
        c0 = self.c0(hid).transpose(1, 3).contiguous().view(-1, self.l_h)

        last_h, last_c = h0, c0
        for _ in range(0, self.k - 1):
            h_map = last_h.view(-1, maze_size, maze_size, self.l_h)
            h_map = h_map.transpose(3, 1)
            inp = self.conv(h_map).transpose(1, 3).contiguous().view(-1, 1)

            last_h, last_c = self.lstm(inp, (last_h, last_c))

        hk = last_h.view(-1, maze_size, maze_size, self.l_h).transpose(3, 1)
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
            return StandardReturn(logits, probs)
        else:
            return NormalDebugReturn(logits, probs, hk, None, inp)
