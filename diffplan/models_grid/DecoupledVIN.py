import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from diffplan.modules.helpers import NormalDebugReturn, StandardReturn


class Planner(nn.Module):
    """
    A VIN with decoupled fixed point iteration procedure
    """

    def __init__(self, num_orient, num_actions, args):
        super(Planner, self).__init__()
        self.args = args

        self.num_orient = num_orient
        self.num_actions = num_actions

        self.l_q = args.l_q
        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f
        self.padding_mode = "circular" if "wrap" in args.mechanism else "zeros"

        self._init_layers(args, num_actions, num_orient)

        # TODO recording forward fixed-point iteration loss
        self.residuals_forward = []

    def _init_layers(self, args, num_actions, num_orient):
        self.h = nn.Conv2d(
            in_channels=(num_orient + 1),  # maze map + goal location
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.r = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=num_orient,  # reward per orientation
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.q = nn.Conv2d(
            in_channels=num_orient,
            out_channels=self.l_q * num_orient,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )

        self.policy = nn.Conv2d(
            in_channels=self.l_q * num_orient,
            out_channels=num_actions * num_orient,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.w = Parameter(
            torch.zeros(self.l_q * num_orient, num_orient, self.f, self.f),
            requires_grad=True,
        )

        self.sm = nn.Softmax2d()

    def cat_pad(self, r, v, padding=0):
        out = torch.cat([r, v], 1)
        padding_mode = "circular" if self.padding_mode == "circular" else "constant"
        if padding != 0:
            out = F.pad(out, (padding,) * 4, mode=padding_mode)
        return out

    def forward(self, maze_map, goal_map, debug=False):
        maze_size = maze_map.size()[-1]
        x = torch.cat([maze_map, goal_map], 1)

        h = self.h(x)
        r = self.r(h)
        q = self.q(r)

        q, v = self._value_iterate(maze_size, q, r)

        logits, probs = self._process_value(maze_size, q)

        if not debug:
            return StandardReturn(logits, probs)
        else:
            return NormalDebugReturn(logits, probs, q, v, r)

    def _process_value(self, maze_size, q):
        logits = self.policy(q)

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

        return logits, probs

    def _value_iterate(self, maze_size, q, r):
        # TODO update VI
        q = q.view(-1, self.num_orient, self.l_q, maze_size, maze_size)
        v, _ = torch.max(q, dim=2, keepdim=True)
        v = v.view(-1, self.num_orient, maze_size, maze_size)

        for _ in range(0, self.k - 1):
            v = self._iterate(maze_size, r, v)

        q = F.conv2d(
            self.cat_pad(r, v, padding=int((self.f - 1.0) / 2)),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
        )

        return q, v

    def _iterate(self, maze_size, r, v):
        q = F.conv2d(
            self.cat_pad(r, v, padding=int((self.f - 1.0) / 2)),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
        )
        q = q.view(-1, self.num_orient, self.l_q, maze_size, maze_size)
        v, _ = torch.max(q, dim=2)
        v = v.view(-1, self.num_orient, maze_size, maze_size)
        return v
