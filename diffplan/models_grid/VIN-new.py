import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from diffplan.modules.helpers import NormalDebugReturn, StandardReturn


class Planner(nn.Module):
    """
    Implementation of the Value Iteration Network.
    """

    def __init__(self, args=None):
        super(Planner, self).__init__()

        self.num_orient = 1
        self.num_actions = 4

        self.l_h = args.l_h
        self.l_q = args.l_q
        self.k = args.k
        self.f = 3
        self.padding_mode = "zeros"
        # self.l_q = args.l_q
        # self.l_h = args.l_h
        # self.k = args.k
        # self.f = args.f
        # self.padding_mode = 'circular' if 'wrap' in args.mechanism else 'zeros'

        self.h = nn.Conv2d(
            in_channels=(self.num_orient + 1),  # maze map + goal location
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.r = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.num_orient,  # reward per orientation
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.q = nn.Conv2d(
            in_channels=self.num_orient,
            out_channels=self.l_q * self.num_orient,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )

        self.policy = nn.Conv2d(
            in_channels=self.l_q * self.num_orient,
            out_channels=self.num_actions * self.num_orient,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.w = Parameter(
            torch.zeros(self.l_q * self.num_orient, self.num_orient, self.f, self.f),
            requires_grad=True,
        )

        self.sm = nn.Softmax2d()

        # > store forward fixed-point iteration loss
        self.residuals_forward = None

    def cat_pad(self, r, v, padding=0):
        out = torch.cat([r, v], 1)
        padding_mode = "circular" if self.padding_mode == "circular" else "constant"
        if padding != 0:
            out = F.pad(out, (padding,) * 4, mode=padding_mode)
        return out

    def forward(self, map_design, goal_map, debug=False):
        maze_size = map_design.size()[-1]
        X = torch.cat([map_design.unsqueeze(1), goal_map], 1)

        h = self.h(X)
        r = self.r(h)
        q = self.q(r)
        q = q.view(-1, self.num_orient, self.l_q, maze_size, maze_size)
        v, _ = torch.max(q, dim=2, keepdim=True)
        v = v.view(-1, self.num_orient, maze_size, maze_size)

        self.residuals_forward = []
        for _ in range(0, self.k - 1):
            v_prev = v.detach().clone()

            q = F.conv2d(
                self.cat_pad(r, v, padding=int((self.f - 1.0) / 2)),
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
            )
            q = q.view(-1, self.num_orient, self.l_q, maze_size, maze_size)
            v, _ = torch.max(q, dim=2)
            v = v.view(-1, self.num_orient, maze_size, maze_size)

            # > compute residuals
            res = (v - v_prev).norm().item() / (1e-5 + v.norm().item())
            self.residuals_forward.append(res)

        q = F.conv2d(
            self.cat_pad(r, v, padding=int((self.f - 1.0) / 2)),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
        )

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

        if not debug:
            return StandardReturn(logits, probs)
        else:
            return NormalDebugReturn(logits, probs, q, v, r)
