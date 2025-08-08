import numpy as np
import torch
import torch_geometric.nn
from torch import nn

from diffplan.modules.helpers import StandardReturn
from diffplan.utils import visualize_graph_v


# matplotlib.use("TkAgg")



class GCNPlanningNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        num_orient = 1  # deprecated, used in VIN grid setting
        # discrete (4), continous (1)
        num_actions = 2 if args.cont_action else 4

        self.l_h = args.l_h
        self.l_q = args.l_q
        self.k = args.k

        num_in_h = 2  # xy position
        if args.task == "Habitat":
            # override everything below
            num_in_h += args.visual_feat * 4
        elif args.obsv_mode == "full" and args.has_obstacle:
            num_in_h += 2  # goal + obstacle
        elif args.obsv_mode == "full" and not args.has_obstacle:
            num_in_h += 1  # goal
        elif args.obsv_mode == "partial" and args.has_obstacle:
            num_in_h += 6  # goal + obstacle in one-hot
        elif args.obsv_mode == "partial" and not args.has_obstacle:
            num_in_h += 3  # goal in one-hot
        else:
            raise Exception("Wierd things happen here")

        num_out_h = self.l_h
        num_in_r = num_out_h + 2
        num_out_r = num_orient
        num_in_q = num_out_r * 2 + 2  # takes R & V
        num_out_q = self.l_q * num_orient
        num_in_pi = num_out_q + 2
        num_out_pi = num_actions

        self.h = torch_geometric.nn.GCNConv(
            in_channels=num_in_h, out_channels=num_out_h
        )

        # TODO 1x1, linear?
        self.r = torch_geometric.nn.GCNConv(
            in_channels=num_in_r,
            out_channels=num_out_r,
        )

        self.q = torch_geometric.nn.GCNConv(
            in_channels=num_in_q,
            out_channels=num_out_q,
        )

        # TODO 1x1, linear?
        self.pi = torch_geometric.nn.GCNConv(
            in_channels=num_in_pi, out_channels=num_out_pi
        )

        # TODO check dim, using 1D, for every node in all graph
        self.sm = torch.nn.Softmax()

        # > store forward fixed-point iteration loss
        self.residuals_forward = None

    def forward(self, graphs, visualize=False, return_qrv=False):
        x = graphs.x
        edge_index = graphs.edge_index

        pos = x[:, :2]
        feat = x[:, 2:] if self.args.obsv_mode == "full" else graphs.partial_x

        q, r, v = self._value_iterate(pos, feat, edge_index)

        if visualize:
            random_idx = np.random.randint(0, graphs.batch.max().cpu())
            our_v = visualize_graph_v(
                graphs[random_idx],
                v[graphs.batch == random_idx],
                title="Our Value Graph",
            )
            gt_v = visualize_graph_v(
                graphs[random_idx],
                graphs.opt_v[graphs.batch == random_idx],
                title="Ground-Truth",
            )

        logits = self.pi(torch.cat([pos, q], dim=1), edge_index)
        probs = self.sm(logits) if not self.args.cont_action else logits

        if return_qrv:
            return StandardReturn(logits, probs), (q, r, v)
        elif visualize:
            return StandardReturn(logits, probs), our_v, gt_v
        else:
            return StandardReturn(logits, probs)

    def _value_iterate(self, pos, feat, edge_index):
        h = self.h(torch.cat([pos, feat], dim=1), edge_index)
        r = self.r(torch.cat([pos, h], dim=1), edge_index)

        v = torch.zeros(r.size(), device=h.device)

        self.residuals_forward = []
        for t in range(self.k - 1):
            v_prev = v.detach().clone()

            rv = torch.cat([r, v], dim=1)  # TODO
            q = self.q(torch.cat([pos, rv], dim=1), edge_index)

            # FIXME dim & keep dim
            v, _ = torch.max(q, dim=1, keepdim=True)

            # > compute residuals
            res = (v - v_prev).norm().item() / (1e-5 + v.norm().item())
            self.residuals_forward.append(res)

        rv = torch.cat([r, v], dim=1)
        q = self.q(torch.cat([pos, rv], dim=1), edge_index)

        return q, r, v
