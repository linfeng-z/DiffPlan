import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from diffplan.modules.helpers import StandardReturn
from diffplan.utils import visualize_graph_v


class MPPlanningNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        num_orient = 1  # deprecated, used in VIN grid setting
        # discrete (4), continous (1)
        num_actions = 2 if args.cont_action else 4

        self.l_h = args.l_h
        self.l_q = args.l_q
        self.k = args.k
        self.mp_h = args.mp_h  # hidden size for Message Passing Layer self.h
        self.mp_q = args.mp_q  # hidden size for Message Passing Layer self.q
        self.mp_agg = args.mp_agg  # Message Passing Layer aggregation method

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
        num_out_pi = num_actions

        self.h = MessagePassingLayer(
            in_dim=num_in_h, out_dim=num_out_h, hidden_dim=self.mp_h, aggr=self.mp_agg
        )

        # TODO 1x1, linear?
        self.r = MessagePassingLayer(
            in_dim=num_in_r, out_dim=num_out_r, hidden_dim=self.mp_h, aggr=self.mp_agg
        )

        self.q = MessagePassingLayer(
            in_dim=num_in_q, out_dim=num_out_q, hidden_dim=self.mp_q, aggr=self.mp_agg
        )

        # TODO 1x1, linear?
        self.pi = MessagePassingLayer(
            in_dim=num_out_q + 2,  # extra relative position input?
            out_dim=num_out_pi,
            hidden_dim=self.mp_h,
            aggr=self.mp_agg,
        )

        self.sm = torch.nn.Softmax()

        # > store forward fixed-point iteration loss
        self.residuals_forward = None

    def forward(self, graphs, visualize=False, return_qrv=False):
        x = graphs.x
        edge_index = graphs.edge_index

        # x = graphs.x

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

        logits = self.pi(pos, q, edge_index)
        probs = self.sm(logits) if not self.args.cont_action else logits

        if return_qrv:
            return StandardReturn(logits, probs), (q, r, v)
        elif visualize:
            return StandardReturn(logits, probs), our_v, gt_v
        else:
            return StandardReturn(logits, probs)

    def _value_iterate(self, pos, feat, edge_index):
        h = self.h(pos, feat, edge_index)
        r = self.r(pos, h, edge_index)

        v = torch.zeros(r.size(), device=h.device)

        self.residuals_forward = []
        for t in range(self.k - 1):
            v_prev = v.detach().clone()

            rv = torch.cat([r, v], dim=1)  # TODO
            q = self.q(pos, rv, edge_index)

            # FIXME dim & keep dim
            v, _ = torch.max(q, dim=1, keepdim=True)

            # > compute residuals
            res = (v - v_prev).norm().item() / (1e-5 + v.norm().item())
            self.residuals_forward.append(res)

        rv = torch.cat([r, v], dim=1)
        q = self.q(pos, rv, edge_index)

        return q, r, v


class MessagePassingLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, hidden_dim=128, edge_dim=0, aggr="add"):
        super().__init__(aggr=aggr)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        # self.node_dim = -1  # TODO

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        # TODO dims: (2 * d_in + d_e) -> d_h
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.in_dim),
        )

        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        # TODO dims: 2 * d_in -> d_out
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * self.in_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, pos, feat, edge_index, edge_attr=None):
        """
        pos: Position (2d)
        feat: Node Feature
        """
        out = self.propagate(edge_index, pos=pos, feat=feat, edge_attr=edge_attr)
        return out

    def message(self, pos_i, pos_j, feat_i, feat_j, edge_attr):
        """
        message
        Args:
            pos_i: (e, d) - destination node position
            pos_j: (e, d) - source node position
            feat_i: (e, d) - destination node features
            feat_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features

        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        if edge_attr is None:
            # pos_j-pos_i, feat_i, feat_j
            msg = torch.cat([pos_i, pos_j, feat_i, feat_j], dim=-1)
        else:
            msg = torch.cat([pos_i, pos_j, feat_i, feat_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):
        """
        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
        # return agg

    def update(self, aggr_out, pos, feat):
        """
        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([pos, feat, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)
