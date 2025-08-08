import escnn
import escnn.gspaces
import numpy as np
import torch
from escnn import nn as esnn
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from diffplan.modules.helpers import StandardReturn
from diffplan.modules.pooling_esnn import GroupReducedMaxOperation
from diffplan.utils import helpers_esnn as h
from diffplan.utils.plot_wandb import visualize_graph_v


# matplotlib.use("TkAgg")


class MPPlanningNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        num_orient = 1  # deprecated, used in VIN grid setting
        # discrete (4 directions), continuous (2 dimensions)
        num_actions = 2 if args.cont_action else 4

        # Initialize symmetry group and geometric space for equivariance
        self.g_name = args.group
        self.group = h.get_group(args.group)
        self.g_space = escnn.gspaces.no_base_space(self.group)

        self.l_h = args.l_h
        self.l_q = args.l_q
        self.k = args.k

        self.mp_agg = args.mp_agg  # Message Passing Layer aggregation method
        # Note: Parameter count kept consistent using latent_dim_factor='sqrt'
        # Hidden size for Message Passing Layer
        self.mp_h = h.get_latent_num(args, g_space=self.g_space, h_dim=args.mp_h)
        # hidden size for Message Passing Layer self.q
        self.mp_q = h.get_latent_num(args, g_space=self.g_space, h_dim=args.mp_q)

        # num_in_h = (num_orient + 1)
        num_in_h = 2 + 1 + 1  # xy position + goal + obstacle
        num_out_h = self.l_h
        num_in_r = num_out_h
        num_out_r = num_orient
        num_in_q = num_out_r * 2  # takes R & V
        num_out_q = self.l_q * num_orient
        num_out_pi = num_actions

        # TODO add pos
        self.repr_pos = h.get_repr("location", self.g_space, args)
        self.field_pos = self.g_space.type(*self.repr_pos)
        self.repr_v = h.get_repr("value", self.g_space, args)  # FIXME changed

        # Initialize the group representations
        # repr_in_h = h.get_repr('goal', self.g_space, args) + h.get_repr('obstacle', self.g_space, args)
        if args.task == "Habitat":
            # TODO: Breaks when it is D4 -> become 8x128, but we have 4x128
            repr_in_h = h.get_repr("img-encoded", self.g_space, args)
        elif args.obsv_mode == "full" and args.has_obstacle:
            # goal + obstacle
            repr_in_h = h.get_repr("goal", self.g_space, args) + h.get_repr(
                "obstacle", self.g_space, args
            )
        elif args.obsv_mode == "full" and not args.has_obstacle:
            # goal
            repr_in_h = h.get_repr("goal", self.g_space, args)
        elif args.obsv_mode == "partial" and args.has_obstacle:
            # goal + obstacle in one-hot
            repr_in_h = [self.g_space.trivial_repr] * 6
        elif args.obsv_mode == "partial" and not args.has_obstacle:
            # goal in one-hot
            repr_in_h = [self.g_space.trivial_repr] * 3
        else:
            raise ValueError

        repr_out_h = h.get_repr("latent", self.g_space, args)
        repr_in_r = repr_out_h
        repr_out_r = self.repr_v
        repr_in_q = repr_out_r * 2
        repr_out_q = h.get_repr("q_map", self.g_space, args)

        repr_in_max = repr_out_q  # input = Q
        repr_out_max = self.repr_v  # output = V

        # TODO add option in args
        if args.cont_action:
            repr_out_pi = h.get_repr("location", self.g_space, args)
        else:
            repr_out_pi = h.get_repr(
                "regular", self.g_space, args
            )  # TODO regular - need to use counterclockwise (4abs-cc)

        self.in_repr = self.repr_pos + repr_in_h
        self.in_type = self.g_space.type(*self.in_repr)

        self.h = Equivariant2DMessagePassingLayer(
            in_repr=repr_in_h,
            out_repr=repr_out_h,
            pos_repr=self.repr_pos,
            hid_num=self.mp_h,
            g_space=self.g_space,
            aggr=self.mp_agg,
        )

        self.r = Equivariant2DMessagePassingLayer(
            in_repr=repr_in_r,
            out_repr=repr_out_r,
            pos_repr=self.repr_pos,
            hid_num=self.mp_h,  # TODO also h?
            g_space=self.g_space,
            aggr=self.mp_agg,
        )

        self.q = Equivariant2DMessagePassingLayer(
            in_repr=repr_in_q,
            out_repr=repr_out_q,
            pos_repr=self.repr_pos,
            hid_num=self.mp_q,
            g_space=self.g_space,
            aggr=self.mp_agg,
        )

        if not self.args.no_equiv_policy:
            assert num_actions == repr_out_pi[0].size, (
                "Equivariant policy layer only works for 2 actions "
                "(continuous) or 4 actions (discrete) for now"
            )
            self.pi = Equivariant2DMessagePassingLayer(
                in_repr=repr_out_q,
                out_repr=repr_out_pi,
                pos_repr=self.repr_pos,
                hid_num=self.mp_h,
                g_space=self.g_space,
                aggr=self.mp_agg,
            )
        else:
            # Consider the size of regular repr, multiply the corresponding size
            # Example: use C4 group and sqrt strategy, then size = |C4| = 2
            dim_equiv = h.get_latent_num(
                self.args, self.g_space, num_out_q, multiply_repr_size=True
            )
            self.pi = MessagePassingLayer(
                in_dim=dim_equiv + 2,
                out_dim=num_out_pi,
                hidden_dim=self.mp_h,
                aggr=self.mp_agg,
            )

        # Customized max operation over repr
        self.max_over_repr = GroupReducedMaxOperation(
            g_space=self.g_space,
            in_repr=repr_in_max,
            # out_mode='keep',
            out_repr=repr_out_max,
            no_base=True,
        )

        # TODO check dim, using 1D, for every node in all graph
        self.sm = torch.nn.Softmax()

        # > store forward fixed-point iteration loss
        self.residuals_forward = None

    def forward(self, graphs, visualize=False, return_qrv=False):
        x = graphs.x
        edge_index = graphs.edge_index

        # TODO add pos
        pos = x[:, :2]
        pos_geo = self.field_pos(pos)

        # Wrap with geometric tensor
        feat_geo = self.h.in_type(x[:, 2:])

        q_geo, r_geo, v_geo = self._value_iterate(pos_geo, feat_geo, edge_index)

        if visualize:
            # FIXME for equivariant with regular repr, take mean or only [0]
            random_idx = np.random.randint(0, graphs.batch.max().cpu())
            our_v = visualize_graph_v(
                # graphs[random_idx], v_geo.tensor[graphs.batch == random_idx], title="Our Value Graph")
                graphs[random_idx],
                v_geo.tensor[graphs.batch == random_idx].mean(-1, keepdim=True),
                title="Our Value Graph",
            )
            gt_v = visualize_graph_v(
                graphs[random_idx],
                graphs.opt_v[graphs.batch == random_idx],
                title="Ground-Truth",
            )

        # Option to use equivariant policy or not
        if self.args.no_equiv_policy:
            q = q_geo.tensor
            logits = self.pi(pos, q, edge_index)
        else:
            logits_geo = self.pi(pos_geo, q_geo, edge_index)
            logits = logits_geo.tensor

        probs = self.sm(logits) if not self.args.cont_action else logits

        if return_qrv:
            return StandardReturn(logits, probs), (q_geo, r_geo, v_geo)
        elif visualize:
            return StandardReturn(logits, probs), our_v, gt_v
        else:
            return StandardReturn(logits, probs)

    def _value_iterate(self, pos_geo, feat_geo, edge_index):
        """
        Args:
            pos_geo: (x, y)
            feat_geo: (goal, obstacle)
        """

        h_geo = self.h(pos_geo, feat_geo, edge_index)
        r_geo = self.r(pos_geo, h_geo, edge_index)

        v_raw = torch.zeros(r_geo.size(), device=h_geo.tensor.device)
        v_geo = self.r.out_type(v_raw)  # just to use trivial repr to wrap a scalar

        self.residuals_forward = []
        for t in range(self.k - 1):
            v_prev = v_geo.tensor.detach().clone()

            rv_geo = esnn.tensor_directsum([r_geo, v_geo])

            q_geo = self.q(pos_geo, rv_geo, edge_index)

            # To maximize only over fiber channel
            v_geo = self.max_over_repr(q_geo)
            # > Q: #nodes x (|G| * #repr) x width x height
            # > V: #nodes x (|G| * 1) x width x height

            # > compute residuals
            res = (v_geo.tensor - v_prev).norm().item() / (
                1e-5 + v_geo.tensor.norm().item()
            )
            self.residuals_forward.append(res)

        rv_geo = esnn.tensor_directsum([r_geo, v_geo])
        q_geo = self.q(pos_geo, rv_geo, edge_index)

        return q_geo, r_geo, v_geo


class Equivariant2DMessagePassingLayer(MessagePassing):
    def __init__(
        self,
        in_repr,
        out_repr,
        pos_repr,
        hid_num,
        g_space: escnn.gspaces.GSpace,
        edge_repr=None,
        aggr="add",
    ):
        super().__init__(aggr=aggr)

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.pos_repr = pos_repr
        self.edge_repr = edge_repr
        self.hid_num = hid_num
        self.g_space = g_space

        self.in_type = g_space.type(*self.in_repr)
        self.out_type = g_space.type(*self.out_repr)
        self.pos_type = g_space.type(*self.pos_repr)

        self.msg_repr = [g_space.regular_repr] * hid_num

        self.msg_in_type = g_space.type(
            *(
                self.pos_repr
                + self.in_repr * 2
                + (self.edge_repr if edge_repr is not None else [])
            )
        )
        # TODO use regular repr for message -> keep equivariant quantities throughout message passing
        self.msg_out_type = g_space.type(*(self.msg_repr))
        self.upd_in_type = g_space.type(*(self.pos_repr + self.in_repr + self.msg_repr))

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # TODO dims: (2 * d_in + d_e) -> d_h
        self.sym_mlp_msg = h.sym_mlp(
            g_space,
            in_field=self.msg_in_type,
            out_field=self.msg_out_type,
            h_num=hid_num,
        )

        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # TODO dims: 2 * d_in -> d_out
        self.sym_mlp_upd = h.sym_mlp(
            g_space, in_field=self.upd_in_type, out_field=self.out_type, h_num=hid_num
        )

    def forward(self, pos, feat, edge_index, edge_attr=None):
        # TODO quick fix: unwrap geometric tensor
        pos = pos.tensor
        feat = feat.tensor
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
        # TODO quick fix: re-wrap to geometric tensor again
        pos_i = self.pos_type(pos_i)
        pos_j = self.pos_type(pos_j)
        feat_i = self.in_type(feat_i)
        feat_j = self.in_type(feat_j)

        if edge_attr is None:
            msg = esnn.tensor_directsum([pos_j - pos_i, feat_i, feat_j])
        else:
            msg = esnn.tensor_directsum([pos_j - pos_i, feat_i, feat_j, edge_attr])

        out = self.sym_mlp_msg(msg)
        return out.tensor

    def aggregate(self, inputs, index):
        """
        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        aggregated = scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
        return aggregated
        # agg_geo = self.msg_out_type(aggregated)  # TODO change to message out type
        # return agg_geo

    def update(self, aggr_out, pos, feat):
        """
        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        # TODO wrap again
        feat = self.in_type(feat)
        pos = self.pos_type(pos)
        aggr_out = self.msg_out_type(aggr_out)  # TODO change to message out type

        upd_out = esnn.tensor_directsum([pos, feat, aggr_out])
        return self.sym_mlp_upd(upd_out)


# TODO just copy here from MP-VIN
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
            nn.Linear(2 * in_dim - 2 + edge_dim, self.hidden_dim),
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
            msg = torch.cat([pos_j - pos_i, feat_i, feat_j], dim=-1)
        else:
            msg = torch.cat([pos_j - pos_i, feat_i, feat_j, edge_attr], dim=-1)
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
