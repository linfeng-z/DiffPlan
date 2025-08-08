import torch
from torch import nn

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool import global_mean_pool

from torch_scatter import scatter


class MessagePassingLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=1, aggr="add"):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        # self.node_dim = -1  # TODO

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + edge_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        """
        message
        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features

        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
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

    def update(self, aggr_out, h):
        """
        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)


class MessagePassingPathPlanningNetwork(nn.Module):
    def __init__(self, x_dim, node_dim, edge_dim=0, num_layers=4, emb_dim=64):
        super().__init__()

        self.embed_in = nn.Linear(x_dim + node_dim, emb_dim)

        layers = []
        for i in range(num_layers):
            layers.append(
                MessagePassingLayer(emb_dim=emb_dim, edge_dim=edge_dim, aggr="sum")
            )
        self.layers = nn.ModuleList(layers)

        self.pool = global_mean_pool
        self.reward_mlp = nn.Linear(emb_dim, emb_dim)
        self.reward_out = nn.Linear(emb_dim, 1)

        self.pred_mlp = nn.Linear(emb_dim, emb_dim)
        self.pred_act = nn.ReLU()
        self.pred_out = nn.Linear(emb_dim, x_dim)

    def forward(self, data, residual=False):
        # NOTE: input - with node features
        # x_in = torch.cat([data.x, data.node_attr], dim=1)
        x_in = data.x

        h = self.embed_in(x_in)

        for layer in self.layers:
            h = h + layer(h=h, edge_index=data.edge_index, edge_attr=data.edge_attr)

        # NOTE: output process 1, pooling - for graph level, e.g. reward
        h_reward = self.reward_mlp(h)
        h_graph = self.pool(h_reward, data.batch)
        out_reward = self.reward_out(h_graph)

        # NOTE: output process 2 - prediction
        h_pred = self.pred_mlp(h)
        h_pred = self.pred_act(h_pred)
        x_out = self.pred_out(h_pred)

        # NOTE: - be consistent with steerable case, not residual
        # FIXME should be a model parameter later
        if residual:
            out = x_out + x_in
        else:
            out = x_out

        return out
