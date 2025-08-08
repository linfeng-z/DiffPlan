# Standard library imports
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# Data processing and visualization
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm  # For colormaps
import matplotlib.colors as mpl_colors  # For color normalization
import networkx as nx  # For graph visualization
import numpy as np
import pandas as pd
from PIL import Image  # For image handling

# Deep learning frameworks
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.utils as geo_utils

# Experiment tracking
import wandb

# Equivariant neural networks
try:
    from escnn import gspaces, group
except ImportError:
    print("Warning: escnn not found. Some functionality may be limited.")
    gspaces = None
    group = None

# Local imports
from diffplan.envs.generate_dataset import extract_goal
from diffplan.envs.maze_utils import extract_policy
from diffplan.modules.mapper import (
    NavMapper,
    GraphNavMapper,
    GraphImgEncoder,
    SymGraphImgEncoder,
)
from diffplan.utils.dijkstra import dijkstra_dist
from diffplan.utils.experiment import get_mechanism
from diffplan.utils.graph_converter import get_partial_graph_obsv_mask


class LitBase(pl.LightningModule):
    """Base Lightning Module for all planners.
    
    This module provides core functionality for:
    1. Graph visualization with value functions
    2. Residual tracking and convergence plotting
    3. Task-specific initialization (VisNav, Habitat)
    """
    
    def __init__(self, args: Any) -> None:
        """Initialize base planner.
        
        Args:
            args: Configuration arguments for the planner including:
                - task: Type of planning task (VisNav, Habitat)
                - algorithm: Planning algorithm type
                - visual_feat: Visual feature dimension
                - group: Group type for equivariant networks
        """
        super().__init__()
        self.args = args
        self.batch_size: Optional[int] = None
        
        # Initialize metrics
        self.success_count = 0
        self.total_count = 0
        
        # Action order for grid transitions: North, West, South, East
        self.action_order = [[-1, 0], [0, -1], [1, 0], [0, 1]]

        # Initialize task-specific components
        if self.args.task == "VisNav":
            if "MP" in args.algorithm:
                self.mapper = GraphNavMapper(4, 32, 32)
            else:
                self.mapper = NavMapper(15, 15, 4, 32, 32)
        elif self.args.task == "Habitat":
            if "Sym-MP" in args.algorithm:
                self.img_encoder = SymGraphImgEncoder(
                    out_dim=args.visual_feat,
                    out_group=args.group
                )
            elif "MP" in args.algorithm or "GAT" in args.algorithm:
                self.img_encoder = GraphImgEncoder(out_dim=args.visual_feat)

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Configure optimizers.
        
        Returns:
            Dict containing optimizer
        """
        if self.args.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optimizer} not implemented")

        return {"optimizer": optimizer}

    def discrete_action_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss for discrete actions."""
        return F.cross_entropy(pred, target)

    def cont_action_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss for continuous actions."""
        return F.mse_loss(pred, target)

    def map_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss for map predictions."""
        return F.binary_cross_entropy_with_logits(pred, target)

    def flatten_repr_channel(self, img_feat: torch.Tensor, group_size: int = 4) -> torch.Tensor:
        """Flatten representation channels.
        
        Args:
            img_feat: Input features of shape (batch_size, feat_size, num_views)
            group_size: Size of the group (number of views)
            
        Returns:
            Flattened tensor of shape (batch_size, feat_size * num_views)
        """
        batch_size, img_embed_dim, num_views = img_feat.size()
        assert group_size == num_views, f"Group size {group_size} != number of views {num_views}"

        out_tensor = torch.empty(batch_size, num_views * img_embed_dim, device=img_feat.device)
        base_indices = torch.arange(start=0, end=num_views * img_embed_dim, step=num_views)
        view2indices = {v: (base_indices + v) for v in range(num_views)}

        for i in range(group_size):
            repr_indices = view2indices[i]
            repr_tensor = img_feat[:, :, i].view(batch_size, img_embed_dim)
            out_tensor[:, repr_indices] = repr_tensor

        return out_tensor

    def discrete_action_to_cont(self, action: torch.Tensor) -> torch.Tensor:
        """Convert one-hot vector to R^2 relative translation.
        
        Args:
            action: One-hot action tensor of shape (batch_size, num_actions)
            
        Returns:
            Continuous action tensor of shape (batch_size, 2)
        """
        indices = torch.argmax(action, dim=1)
        action_order = torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1]], 
                                  dtype=torch.float32, 
                                  device=action.device)
        return action_order[indices]

    def cont_action_to_discrete(self, action: torch.Tensor) -> torch.Tensor:
        """Convert continuous action to one-hot vector.
        
        Args:
            action: Continuous action tensor of shape (batch_size, 2)
            
        Returns:
            One-hot action tensor of shape (batch_size, 4)
        """
        NEWS = torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1]],
                          dtype=torch.float32,
                          device=action.device)
        distances = torch.cdist(action, NEWS)  # find closest neighbor
        closest_indices = torch.argmin(distances, dim=1)
        return F.one_hot(closest_indices, 4)

    def accuracy(self, labels: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Compute accuracy between predicted and target actions.
        
        Args:
            labels: Target action tensor
            probs: Predicted action probabilities
            
        Returns:
            Accuracy as percentage
        """
        return (torch.sum(torch.argmax(probs, dim=1) == torch.argmax(labels, dim=1)) 
                / len(probs) * 100.0)

    def graph_sample_traj(self, graph: Data) -> Tuple[int, int]:
        """Sample a random trajectory on a graph.
        
        Args:
            graph: PyG graph data object
            
        Returns:
            Tuple of (start_node_idx, path_length)
        """
        obstacle_v = graph.opt_v.min()  # obstacle value
        
        # Randomly choose valid start node (not obstacle & not goal)
        while True:
            rand_node_num = np.random.randint(0, len(graph.x))
            if self.args.task == "Habitat" and graph.opt_v[rand_node_num] > 0:
                break
            if (graph.opt_v[rand_node_num] > obstacle_v and 
                graph.opt_v[rand_node_num] < 0):
                break
                
        # Compute path length
        if self.args.task in ["Habitat", "GraphWorld"]:
            steps = len(graph.x)  # total number of nodes
        else:
            steps = -graph.opt_v[rand_node_num].detach().cpu().int().numpy()[0]
            
        return rand_node_num, steps

    def graph_transition_helper(self, graph: nx.Graph, curr_node: int, 
                              action: torch.Tensor) -> int:
        """Compute next node after taking action.
        
        Args:
            graph: NetworkX graph
            curr_node: Current node index
            action: Action tensor
            
        Returns:
            Next node index
        """
        # Get positions of neighbors
        neighbors = []
        neighbor_idx = []
        for neighbor in graph.neighbors(curr_node):
            if neighbor == curr_node:
                continue
            neighbors.append(graph.nodes[neighbor]["x"])
            neighbor_idx.append(neighbor)
            
        if not neighbors:  # No valid neighbors
            return curr_node
            
        neighbor_num = len(neighbors)
        neighbors = torch.tensor(neighbors, device=action.device)

        # Find nearest neighbor after taking action
        curr_pos = torch.tensor(graph.nodes[curr_node]["x"][:2], device=action.device)
        curr_pos = curr_pos.unsqueeze(0).repeat(neighbor_num, 1)
        action = action.unsqueeze(0)
        distances = torch.cdist(action, neighbors[:, :2] - curr_pos)
        closest_indices = torch.argmin(distances, dim=1)
        node = neighbor_idx[closest_indices]

        # Check for obstacles
        if self.args.has_obstacle:
            if graph.nodes[node]["x"][-1] != 1.0:
                return node
            return curr_node
            
        return node

    def graph_success_rate(self, sample: Data, probs: torch.Tensor) -> float:
        """Compute success rate for graph navigation.
        
        Args:
            sample: PyG graph data
            probs: Action probabilities
            
        Returns:
            Success rate as percentage
        """
        total = 0
        success = 0
        probs = geo_utils.unbatch(probs, sample.batch)
        
        for i in range(sample.batch.max() + 1):
            start_node, length = self.graph_sample_traj(sample[i])
            policy = probs[i]  # continuous action
            
            # Convert to NetworkX for easier operations
            nx_g = geo_utils.to_networkx(
                sample[i], 
                to_undirected=True,
                node_attrs=["x", "opt_v"]
            )
            
            curr_node = start_node
            for _ in range(2 * length):
                curr_node = self.graph_transition_helper(nx_g, curr_node, policy[curr_node])
                
                if self.args.task == "Habitat":
                    if nx_g.nodes[curr_node]["opt_v"] <= 10:
                        success += 1
                        break
                else:
                    if nx_g.nodes[curr_node]["x"][2] == 1.0:
                        success += 1
                        break
                        
            total += 1
            
        return success / total * 100.0

    def visualize_graph_v(self, graph: Data, v: torch.Tensor, 
                        curr_node: int, prev_mask: List[int], 
                        title: str = "Untitled") -> Image.Image:
        """Visualize the value function on the graph.
        
        This method creates a visualization of a graph with node colors representing 
        the value function. Special nodes (current, goal, visited) are highlighted 
        with different colors.
        
        Args:
            graph: PyG graph data containing node features and edge indices
            v: Value function tensor for each node
            curr_node: Index of the current node (highlighted in blue)
            prev_mask: List of previously visited nodes (highlighted with blue edges)
            title: Plot title
            
        Returns:
            PIL Image containing the graph visualization with colorbar
        """
        # Remove self loops for cleaner visualization
        edge_index = geo_utils.remove_self_loops(graph.edge_index)[0]
        g = geo_utils.to_networkx(graph.clone(), to_undirected=True)
        g.remove_edges_from(nx.selfloop_edges(g))
        
        # Extract and preprocess node features and value function
        features = graph.x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        # Normalize extreme values for better visualization
        v_np[:, 0][v_np[:, 0] == -225.0] = 225.0
        v_np[:, 0][v_np[:, 0] == 225.0] = np.min(v_np[:, 0])
        
        # Set up node positions and color mapping
        positions = {str(i): pos for i, pos in enumerate(features[:, :2])}
        norm = mpl_colors.Normalize(vmin=np.min(v_np[:, 0]), vmax=np.max(v_np[:, 0]))
        colors = mpl_cm.get_cmap('Greys')(norm(v_np[:, 0]))
        
        # Create node colors list:
        # - Regular nodes: colored by value function
        # - Goal nodes (features[:, 2] == 1): red
        # - Current node: blue
        node_colors = [
            colors[i] if features[i, 2] == 0 else "red" 
            for i in range(len(features))
        ]
        node_colors[curr_node] = "blue"
        
        # Create edge colors list:
        # - Visited edges (in prev_mask): blue
        # - Unvisited edges: white
        edgecolors = [
            "blue" if i in prev_mask else "white" 
            for i in range(len(features))
        ]
        
        # Create the plot
        fig = plt.figure()
        plt.title(title)
        nx.draw(g, positions, node_color=node_colors, edgecolors=edgecolors)
        
        # Add colorbar showing value function scale
        sm = mpl_cm.ScalarMappable(cmap='Greys', norm=norm)
        sm.set_array([])  # Required for colorbar to display correctly
        plt.colorbar(sm)
        
        # Convert matplotlib figure to PIL image
        buf = BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        plt.close()
        
        return Image.open(buf)

    def partial_graph_success_rate(self, sample, probs):
        total = 0
        success = 0
        for i in range(1):
            # for i in range(sample.batch.max()):
            start_node, length = self.graph_sample_traj(sample[i])
            # convert to networkx for easier operation
            nx_g = torch_geometric.utils.to_networkx(
                sample[i], to_undirected=True, node_attrs=["x", "partial_x"]
            )
            curr_node = start_node
            prev_mask = []
            for _ in range(2 * length):
                obsv_mask = get_partial_graph_obsv_mask(nx_g, curr_node)
                # merge just discovered state with previously discovered state
                prev_mask += obsv_mask
                prev_mask = [*set(prev_mask)]

                # update partial observation according to current observation mask
                new_sample = torch_geometric.utils.from_networkx(nx_g)
                new_partial_x = torch.zeros(
                    sample[i].partial_x.shape,
                    device=sample[i].partial_x.device,
                    dtype=sample[i].partial_x.dtype,
                )

                # new goal map
                new_partial_x[:, :3] = torch.tensor(
                    [0, 0, 1],
                    device=sample[i].partial_x.device,
                    dtype=sample[i].partial_x.dtype,
                ).repeat(len(new_sample.x), 1)
                new_partial_x[prev_mask, :3] = torch.tensor(
                    [0, 1, 0],
                    device=sample[i].partial_x.device,
                    dtype=sample[i].partial_x.dtype,
                ).repeat(len(prev_mask), 1)
                if (new_sample.x[:, 2] == 1).nonzero()[0, 0] in prev_mask:
                    new_partial_x[new_sample.x[:, 2] == 1, :3] = torch.tensor(
                        [1, 0, 0],
                        device=sample[i].partial_x.device,
                        dtype=sample[i].partial_x.dtype,
                    )

                # use obstacle map
                if self.args.has_obstacle:
                    new_partial_x[:, 3:] = torch.tensor(
                        [0, 0, 1],
                        device=sample[i].partial_x.device,
                        dtype=sample[i].partial_x.dtype,
                    ).repeat(len(new_sample.x), 1)
                    for node in prev_mask:
                        if new_sample.x[node, 3] == 1:
                            new_partial_x[node, 3:] = torch.tensor(
                                [0, 1, 0],
                                device=sample[i].partial_x.device,
                                dtype=sample[i].partial_x.dtype,
                            )
                        else:
                            new_partial_x[node, 3:] = torch.tensor(
                                [1, 0, 0],
                                device=sample[i].partial_x.device,
                                dtype=sample[i].partial_x.dtype,
                            )
                new_sample.partial_x = new_partial_x

                # new prediction based on current observation
                self.model.eval()
                out, (q, r, v) = self.model(new_sample.cuda(), return_qrv=True)
                logits, policy = out.logits, out.probs

                partial_v = self.visualize_graph_v(
                    new_sample,
                    v,
                    curr_node,
                    prev_mask,
                    "Value graph of Partial Environment",
                )
                wandb.log({"val/partial_v_fig": wandb.Image(partial_v)})

                curr_node = self.graph_transition_helper(
                    nx_g, curr_node, policy[curr_node]
                )
                if nx_g.nodes[curr_node]["x"][2] == 1.0:
                    success += 1
                    break
            total += 1
        return success / total * 100.0

    def _equiv_err(self, sample, theta=0, cyclic_n=4):
        if hasattr(self.model, "g_space"):
            _gspace = self.model.g_space
        else:
            _group = group.cyclic_group(N=cyclic_n)
            _gspace = gspaces.no_base_space(_group)

        # specify input repr: position (standard repr) + obstacle and goal (2 trivial repr)
        if hasattr(self.model, "in_type"):
            _in_field = self.model.in_type
        else:
            _in_repr = [_gspace.irrep(1)] * 1 + [_gspace.trivial_representation] * 2
            _in_field = _gspace.type(*_in_repr)

        test_geo = _in_field(sample.x)

        q_err_arr = []
        r_err_arr = []
        v_err_arr = []
        h_err_arr = []

        g = _gspace.fibergroup.elements[theta]
        # need to iter over all elements
        _transformed_input = sample.clone()
        _transformed_input.x = test_geo.transform(g).tensor
        _, (q, r, v) = self.model(_transformed_input, visualize=False, return_qrv=True)
        _, (_q_rot, _r_rot, _v_rot) = self.model(
            sample, visualize=False, return_qrv=True
        )
        q_rot = _q_rot.transform(g)
        r_rot = _r_rot.transform(g)
        v_rot = _v_rot.transform(g)

        # calculate EE
        equiv_err_q = torch.mean(torch.abs(q.tensor - q_rot.tensor))
        equiv_err_r = torch.mean(torch.abs(r.tensor - r_rot.tensor))
        equiv_err_v = torch.mean(torch.abs(v.tensor - v_rot.tensor))

        return equiv_err_q, equiv_err_r, equiv_err_v

    def MP_VIN_training_step(self, sample, batch_idx):
        if self.args.task == "VisNav":
            # enable mapping
            pano_obs = sample.pano_obs
            pred_map_logits = self.mapper(pano_obs)
            # calculate the map loss first
            map_loss = self.map_loss(pred_map_logits[:, 0], sample.x[:, 3])
            self.log(
                "train/map_loss", map_loss, on_epoch=True, batch_size=self.batch_size
            )

            pred_map = F.sigmoid(pred_map_logits)
            # change the obstacle to the predicted using visual input
            sample.x = torch.cat([sample.x[:, :3].detach(), pred_map], axis=1)
        elif self.args.task == "Habitat":
            # img_feat = self.get_visual_feat(sample)
            # TODO use encoder
            img_feat = self.img_encoder(sample)
            sample.x = torch.cat([sample.x[:, :2], img_feat], axis=1)  # 2+512

        out = self.model(sample)
        logits, probs = out.logits, out.probs
        if self.args.cont_action:
            loss = self.cont_action_loss(logits, sample.labels.float())
        else:
            loss = self.discrete_action_loss(logits, torch.argmax(sample.labels, dim=1))

        if self.args.task == "VisNav":
            loss += map_loss

        self.log("train/loss", loss, batch_size=self.batch_size)

        pred = (
            self.discrete_action_to_cont(probs)
            if not self.args.cont_action
            else probs.detach()
        )
        labels = (
            self.discrete_action_to_cont(sample.labels)
            if not self.args.cont_action
            else sample.labels.detach()
        )
        if self.args.obsv_mode == "full":
            self.log(
                "train/success",
                self.graph_success_rate(sample, pred),
                on_epoch=True,
                batch_size=self.batch_size,
            )  # monitor this value for checkpoint
        else:
            self.log(
                "train/success",
                self.partial_graph_success_rate(sample, pred),
                on_epoch=True,
                batch_size=self.batch_size,
            )  # monitor this value for checkpoint

        # if wandb.run is not None:
        #     info_convergence = self._plot_convergence_curve()
        #     wandb.log(info_convergence)

        # TODO update
        # info = {}
        # info['train/success'] = self.graph_success_rate(sample, pred)
        # self.log_dict(info)

        return loss

    def _plot_convergence_curve(self, num_samples=10):
        info = {}

        if (
            hasattr(self.model, "residuals_forward")
            and self.model.residuals_forward is not None
        ):
            curve_fw_wandb, curve_fw_plotly = self._get_convergence_curve(
                residuals=self.model.residuals_forward,
                title="Forward Pass: Iteration Residuals",
                num_samples=num_samples,
            )

            info.update(
                {
                    "forward_residual_curve": curve_fw_wandb,
                    "forward_residual_curve_plotly": curve_fw_plotly,
                    "forward_residual_final": self.model.residuals_forward[-1],
                    "forward_residual_avg": np.mean(self.model.residuals_forward),
                    "forward_num_iter": len(self.model.residuals_forward),
                }
            )

        if (
            hasattr(self.model, "residuals_backward")
            and self.model.residuals_backward is not None
        ):
            curve_bw_wandb, curve_bw_plotly = self._get_convergence_curve(
                residuals=self.model.residuals_backward,
                title="Backward Pass: Iteration Residuals",
                num_samples=num_samples,
            )

            info.update(
                {
                    "backward_residual_curve": curve_bw_wandb,
                    "backward_residual_curve_plotly": curve_bw_plotly,
                    "backward_residual_final": self.model.residuals_backward[-1],
                    "backward_residual_avg": np.mean(self.model.residuals_backward),
                    "backward_num_iter": len(self.model.residuals_backward),
                }
            )

        if "DE" in self.args.algorithm:
            assert len(info) > 0

        return info

    def _get_convergence_curve(self, residuals: List[float], title: str, 
                             num_samples: int = 10) -> Tuple[Any, None]:
        """Generate convergence curve plots.
        
        Creates a wandb line plot showing how residuals converge over iterations.
        The curve is sampled to reduce data points while preserving the shape.
        
        Args:
            residuals: List of residual values
            title: Plot title
            num_samples: Number of points to sample from the curve
            
        Returns:
            Tuple of (wandb plot, None)
        """
        len_res = len(residuals)
        if len_res == 0:
            return None, None
            
        # Sample points evenly across iterations
        step = max(1, (len_res // num_samples))
        samples = list(range(0, len_res, step))
        if samples[-1] != len_res - 1:
            samples.append(len_res - 1)

        # Create pandas DataFrame for plotting
        df = pd.DataFrame({
            "#iterations": samples,
            "Normalized Residual": [residuals[x] for x in samples],
        })
        
        # Create wandb table and plot
        table = wandb.Table(
            data=[[x, np.log10(residuals[x])] for x in samples],
            columns=["#iterations", "Normalized Residual (L2 norm) (log10)"]
        )
        
        curve_wandb = wandb.plot.line(
            table,
            "#iterations", 
            "Normalized Residual (L2 norm) (log10)",
            title=title
        )
        
        return curve_wandb, None

    def _process_residuals(self, residuals: torch.Tensor) -> List[float]:
        """Process residuals tensor into list of floats for plotting.
        
        This helper method safely converts residual tensors to Python lists,
        handling None values and different input types.
        
        Args:
            residuals: Tensor of residual values or None
            
        Returns:
            List of float residual values, empty list if input is None
        """
        if residuals is None:
            return []
            
        if isinstance(residuals, torch.Tensor):
            return residuals.detach().cpu().numpy().tolist()
            
        return list(residuals)

    def _update_residual_logs(self, logs: Dict[str, Any],
                            prefix: str,
                            residuals: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Update logs with residual information.
        
        Processes residuals and adds various metrics to the logs dictionary:
        - Convergence curve plot
        - Final residual value
        - Average residual value
        - Number of iterations
        
        Args:
            logs: Dictionary of logs to update
            prefix: Prefix for log keys (e.g. 'forward' or 'backward')
            residuals: Tensor of residual values
            
        Returns:
            Updated logs dictionary
        """
        if residuals is None:
            return logs
            
        residuals_list = self._process_residuals(residuals)
        if not residuals_list:
            return logs
            
        curve, _ = self._get_convergence_curve(
            residuals_list,
            f"{prefix} Residual Convergence"
        )
        
        logs.update({
            f"{prefix}_residual_curve": curve,
            f"{prefix}_residual_curve_plotly": None,
            f"{prefix}_residual_final": residuals_list[-1],
            f"{prefix}_residual_avg": float(np.mean(residuals_list)),
            f"{prefix}_num_iter": len(residuals_list)
        })
        
        return logs

    def MP_VIN_validation_step(self, sample, batch_idx):
        if self.args.task == "VisNav":
            # enable mapping
            pano_obs = sample.pano_obs
            pred_map_logits = self.mapper(pano_obs)
            # calculate the map loss first
            if self.args.has_obstacle:
                map_loss = self.map_loss(pred_map_logits[:, 0], sample.x[:, 3])
                self.log(
                    "val/map_loss", map_loss, on_epoch=True, batch_size=self.batch_size
                )

            pred_map = F.sigmoid(pred_map_logits)
            # change the obstacle to the predicted using visual input
            sample.x = torch.cat([sample.x[:, :3].detach(), pred_map], axis=1)
        elif self.args.task == "Habitat":
            # img_feat = self.get_visual_feat(sample)
            # TODO use encoder
            img_feat = self.img_encoder(sample)
            sample.x = torch.cat([sample.x[:, :2], img_feat], axis=1)  # 2+512

        if "Sym" in self.args.algorithm:
            self.model.train()  # fix escnn issue

        if self.args.task == "Habitat":
            # TODO: Make this work for habitat
            out = self.model(sample, visualize=False)
        else:
            out, our_v, gt_v = self.model(sample, visualize=True)
            if wandb.run is not None:
                wandb.log(
                    {
                        "val/our_v_fig": wandb.Image(our_v),
                        "val/gt_v_fig": wandb.Image(gt_v),
                    }
                )

        logits, probs = out.logits, out.probs
        if self.args.cont_action:
            loss = self.cont_action_loss(logits, sample.labels)
        else:
            loss = self.discrete_action_loss(logits, torch.argmax(sample.labels, dim=1))

        if self.args.task == "VisNav" and self.args.has_obstacle:
            loss += map_loss
        self.log("val/loss", loss, on_epoch=True, batch_size=self.batch_size)

        # calculate accuracy & success rate
        # pred = self.cont_action_to_discrete(probs) if self.args.cont_action else probs.detach()
        # labels = self.cont_action_to_discrete(sample.labels) if self.args.cont_action else sample.labels.detach()
        # acc = self.accuracy(labels, pred) # we need discrete for this
        # self.log("val/acc", acc, on_epoch=True, batch_size=self.batch_size)

        pred = (
            self.discrete_action_to_cont(probs)
            if not self.args.cont_action
            else probs.detach()
        )
        labels = (
            self.discrete_action_to_cont(sample.labels)
            if not self.args.cont_action
            else sample.labels.detach()
        )
        if self.args.obsv_mode == "full":
            self.log(
                "val/success",
                self.graph_success_rate(sample, pred),
                on_epoch=True,
                batch_size=self.batch_size,
            )  # monitor this value for checkpoint
        else:
            self.log(
                "val/success",
                self.partial_graph_success_rate(sample, pred),
                on_epoch=True,
                batch_size=self.batch_size,
            )  # monitor this value for checkpoint

        # if wandb.run is not None:
        #     info_convergence = self._plot_convergence_curve()
        #     wandb.log(info_convergence)

        return loss

    def MP_VIN_test_step(self, sample, batch_idx):
        if self.args.test_equiv_err:
            equiv_err_q, equiv_err_r, equiv_err_v = self._equiv_err(
                sample, self.args.equiv_err_theta
            )
            self.log(
                "test/equiv_err_theta",
                self.args.equiv_err_theta * (np.pi * 2) / 4,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.log(
                "test/equiv_err_q",
                equiv_err_q,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.log(
                "test/equiv_err_r",
                equiv_err_r,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.log(
                "test/equiv_err_v",
                equiv_err_v,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            # self.log("test/equiv_err_h", equiv_err_h, on_epoch=True, batch_size=self.batch_size)
            return equiv_err_q, equiv_err_r, equiv_err_v
        else:
            if self.args.task == "VisNav":
                # enable mapping
                pano_obs = sample.pano_obs
                pred_map_logits = self.mapper(pano_obs)
                # calculate the map loss first
                map_loss = self.map_loss(pred_map_logits[:, 0], sample.x[:, 3])
                self.log(
                    "test/map_loss", map_loss, on_epoch=True, batch_size=self.batch_size
                )

                pred_map = F.sigmoid(pred_map_logits)
                # change the obstacle to the predicted using visual input
                sample.x = torch.cat([sample.x[:, :3].detach(), pred_map], dim=1)
            elif self.args.task == "Habitat":
                # img_feat = self.get_visual_feat(sample)
                # TODO use encoder
                img_feat = self.img_encoder(sample)
                sample.x = torch.cat([sample.x[:, :2], img_feat], dim=1)  # 2+512

            if "Sym" in self.args.algorithm:
                self.model.train()  # fix escnn issue

            out = self.model(sample, visualize=False)
            logits, probs = out.logits, out.probs
            if self.args.cont_action:
                loss = self.cont_action_loss(logits, sample.labels)
            else:
                loss = self.discrete_action_loss(
                    logits, torch.argmax(sample.labels, dim=1)
                )

            if self.args.task == "VisNav":
                loss += map_loss
            self.log("test/loss", loss, on_epoch=True, batch_size=self.batch_size)

            # acc = self.accuracy(labels, pred)
            # self.log("test/acc", acc, on_epoch=True, batch_size=self.batch_size)  # monitor this value for checkpoint

            pred = (
                self.discrete_action_to_cont(probs)
                if not self.args.cont_action
                else probs.detach()
            )
            labels = (
                self.discrete_action_to_cont(sample.labels)
                if not self.args.cont_action
                else sample.labels.detach()
            )
            self.log(
                "test/success",
                self.graph_success_rate(sample, pred),
                on_epoch=True,
                batch_size=self.batch_size,
            )
            return loss

    ################################################################################
    # Grid-based method starts from here
    ################################################################################

    def grid_sample_traj(self, maze_map, opt_policy):
        # sample a random trajectory on a grid
        # return the start node and the length
        H, W = maze_map.shape
        obstacle_value = maze_map.min()  # the value of the obstacle (should be 0)
        while True:
            # randomly choose a valid start position (not obstacle & not the goal)
            rand_H, rand_W = np.random.randint(0, H), np.random.randint(0, W)
            if (
                maze_map[rand_H, rand_W] > obstacle_value
                and opt_policy[rand_H, rand_W] < 0
            ):
                break
        return (rand_H, rand_W), int(-opt_policy[rand_H, rand_W])

    def grid_NEWS_transition_helper(self, maze_map: torch.Tensor, 
                                 curr_node: Tuple[int, int],
                                 action: torch.Tensor) -> Tuple[int, int]:
        """Helper function for grid transitions.
        
        Takes the current grid position and action, returns the next position
        while respecting maze boundaries and obstacles.
        
        Args:
            maze_map: Map of the maze environment
            curr_node: Current (row, col) position
            action: Action probabilities for NWSE directions
            
        Returns:
            Next (row, col) position after taking the action
        """
        H, W = maze_map.shape
        delta = np.array(self.action_order[torch.argmax(action).item()])
        obstacle_value = maze_map.min()  # Value representing obstacles (usually 0)
        
        # Check if next position would be out of bounds
        next_row = curr_node[0] + delta[0]
        next_col = curr_node[1] + delta[1]
        if (next_row >= H or next_row < 0 or 
            next_col >= W or next_col < 0):
            return curr_node
            
        # Check if next position is an obstacle
        if maze_map[next_row, next_col] != obstacle_value:
            return (next_row, next_col)
            
        return curr_node

    def grid_success_rate(self, maze_map: torch.Tensor, 
                          goal_map: torch.Tensor,
                          opt_values: torch.Tensor,
                          probs: torch.Tensor) -> Tuple[float, float]:
        """Calculate success rate for grid navigation.
        
        Args:
            maze_map: Tensor of maze layout
            goal_map: Tensor indicating goal locations
            opt_values: Tensor of optimal values
            probs: Tensor of action probabilities
            
        Returns:
            Tuple of (success rate, average error)
        """
        batch_size = maze_map.shape[0]
        success = 0
        error = 0.0
        
        for i in range(batch_size):
            curr_node = (0, 0)  # Start from top-left
            path_length = 0
            max_steps = maze_map.shape[1] * maze_map.shape[2]  # H * W
            
            while path_length < max_steps:
                # Check if reached goal
                if goal_map[i, 0, curr_node[0], curr_node[1]] == 1.0:
                    success += 1
                    break
                    
                # Take action based on policy
                action = probs[i, :, curr_node[0], curr_node[1]]
                curr_node = self.grid_NEWS_transition_helper(
                    maze_map[i, 0],  # Remove channel dimension
                    curr_node,
                    action
                )
                path_length += 1
                
            # Calculate error if didn't reach goal
            if path_length == max_steps:
                error += float(opt_values[i, 0, curr_node[0], curr_node[1]])
                
        return success / batch_size, error / batch_size

    def grid_graph_world_success_rate(self, sample, probs):
        # calculate success rate for VIN/SymVIN on GraphWorld task
        # probs is the grid probs not the graph probs
        total = 0
        success = 0
        for i in range(sample.batch.max() + 1):
            start_node, length = self.graph_sample_traj(sample[i])
            policy = probs[i]
            # convert to networkx for easier operation
            nx_g = torch_geometric.utils.to_networkx(
                sample[i], to_undirected=True, node_attrs=["x", "opt_v"]
            )
            curr_node = start_node
            for _ in range(2 * length):
                # transform curr_node from graph position to grid position
                curr_node_idx = torch.floor(sample[i].x[curr_node, :2]).int()
                action = policy[..., curr_node_idx[0], curr_node_idx[1]].T
                curr_node = self.graph_transition_helper(
                    nx_g, curr_node, self.discrete_action_to_cont(action)[0]
                )
                if nx_g.nodes[curr_node]["x"][2] == 1.0:
                    success += 1
                    break
            total += 1
        return success / total * 100.0

    def get_graph_world_data(self, sample):
        B = sample.batch.max() + 1
        maze_map = np.ones((B, 15, 15))
        goal_map = np.zeros((B, 1, 15, 15))
        opt_value = np.zeros((B, 1, 15, 15))
        opt_policy = np.zeros((B, 4, 1, 15, 15))
        for i in range(B):
            # discretize the graph
            graph = sample[i].x.detach().cpu().numpy()
            x_floor = np.floor(graph[:, :2]).astype(int)
            _maze_map = np.ones((15, 15))
            _goal_map = np.zeros((1, 15, 15))
            obstacle_idx = graph[:, 3] == 1
            _maze_map[x_floor[obstacle_idx, 0], x_floor[obstacle_idx, 1]] = 0.0
            goal_idx = graph[:, 2] == 1
            _goal_map[0, x_floor[goal_idx, 0], x_floor[goal_idx, 1]] = 1.0

            # generate ground-truth policy
            mechanism = get_mechanism("4abs-cc")
            _opt_value = dijkstra_dist(
                _maze_map, mechanism, extract_goal(_goal_map, mechanism, 15)
            )
            _opt_policy = extract_policy(
                _maze_map, mechanism, _opt_value, is_full_policy=False
            )

            maze_map[i] = _maze_map
            goal_map[i] = _goal_map
            opt_value[i] = _opt_value
            opt_policy[i] = _opt_policy

        maze_map = torch.tensor(maze_map, device=sample.x.device, dtype=sample.x.dtype)
        goal_map = torch.tensor(goal_map, device=sample.x.device, dtype=sample.x.dtype)
        opt_value = torch.tensor(
            opt_value, device=sample.x.device, dtype=sample.x.dtype
        )
        opt_policy = torch.tensor(
            opt_policy, device=sample.x.device, dtype=sample.x.dtype
        )

        return maze_map, goal_map, opt_value, opt_policy

    def VIN_training_step(self, sample, batch_idx):
        if self.args.task == "GraphWorld":
            maze_map, goal_map, opt_value, opt_policy = self.get_graph_world_data(
                sample
            )
        else:
            maze_map, goal_map, opt_policy, opt_values = (
                sample["maze"],
                sample["goal_map"],
                sample["opt_policy"],
                sample["opt_values"],
            )
        if self.args.task == "VisNav":
            # enable mapping
            pano_obs = sample["pano_obs"]
            pred_map_logits = self.mapper(pano_obs)
            # calculate the map loss first
            if self.args.has_obstacle:
                batch_size = pred_map_logits.size(0)
                flat_pred_map = pred_map_logits.squeeze(1)
                flat_pred_map = flat_pred_map.reshape(batch_size, -1)
                flat_target_map = maze_map.squeeze(1)
                flat_target_map = flat_target_map.reshape(batch_size, -1)
                map_loss = self.map_loss(flat_pred_map, flat_target_map)
                self.log(
                    "val/map_loss", map_loss, on_epoch=True, batch_size=self.batch_size
                )
            maze_map = F.sigmoid(pred_map_logits).squeeze(1)

        out = self.model(maze_map, goal_map)
        logits, probs = out.logits, out.probs
        labels, loss = self._compute_planner_loss(opt_policy, logits)

        if self.args.task == "VisNav" and self.args.has_obstacle:
            loss += map_loss

        self.log("train/loss", loss, batch_size=self.batch_size)
        if self.args.task == "GraphWorld":
            self.log(
                "train/success",
                self.grid_graph_world_success_rate(sample, probs),
                on_epoch=True,
                batch_size=self.batch_size,
            )
        else:
            self.log(
                "train/success",
                self.grid_success_rate(maze_map, goal_map, opt_values, probs),
                on_epoch=True,
                batch_size=self.batch_size,
            )

        return loss

    def VIN_validation_step(self, sample, batch_idx):
        if self.args.task == "GraphWorld":
            maze_map, goal_map, opt_value, opt_policy = self.get_graph_world_data(
                sample
            )
        else:
            maze_map, goal_map, opt_policy, opt_values = (
                sample["maze"],
                sample["goal_map"],
                sample["opt_policy"],
                sample["opt_values"],
            )

        if self.args.task == "VisNav":
            # enable mapping
            pano_obs = sample["pano_obs"]
            pred_map_logits = self.mapper(pano_obs)
            # calculate the map loss first
            if self.args.has_obstacle:
                batch_size = pred_map_logits.size(0)
                flat_pred_map = pred_map_logits.squeeze(1)
                flat_pred_map = flat_pred_map.reshape(batch_size, -1)
                flat_target_map = maze_map.squeeze(1)
                flat_target_map = flat_target_map.reshape(batch_size, -1)
                map_loss = self.map_loss(flat_pred_map, flat_target_map)
                self.log(
                    "val/map_loss", map_loss, on_epoch=True, batch_size=self.batch_size
                )
            maze_map = F.sigmoid(pred_map_logits).squeeze(1)

        out = self.model(maze_map, goal_map)
        logits, probs = out.logits, out.probs
        labels, loss = self._compute_planner_loss(opt_policy, logits)

        if self.args.task == "VisNav" and self.args.has_obstacle:
            loss += map_loss

        acc = (
            torch.sum(torch.argmax(logits, dim=1) == torch.argmax(opt_policy, dim=1))
            / (len(logits.flatten()) / 4)
            * 100.0
        )
        self.log("val/loss", loss, on_epoch=True, batch_size=self.batch_size)
        # monitor this value for checkpoint
        self.log("val/acc", acc, on_epoch=True, batch_size=self.batch_size)
        if self.args.task == "GraphWorld":
            self.log(
                "val/success",
                self.grid_graph_world_success_rate(sample, probs),
                on_epoch=True,
                batch_size=self.batch_size,
            )
        else:
            self.log(
                "val/success",
                self.grid_success_rate(maze_map, goal_map, opt_values, probs),
                on_epoch=True,
                batch_size=self.batch_size,
            )

        return loss

    def VIN_test_step(self, sample, batch_idx):
        if self.args.task == "GraphWorld":
            maze_map, goal_map, opt_value, opt_policy = self.get_graph_world_data(
                sample
            )
        else:
            maze_map, goal_map, opt_policy, opt_values = (
                sample["maze"],
                sample["goal_map"],
                sample["opt_policy"],
                sample["opt_values"],
            )

        if self.args.task == "VisNav":
            # enable mapping
            pano_obs = sample["pano_obs"]
            pred_map_logits = self.mapper(pano_obs)
            # calculate the map loss first
            if self.args.has_obstacle:
                batch_size = pred_map_logits.size(0)
                flat_pred_map = pred_map_logits.squeeze(1)
                flat_pred_map = flat_pred_map.reshape(batch_size, -1)
                flat_target_map = maze_map.squeeze(1)
                flat_target_map = flat_target_map.reshape(batch_size, -1)
                map_loss = self.map_loss(flat_pred_map, flat_target_map)
                self.log(
                    "val/map_loss", map_loss, on_epoch=True, batch_size=self.batch_size
                )
            maze_map = F.sigmoid(pred_map_logits).squeeze(1)

        out = self.model(maze_map, goal_map)
        logits, probs = out.logits, out.probs
        labels, loss = self._compute_planner_loss(opt_policy, logits)

        if self.args.task == "VisNav" and self.args.has_obstacle:
            loss += map_loss

        acc = (
            torch.sum(torch.argmax(logits, dim=1) == torch.argmax(opt_policy, dim=1))
            / (len(logits.flatten()) / 4)
            * 100.0
        )
        self.log("test/loss", loss, on_epoch=True, batch_size=self.batch_size)
        # monitor this value for checkpoint
        self.log("test/acc", acc, on_epoch=True, batch_size=self.batch_size)
        if self.args.task == "GraphWorld":
            self.log(
                "test/success",
                self.grid_graph_world_success_rate(sample, probs),
                on_epoch=True,
                batch_size=self.batch_size,
            )
        else:
            self.log(
                "test/success",
                self.grid_success_rate(maze_map, goal_map, opt_values, probs),
                on_epoch=True,
                batch_size=self.batch_size,
            )

        return loss

class LitGraphPlanner(LitBase):
    def __init__(self, model, args) -> None:
        super().__init__(args)
        self.model = model
        self.batch_size = args.batch_size
        self.args = args
        # Some models use CrossEntropyLoss, but it's not actually used anywhere
        # self.loss_fn = nn.CrossEntropyLoss()  
        self.save_hyperparameters(ignore="model")

    def training_step(self, sample, batch_idx):
        return self.MP_VIN_training_step(sample, batch_idx)

    def validation_step(self, sample, batch_idx):
        return self.MP_VIN_validation_step(sample, batch_idx)

    def test_step(self, sample, batch_idx):
        return self.MP_VIN_test_step(sample, batch_idx)

class LitGridPlanner(LitBase):
    """Base class for grid-based planners like VIN and SymVIN"""
    def __init__(self, model, args) -> None:
        super().__init__(args)
        self.model = model
        self.criterion_planner = nn.CrossEntropyLoss()
        self.label = "one_hot"
        self.batch_size = args.batch_size
        self.save_hyperparameters(ignore="model")

    def _compute_planner_loss(self, opt_policy, logits):
        """
        Compute loss for planning (and mapping)
        """
        flat_logits = logits.transpose(1, 4).contiguous()
        flat_logits = flat_logits.view(-1, flat_logits.size()[-1]).contiguous()

        if self.label == "one_hot":
            _, labels = opt_policy.max(1, keepdim=True)
            flat_labels = labels.transpose(1, 4).contiguous()
            flat_labels = flat_labels.view(-1).contiguous()

        elif self.label == "full":
            opt_policy_shape = opt_policy.shape
            labels = opt_policy.reshape(opt_policy_shape[0], opt_policy_shape[1], -1)
            labels = labels / labels.sum(1).unsqueeze(1)
            labels = labels.reshape(opt_policy_shape)
            flat_labels = opt_policy.transpose(1, 4).contiguous()
            flat_labels = flat_labels.view(-1, flat_labels.size()[-1]).contiguous()

        else:
            raise NotImplementedError("Only support for label 'one_hot' or 'full'.")

        # > imitation learning loss for planning
        loss = self.criterion_planner(flat_logits, flat_labels)

        return labels, loss

    def training_step(self, sample, batch_idx):
        loss = self.VIN_training_step(sample, batch_idx)
        return loss

    def validation_step(self, sample, batch_idx):
        loss = self.VIN_validation_step(sample, batch_idx)
        return loss

    def test_step(self, sample, batch_idx):
        loss = self.VIN_test_step(sample, batch_idx)
        return loss
