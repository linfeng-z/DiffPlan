import importlib
import random

import numpy as np
from collections import defaultdict
from itertools import chain

import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import wandb

import plotly.express as px

from diffplan.modules.helpers import EquivariantDebugReturn, NormalDebugReturn, Timer
from diffplan.utils.dijkstra import dijkstra_dist, dijkstra_policy
from diffplan.envs.maze_utils import extract_goal
from diffplan.utils.experiment import get_optimizer
from diffplan.utils.vis_fields import visualize_policy_field


class Runner:
    """
    The Runner class runs a planner model on a given dataset and records
    statistics such as loss, prediction error, % Optimal, and % Success.
    """

    def __init__(self, args, mechanism, maze_size, data_size, verbose=True):
        """
        Args:
          model (torch.nn.Module): The Planner model
          mechanism (utils.mechanism.Mechanism): Environment transition kernel
          args (Namespace): Arguments
        """
        self.args = args

        self.datafile = args.datafile
        self.maze_size = maze_size

        self.use_gpu = args.use_gpu
        self.clip_grad = args.clip_grad
        self.lr_decay = args.lr_decay
        self.use_percent_successful = args.use_percent_successful

        self.mechanism = mechanism
        self.label = args.label  # 'one_hot' or 'full'

        # > Loss function for planner and mapper
        self.criterion_planner = nn.CrossEntropyLoss()
        if args.mapper_loss_func == "BCE":
            self.criterion_mapper = nn.BCEWithLogitsLoss()
        elif args.mapper_loss_func == "MSE":
            self.criterion_mapper = (
                nn.MSELoss()
            )  # ToDo find the proper loss for the manipulation mapper
        else:
            raise ValueError

        # Instantiate the model
        model_module = importlib.import_module(args.model)
        self.model = model_module.Planner(
            mechanism.num_orient, mechanism.num_actions, args
        )
        self.best_model = model_module.Planner(
            mechanism.num_orient, mechanism.num_actions, args
        )

        if verbose:
            print(self.model)
        if self.args.enable_wandb and self.args.wandb_watch:
            wandb.watch(self.model)

        # Setup for the mapper
        self.mapper_type = args.mapper
        self.mapper_loss = args.mapper_loss
        self.mapper_l_ratio = args.mapper_l_ratio
        self.enable_mapping = args.mapper is not None

        if self.enable_mapping:
            mapper_module = importlib.import_module(
                "modules.mapper"
            )  # moved to a separate folder
            mapper_module = getattr(mapper_module, self.mapper_type)
            self.mapper = mapper_module(
                maze_size,
                maze_size,
                args.num_views,
                args.img_height,
                args.img_width,
                workspace_size=args.workspace_size,
            )
            if verbose:
                print(self.mapper)
            if self.args.enable_wandb and self.args.wandb_watch:
                wandb.watch(self.mapper)

        # Load model from file if provided
        if args.model_path != "":
            if not args.model_path.endswith(".pth"):
                args.model_path += "/planner.final.pth"
            saved_model = torch.load(args.model_path)
            if args.load_best:
                self.model.load_state_dict(saved_model["best_model"])
            else:
                self.model.load_state_dict(saved_model["model"])
            self.best_model.load_state_dict(saved_model["best_model"])
        else:
            self.best_model.load_state_dict(self.model.state_dict())

        # Track the best performing model so far
        self.best_metric = 0.0

        # Use GPU if available
        if self.use_gpu:
            self.model = self.model.cuda()
            self.best_model = self.best_model.cuda()
            if self.enable_mapping:
                self.mapper = self.mapper.to("cuda")

        # > Initialize optimizer, note also for mapper!
        if self.enable_mapping:
            self.optimizer, self.scheduler = get_optimizer(
                args,
                chain(self.mapper.parameters(), self.model.parameters()),
                data_size,
            )
        else:
            self.optimizer, self.scheduler = get_optimizer(
                args, self.model.parameters(), data_size
            )

    def _compute_stats(
        self,
        batch_size,
        maze_map,
        goal_map,
        logits,
        probs,
        labels,
        opt_policy,
        sample=False,
    ):
        # Select argmax policy
        _, pred_pol = torch.max(logits, dim=1, keepdim=True)

        # Convert to numpy arrays
        maze_map = maze_map.cpu().numpy()
        goal_map = goal_map.cpu().numpy()
        logits = logits.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        # labels = labels.detach().cpu().numpy()
        opt_policy = opt_policy.detach().cpu().numpy()
        pred_pol = pred_pol.cpu().numpy()

        max_pred = (probs == probs.max(axis=1)[:, None]).astype(np.float32)
        match_action = np.sum((max_pred != opt_policy).astype(np.float32), axis=1)
        match_action = (match_action == 0).astype(np.float32)
        match_action = np.reshape(match_action, (batch_size, -1))
        batch_error = 1 - np.mean(match_action)

        def calc_optimal_and_success(i):
            # Get current sample
            md = maze_map[i][0]
            gm = goal_map[i]
            op = opt_policy[i]
            pp = pred_pol[i][0]
            # ll = labels[i][0]

            # Extract the goal in 2D coordinates
            goal = extract_goal(gm)

            # Check how different the predicted policy is from the optimal one
            # in terms of path lengths
            pred_dist = dijkstra_policy(md, self.mechanism, goal, pp)
            opt_dist = dijkstra_dist(md, self.mechanism, goal)
            diff_dist = pred_dist - opt_dist

            wall_dist = np.min(pred_dist)  # impossible distance
            reachable_state = (
                opt_dist > wall_dist
            )  # reachable state is more accurate than empty states,
            # for some of empty states are not reachable
            reachable_state = reachable_state[0]

            # > SPL metric (Success rate weighted by (normalized inverse) Path Length)
            pred_dist[pred_dist == 0.0] = 1.0  # for goal pos, pred_dict should be 0.
            ratio_dist = opt_dist / pred_dist  # negative signs cancelled
            spl_state = np.nan_to_num(ratio_dist)  # in case of nan, to 0

            # > success = 1. reachable and indeed reached, or 2. reachable and NOT failed to reach (fail = worst dist)
            success_state = reachable_state & (pred_dist > wall_dist)
            # success_state = reachable_state ^ (pred_dist == wall_dist)

            for o in range(self.mechanism.num_orient):
                # Refill the walls in the difference with the impossible distance
                diff_dist[o] += (1 - reachable_state) * wall_dist

                # Mask out the walls in the prediction distances
                pred_dist[o] = pred_dist[o] - np.multiply(
                    1 - reachable_state, pred_dist[o]
                )

                # > Mask out unreachable states for SPL
                spl_state[o] = spl_state[o] - np.multiply(
                    1 - reachable_state, spl_state[o]
                )
                # > success weighted by (relative) path length
                spl_state[o] = spl_state[o] * success_state

            num_open = (
                reachable_state.sum() * self.mechanism.num_orient
            )  # number of reachable locations
            _po = (diff_dist == 0).sum() / num_open
            _ps = 1.0 - (pred_dist == wall_dist).sum() / num_open
            _spl = spl_state.sum() / num_open  # averaged SPL
            return _po, _ps, _spl

        if sample:
            percent_optimal, percent_successful, percent_spl = calc_optimal_and_success(
                np.random.randint(batch_size)
            )
        else:
            percent_optimal, percent_successful, percent_spl = 0, 0, 0
            for i in range(batch_size):
                po, ps, spl = calc_optimal_and_success(i)
                percent_optimal += po
                percent_successful += ps
                percent_spl += spl
            percent_optimal = percent_optimal / float(batch_size)
            percent_successful = percent_successful / float(batch_size)
            percent_spl = percent_spl / float(batch_size)

        return {
            "error": batch_error,
            "optimal": percent_optimal,
            "success": percent_successful,
            "SPL": percent_spl,
        }

    def _get_convergence_curve(self, residuals, title, num_samples=10):
        len_res = len(residuals)
        samples = range(0, len_res, (len_res // num_samples) + 1)

        # > Use plotly to plot curves - can view by #epochs
        d = {
            "#iterations": samples,
            "Normalized Residual": [residuals[x] for x in samples],
        }
        df = pd.DataFrame(d)
        curve_plotly = px.line(
            df, x="#iterations", y="Normalized Residual", log_y=True, title=None
        )

        # > Use wandb native plot - averaged across epochs?
        table = wandb.Table(
            data=[[x, np.log10(residuals[x])] for x in samples],
            columns=["#iterations", "Normalized Residual (L2 norm) (log10)"],
        )
        curve_wandb = wandb.plot.line(
            table, "#iterations", "Normalized Residual (L2 norm) (log10)", title=title
        )

        return curve_wandb, curve_plotly

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

        if "DE" in self.args.model:
            assert len(info) > 0

        return info

    def _forward(self, goal_map, maze_map, obs=None, debug=False, include_mapper=True):
        if self.enable_mapping and include_mapper:
            # > If enabled mapping, use true goal map but predicted maze occupancy map
            pred_map_logits = self.mapper(obs)
            assert pred_map_logits.size() == maze_map.size()
            pred_map = (
                torch.sigmoid(pred_map_logits)
                if self.args.mapper2probability
                else pred_map_logits
            )
            out = self.model(pred_map, goal_map, debug)
            return out, pred_map_logits

        elif self.enable_mapping and not include_mapper:
            # model takes maze map while mapper takes obs
            pred_map_logits = self.mapper(obs)
            assert pred_map_logits.size() == maze_map.size()
            out = self.model(maze_map, goal_map, debug)
            return out, pred_map_logits

        else:
            # Forward pass using given map
            out = self.model(maze_map, goal_map, debug)
            return out, None

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

    def _compute_mapper_loss(self, pred_map, target_map):
        # > flat computation (checked: pytorch 1.11 has different behavior in CE loss...)
        batch_size = pred_map.size(0)
        flat_pred_map = pred_map.squeeze(1)
        flat_pred_map = flat_pred_map.reshape(batch_size, -1)
        flat_target_map = target_map.squeeze(1)
        flat_target_map = flat_target_map.reshape(batch_size, -1)

        # > auxiliary loss for mapping
        mapper_loss = self.criterion_mapper(flat_pred_map, flat_target_map)

        return mapper_loss

    @staticmethod
    def _visualize_planner(
        output, mechanism, maze_map, goal_map, batch_idx=0, channel_idx=0
    ):
        """
        Visualize value V(s),Q(s,a) and policy pi(s), and also the transformed version V(g.s),Q(g.s,g.a) and pi(g.s)
        Note that GPPN variants don't have separate V(s) = max_a Q(s,a)
        """

        # Note: v_geo is normally from regular repr, which has |G| channels (e.g., 8 for D4)
        if type(output) == EquivariantDebugReturn:
            q_value = output.q_geo.tensor
            v_value = output.v_geo.tensor
        elif type(output) == NormalDebugReturn:
            q_value = output.q
            v_value = output.v
        else:
            raise ValueError

        info = {
            "pred_value": px.imshow(
                q_value[batch_idx][channel_idx].detach().cpu().numpy(),
                color_continuous_scale="Blues_r",
            )
            if q_value is not None
            else None,
            "pred_v_value": px.imshow(
                v_value[batch_idx][channel_idx].detach().cpu().numpy(),
                color_continuous_scale="Blues_r",
            )
            if v_value is not None
            else None,
            # > a group average version (for multiple group channels)
            "pred_v_avg_value": px.imshow(
                v_value.mean(1)[batch_idx].detach().cpu().numpy(),
                color_continuous_scale="Blues_r",
            )
            if v_value is not None
            else None,
            "pred_policy": visualize_policy_field(
                mechanism,
                maze_map.detach().cpu().numpy(),
                goal_map.detach().cpu().numpy(),
                output.logits.detach().cpu(),  # still torch tensor
            ),
        }

        # > close figures
        plt.close("all")

        return info

    # @staticmethod
    def _visualize_mapper(self, maze_map, pred_map):
        pred_map = torch.sigmoid(pred_map) if self.args.mapper2probability else pred_map
        info = {
            "pred_map": px.imshow(
                pred_map[0].squeeze().detach().cpu().numpy(),
                color_continuous_scale="Blues_r",
            ),
            "maze_map": px.imshow(
                maze_map[0].squeeze().detach().cpu().numpy(),
                color_continuous_scale="Blues_r",
            ),
        }
        return info

    def _run(
        self,
        dataloader,
        train=False,
        batch_size=-1,
        store_best=False,
        sample_stats=True,
        include_mapper=True,
    ):
        """
        Runs the model on the given data.
        Args:
          model (torch.nn.Module): The Planner model
          dataloader (torch.utils.data.Dataset): Dataset loader
          train (bool): Whether to train the model
          batch_size (int): Only used if train=True
          store_best (bool): Whether to store the best model
        Returns:
          info (dict): Performance statistics, including
          info["avg_loss"] (float): Average loss
          info["avg_error"] (float): Average error
          info["avg_optimal"] (float): Average % Optimal
          info["avg_success"] (float): Average % Success
          info["weight_norm"] (float): Model weight norm, stored if train=True
          info["grad_norm"]: Gradient norm, stored if train=True
          info["is_best"] (bool): Whether the model is best, stored if store_best=True
        """

        info = defaultdict(lambda: 0.0)

        last_maze_map, last_goal_map, last_pred_map, last_obs = None, None, None, None

        for i, out in enumerate(dataloader):
            # Get input batch.
            maze_map, goal_map, opt_policy = (
                out["maze"],
                out["goal_map"],
                out["opt_policy"],
            )
            if "pano_obs" in out:
                obs = out["pano_obs"]
            else:
                obs = None

            if train:
                if maze_map.size()[0] != batch_size:
                    continue  # Drop those data, if not enough for a batch
                self.optimizer.zero_grad()  # Zero the parameter gradients
            else:
                batch_size = maze_map.size()[0]

            # Send tensor to GPU if available
            if self.use_gpu:
                maze_map = maze_map.cuda()
                goal_map = goal_map.cuda()
                opt_policy = opt_policy.cuda()
                if "pano_obs" in out:
                    obs = obs.cuda()

            # Reshape batch-wise if necessary
            if maze_map.dim() == 3:
                maze_map = maze_map.unsqueeze(1)

            # > Forward pass (potentially with mapper using observation)
            with Timer(name="forward", verbose=False) as fwd_t:
                out, pred_map = self._forward(
                    goal_map, maze_map, obs, include_mapper=include_mapper
                )

            logits, probs = out.logits, out.probs
            # > Note: renamed `outputs` to `logits` and `predictions` to `probs` (used in model)

            # > Compute loss (including mapper prediction loss)
            labels, planning_loss = self._compute_planner_loss(opt_policy, logits)
            # > Log loss & forward time (full forward; including mapper)
            info_loss = {
                "planner_loss": planning_loss.item(),
                "forward_time": fwd_t.interval,
            }

            if self.enable_mapping:
                mapper_loss = self._compute_mapper_loss(pred_map, maze_map)
                loss = 0.0
                assert self.args.planner_loss or self.args.mapper_loss
                if self.args.planner_loss:
                    loss += planning_loss
                if self.args.mapper_loss:
                    loss += self.mapper_l_ratio * mapper_loss
                info_loss.update({"mapper_loss": mapper_loss.item()})
            else:
                loss = planning_loss

            # > Jacobian regularization (optional) for DEQ-style VIN variants
            if train and (self.args.jacobian_reg or self.args.jacobian_log):
                if not hasattr(out, "info") or out.info["jac_loss"] is None:
                    print("> Warning: no Jacobian reg. loss")
                else:
                    # > directly get loss from the model class
                    jac_loss = out.info["jac_loss"]
                    info_loss.update(
                        {
                            "jac_reg_loss": jac_loss  # TODO add option to log Jac loss but no tracking grad!
                        }
                    )
                    if (
                        self.args.jacobian_reg
                        and random.random() < self.args.jac_reg_freq
                    ):
                        loss += self.args.jac_reg_weight * jac_loss

            info_loss.update({"loss": loss.item()})

            # Update parameters
            if train:
                # Backward pass
                with Timer(name="backward", verbose=False) as bwd_t:
                    loss.backward()
                # > Log backward time
                info_loss["backward_time"] = bwd_t.interval

                # Clip the gradient norm
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad
                    )
                    # torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), self.clip_grad)

                # Update parameters
                self.optimizer.step()

            # Compute loss and error
            info_stats = self._compute_stats(
                batch_size,
                maze_map,
                goal_map,
                logits,
                probs,
                labels,
                opt_policy,
                sample=sample_stats,
            )

            # > Aggregate info and sum over batches
            info_stats.update(info_loss)
            for k, v in info_stats.items():
                info["avg_" + k] += v

            # > prepare for use later
            last_maze_map, last_goal_map, last_pred_map, last_obs = (
                maze_map,
                goal_map,
                pred_map,
                obs,
            )

        for k in info.keys():
            # > only average over batches for rates/losses, not for time
            if k.startswith("avg_") and ("time" not in k):
                info[k] /= len(dataloader)

        # > scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()
            assert len(lr) == 1
            info["lr"] = lr[0]

        # > visualize for planner and mapper for each epoch (using last batch)
        if self.args.visualize_training_model:
            out, _ = self._forward(last_goal_map, last_maze_map, last_obs, debug=True)
            info.update(
                self._visualize_planner(
                    out, self.mechanism, last_maze_map, last_goal_map
                )
            )
        if self.enable_mapping and self.args.visualize_map:
            info.update(self._visualize_mapper(last_maze_map, last_pred_map))

        # > Log DEQ convergence loss (after backward) - use last batch (every epoch)
        if self.args.enable_wandb:
            info.update(self._plot_convergence_curve())

        if train:
            # Calculate weight norm
            with torch.no_grad():
                weight_norm = 0
                grad_norm = 0
                for p in self.model.parameters():
                    weight_norm += torch.norm(p.detach()) ** 2
                    if p.grad is not None:
                        grad_norm += torch.norm(p.grad.detach()) ** 2
                info["weight_norm"] = float(
                    np.sqrt(weight_norm.detach().cpu().numpy().item())
                )
                if (
                    self.args.planner_loss
                ):  # > don't compute grad norm if not using planning loss
                    info["grad_norm"] = float(
                        np.sqrt(grad_norm.detach().cpu().numpy().item())
                    )

        if store_best:
            # Was the validation accuracy greater than the best one?
            metric = (
                info["avg_success"]
                if self.use_percent_successful
                else info["avg_optimal"]
            )
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_model.load_state_dict(self.model.state_dict())
                info["is_best"] = True
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * self.lr_decay
                info["is_best"] = False
        return info

    def train(self, dataloader, batch_size, include_mapper):
        """
        Trains the model on the given training dataset.
        """
        return self._run(
            dataloader,
            train=True,
            batch_size=batch_size,
            sample_stats=True,
            include_mapper=include_mapper,
        )

    def validate(self, dataloader, include_mapper):
        """
        Evaluates the model on the given validation dataset. Stores the
        current model if it achieves the best validation performance.
        """
        # by default, samples in training while computes full stats in eval
        # however, it's too slow for larger maps
        # calculating optimal paths for all maps take over 90% time for 50x50 maps...
        store_best = not self.enable_mapping or (
            self.enable_mapping and include_mapper
        )  # when no mapper or (mapper
        # with include mapper) store the best model
        return self._run(
            dataloader,
            store_best=store_best,
            sample_stats=self.args.sample_stats,
            include_mapper=include_mapper,
        )

    def test(self, dataloader, use_best=False):
        """
        Tests the model on the given dataset.
        """
        if use_best:
            model = self.best_model
        else:
            model = self.model
        model.eval()

        with torch.no_grad():
            stats = self._run(dataloader, store_best=True, sample_stats=False)

        return stats
