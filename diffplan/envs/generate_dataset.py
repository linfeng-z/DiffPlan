"""
Generates a 2D maze dataset.

Example usage:
>>> python generate_dataset.py --output-path mazes.npz --mechanism news \
    --maze-size 9 --train-size 5000 --valid-size 1000 --test-size 1000
"""
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
import torch_geometric
import torch

from diffplan.envs.maze_env import RandomMaze
from diffplan.envs.maze_utils import extract_policy
from diffplan.utils.dijkstra import dijkstra_dist
from diffplan.utils.experiment import get_mechanism
from diffplan.utils.graph_converter import convert_maze, assign_edge_value
from diffplan.utils.helpers_maze import get_partial_obsv_mask

# from memory_profiler import profile
import torch_cluster
import networkx as nx
import random

# for debugging


def generate_data(
    filename,
    train_size,
    valid_size,
    test_size,
    mechanism,
    maze_size,
    workspace_size,
    min_decimation,
    max_decimation,
    label,
    start_pos=(1, 1),
    env_name=None,
    shuffle=False,
    cont_act=False,
    has_obstacle=False,
    obsv_mode="full",
):
    if env_name in ["RandomMaze", "RandomMazeGraph"]:
        env_class = RandomMaze
    elif env_name in ["Visual3DNav", "Visual3DNavGraph", "Visual3DNavRandomGraph"]:
        # Visual3DNav: Standard grid-based mini world
        # Visual3DNavGraph: convert grid-based to graph
        # Visual3DNavRandomGraph: graph-map mini world
        if env_name == "Visual3DNavRandomGraph":
            from diffplan.envs.visual_nav.nav_env import (
                VisNavGraphEnv,
            )  # move in to avoid requiring handling display

            env_class = VisNavGraphEnv
        else:
            from diffplan.envs.visual_nav.nav_env import (
                VisNavEnv,
            )  # move in to avoid requiring handling display

            env_class = VisNavEnv
    elif env_name == "RandomGraph":
        pass
    else:
        raise NotImplementedError(
            "The environment " + env_name + " is not implemented."
        )

    if env_name != "RandomGraph":
        env = env_class(
            mechanism,
            maze_size,
            maze_size,
            min_decimation,
            max_decimation,
            start_pos=start_pos,
        )

    if env_name in ["RandomMazeGraph", "Visual3DNavGraph", "Visual3DNavRandomGraph"]:
        # LHY: graph version of Grid-world
        print("Creating valid+test dataset...")
        valid_test_mazes = create_dataset(
            env_name,
            env,
            mechanism,
            maze_size,
            workspace_size,
            label,
            test_size + valid_size,
            cont_act=cont_act,
            has_obstacle=has_obstacle,
            obsv_mode=obsv_mode,
        )

        print("Creating training dataset...")
        train_mazes = create_dataset(
            env_name,
            env,
            mechanism,
            maze_size,
            workspace_size,
            label,
            # train_size, compare_mazes=valid_test_mazes
            train_size,
            cont_act=cont_act,
            has_obstacle=has_obstacle,
            obsv_mode=obsv_mode,
        )
    elif env_name == "RandomGraph":
        # LHY: Graph-world
        print("Creating valid+test dataset...")
        valid_test_mazes = create_graph_dataset(
            env_name,
            mechanism,
            maze_size,
            workspace_size,
            label,
            test_size + valid_size,
            has_obstacle=has_obstacle,
            cont_act=cont_act,
        )

        print("Creating training dataset...")
        train_mazes = create_graph_dataset(
            env_name,
            mechanism,
            maze_size,
            workspace_size,
            label,
            # train_size, compare_mazes=valid_test_mazes
            train_size,
            has_obstacle=has_obstacle,
            cont_act=cont_act,
        )
    elif env_name in ["Visual3DNav", "Arm2DoFsWorkSpaceEnv"]:
        # Generate test set first
        print("Creating valid+test dataset...")
        (
            valid_test_mazes,
            valid_test_goal_maps,
            valid_test_opt_policies,
            valid_test_opt_values,
            valid_test_obs,
        ) = create_dataset(
            env_name,
            env,
            mechanism,
            maze_size,
            workspace_size,
            label,
            test_size + valid_size,
            has_obstacle=has_obstacle,
            cont_act=cont_act,
        )

        # Generate train set while avoiding test geometries
        print("Creating training dataset...")
        (
            train_mazes,
            train_goal_maps,
            train_opt_policies,
            train_opt_values,
            train_obs,
        ) = create_dataset(
            env_name,
            env,
            mechanism,
            maze_size,
            workspace_size,
            label,
            train_size,
            compare_mazes=valid_test_mazes,
            has_obstacle=has_obstacle,
            cont_act=cont_act,
        )
    else:
        # Grid-world
        print("Creating valid+test dataset...")
        (
            valid_test_mazes,
            valid_test_goal_maps,
            valid_test_opt_policies,
            valid_test_opt_values,
        ) = create_dataset(
            env_name,
            env,
            mechanism,
            maze_size,
            workspace_size,
            label,
            test_size + valid_size,
            cont_act=cont_act,
            obsv_mode=obsv_mode,
        )

        print("Creating training dataset...")
        (
            train_mazes,
            train_goal_maps,
            train_opt_policies,
            train_opt_values,
        ) = create_dataset(
            env_name,
            env,
            mechanism,
            maze_size,
            workspace_size,
            label,
            train_size,
            compare_mazes=valid_test_mazes,
            cont_act=cont_act,
            obsv_mode=obsv_mode,
        )

    # Split valid and test
    valid_mazes = valid_test_mazes[0:valid_size]
    test_mazes = valid_test_mazes[valid_size:]
    if env_name in ["RandomMaze", "Visual3DNav"]:
        valid_goal_maps = valid_test_goal_maps[0:valid_size]
        test_goal_maps = valid_test_goal_maps[valid_size:]
        valid_opt_policies = valid_test_opt_policies[0:valid_size]
        test_opt_policies = valid_test_opt_policies[valid_size:]
        valid_opt_values = valid_test_opt_values[0:valid_size]
        test_opt_values = valid_test_opt_values[valid_size:]

    if env_name in ["Visual3DNav", "Arm2DoFsWorkSpaceEnv"]:
        valid_obs = valid_test_obs[0:valid_size]
        test_obs = valid_test_obs[valid_size:]

    if shuffle:
        (
            train_goal_maps,
            train_mazes,
            train_opt_policies,
            train_opt_values,
            valid_goal_maps,
            valid_mazes,
            valid_opt_policies,
            valid_opt_values,
            test_goal_maps,
            test_mazes,
            test_opt_policies,
            test_opt_values,
        ) = shuffle_dataset(
            env_name,
            train_size,
            valid_size,
            train_goal_maps,
            train_mazes,
            train_opt_policies,
            train_opt_values,
            valid_goal_maps,
            valid_mazes,
            valid_opt_policies,
            valid_opt_values,
            test_goal_maps,
            test_mazes,
            test_opt_policies,
            test_opt_values,
        )

    if env_name in [None, "RandomMaze"]:
        # Save to numpy
        np.savez_compressed(
            filename,
            train_mazes,
            train_goal_maps,
            train_opt_policies,
            train_opt_values,
            valid_mazes,
            valid_goal_maps,
            valid_opt_policies,
            valid_opt_values,
            test_mazes,
            test_goal_maps,
            test_opt_policies,
            test_opt_values,
        )
    elif env_name in ["Visual3DNav", "Arm2DoFsWorkSpaceEnv"]:
        np.savez_compressed(
            filename,
            train_mazes,
            train_goal_maps,
            train_opt_policies,
            train_opt_values,
            valid_mazes,
            valid_goal_maps,
            valid_opt_policies,
            valid_opt_values,
            test_mazes,
            test_goal_maps,
            test_opt_policies,
            test_opt_values,
            train_obs,
            valid_obs,
            test_obs,
        )

    elif env_name in [
        "RandomMazeGraph",
        "Visual3DNavGraph",
        "RandomGraph",
        "Visual3DNavRandomGraph",
    ]:
        torch.save([train_mazes, valid_mazes, test_mazes], filename)
    else:
        raise ValueError


def shuffle_dataset(
    env_name,
    train_size,
    valid_size,
    train_goal_maps,
    train_mazes,
    train_opt_policies,
    train_opt_values,
    valid_goal_maps,
    valid_mazes,
    valid_opt_policies,
    valid_opt_values,
    test_goal_maps,
    test_mazes,
    test_opt_policies,
    test_opt_values,
):
    # Re-shuffle
    mazes = np.concatenate((train_mazes, valid_mazes, test_mazes), 0)
    goal_maps = np.concatenate((train_goal_maps, valid_goal_maps, test_goal_maps), 0)
    opt_policies = np.concatenate(
        (train_opt_policies, valid_opt_policies, test_opt_policies), 0
    )
    opt_values = np.concatenate(
        (train_opt_values, valid_opt_values, test_opt_values), 0
    )

    shuffle_idx = np.random.permutation(mazes.shape[0])
    mazes = mazes[shuffle_idx]
    goal_maps = goal_maps[shuffle_idx]
    opt_policies = opt_policies[shuffle_idx]
    opt_values = opt_values[shuffle_idx]

    train_mazes = mazes[:train_size]
    train_goal_maps = goal_maps[:train_size]
    train_opt_policies = opt_policies[:train_size]
    train_opt_values = opt_values[:train_size]

    valid_mazes = mazes[train_size : train_size + valid_size]
    valid_goal_maps = goal_maps[train_size : train_size + valid_size]
    valid_opt_policies = opt_policies[train_size : train_size + valid_size]
    valid_opt_values = opt_values[train_size : train_size + valid_size]

    test_mazes = mazes[train_size + valid_size :]
    test_goal_maps = goal_maps[train_size + valid_size :]
    test_opt_policies = opt_policies[train_size + valid_size :]
    test_opt_values = opt_values[train_size + valid_size :]

    if env_name == "Visual3DNav":
        raise NotImplementedError

    return (
        train_goal_maps,
        train_mazes,
        train_opt_policies,
        train_opt_values,
        valid_goal_maps,
        valid_mazes,
        valid_opt_policies,
        valid_opt_values,
        test_goal_maps,
        test_mazes,
        test_opt_policies,
        test_opt_values,
    )


# @profile
def create_dataset(
    env_name,
    env,
    mechanism,
    maze_size,
    workspace_size,
    label,
    data_size,
    compare_mazes=None,
    cont_act=False,
    has_obstacle=False,
    obsv_mode="full",
):
    if obsv_mode == "full":
        mazes = np.zeros((data_size, maze_size, maze_size))
        goal_maps = np.zeros((data_size, mechanism.num_orient, maze_size, maze_size))
    else:
        mazes = np.zeros((data_size, 3, maze_size, maze_size))
        goal_maps = np.zeros((data_size, 3, maze_size, maze_size))
    opt_policies = np.zeros(
        (data_size, mechanism.num_actions, mechanism.num_orient, maze_size, maze_size)
    )
    opt_values = np.zeros((data_size, maze_size, maze_size))
    graphs = []

    if env_name == "Visual3DNav":
        env.reset()
        pano_obs_array = np.zeros(
            (
                data_size,
                maze_size,
                maze_size,
                env.nav_world.num_views,
                env.obs_height,
                env.obs_width,
                env.nav_world.num_rgb,
            ),
            dtype=np.uint8,  # > 8-bit images; keep consistent; save memory
        )

    maze_hash = {}

    if compare_mazes is not None:
        for i in range(compare_mazes.shape[0]):
            maze = compare_mazes[i]
            maze_key = hash_maze_to_string(maze)
            maze_hash[maze_key] = 1
    i = 0
    while i < data_size:
        maze, player_map, goal_map = env.reset()

        # visualize and save
        # plt.imshow(maze, cmap="plasma")
        # plt.axis("off")
        # plt.savefig("miniworld_maze.png", bbox_inches="tight")

        # goal cannot be on obstacle
        if maze[goal_map[0] == 1] == 0:
            continue

        maze_key = hash_maze_to_string(maze)

        # Make sure we sampled a unique maze from the compare set
        if hashed_check_maze_exists(maze_key, maze_hash):
            continue
        maze_hash[maze_key] = 1

        # detect if only contains obstacle
        if maze.max() < 1.0:
            continue

        # Use Dijkstra's to construct the optimal policy
        opt_value = dijkstra_dist(
            maze, mechanism, extract_goal(goal_map, mechanism, maze_size)
        )
        opt_policy = extract_policy(
            maze, mechanism, opt_value, is_full_policy=(label == "full")
        )

        # detect if obstacle is too much
        if (
            len(
                opt_value[
                    opt_value == -mechanism.num_orient * maze.shape[0] * maze.shape[1]
                ]
            )
            >= mechanism.num_orient * maze.shape[0] * maze.shape[1] / 2
        ):
            continue

        pos2pano = None
        if env_name in ["Visual3DNavGraph", "Visual3DNav", "Visual3DNavRandomGraph"]:
            pos2pano = env.render_all_pano()
            if pos2pano is None:
                # there is issue calculating ground-truth
                # find a new sample
                continue
            if env_name == "Visual3DNav":
                pano_obs_array[
                    i, ...
                ] = pos2pano  # > this step will have substantial memory use (since replace zeros?)
                print(
                    "> Generate full 3D egocentric panoramic views",
                    i,
                    pano_obs_array.shape,
                )

        # partial observation
        if obsv_mode == "partial":
            # randomly choose a valid start node (not obstacle & not the goal)
            while True:
                rand_node_pos = np.random.randint(0, maze_size, 2)
                if (
                    maze[rand_node_pos[0], rand_node_pos[1]] == 1
                    and goal_map[0, rand_node_pos[0], rand_node_pos[1]] == 0
                ):
                    break

            # init observation mask
            prev_obsv_mask = np.zeros((maze_size, maze_size))

            curr_node = rand_node_pos
            while goal_map[0, curr_node[0], curr_node[1]] == 0 and i < data_size:
                obsv_mask = get_partial_obsv_mask(maze, curr_node)
                prev_obsv_mask = np.logical_or(prev_obsv_mask, obsv_mask).astype(int)

                # one-hot representation for partial observation
                # [1, 0, 0] means unoccupied
                # [0, 1, 0] means occupied
                # [0, 0, 1] means unknown
                maze_onehot = np.zeros((3, *maze.shape))
                maze_masked = maze * prev_obsv_mask
                maze_onehot[0, np.logical_and(prev_obsv_mask == 1, maze == 1)] = 1
                maze_onehot[1, np.logical_and(prev_obsv_mask == 1, maze == 0)] = 1
                maze_onehot[2, prev_obsv_mask == 0] = 1

                # goal_map_onehot
                # [1, 0, 0] means is goal
                # [0, 1, 0] means is not goal
                # [0, 0, 1] means unknown
                goal_map_onehot = np.zeros((3, *maze.shape))
                goal_map_masked = goal_map * np.expand_dims(prev_obsv_mask, 0)
                goal_map_onehot[
                    0, np.logical_and(prev_obsv_mask == 1, goal_map[0] == 1)
                ] = 1
                goal_map_onehot[
                    1, np.logical_and(prev_obsv_mask == 1, goal_map[0] == 0)
                ] = 1
                goal_map_onehot[2, prev_obsv_mask == 0] = 1

                # > Store the demonstration data
                if env_name in ["RandomMazeGraph", "Visual3DNavGraph"]:
                    graph = convert_maze(
                        maze,
                        goal_map,
                        opt_policy,
                        mechanism,
                        opt_value,
                        cont_act=cont_act,
                        has_obstacle=has_obstacle,
                        pos2pano=pos2pano,
                        obsv_mode=obsv_mode,
                        maze_onehot=maze_onehot,
                        goal_onehot=goal_map_onehot,
                    )

                    if graph is None:
                        continue

                    graphs.append(graph)

                elif env_name in ["Visual3DNavRandomGraph"]:
                    graphs.append(pos2pano)

                mazes[i, :, :, :] = maze_onehot
                goal_maps[i, :, :, :] = goal_map_onehot
                opt_policies[i, :, :, :, :] = opt_policy
                opt_values[i, :, :] = opt_value

                # plt.imshow(maze_masked)
                # plt.show()

                i += 1
                # transit to next node using opt_policy
                action_order = [[-1, 0], [0, -1], [1, 0], [0, 1]]
                delta = action_order[
                    np.argmax(opt_policy[:, 0, curr_node[0], curr_node[1]])
                ]
                curr_node += delta

        # > Store the demonstration data
        if env_name in ["RandomMazeGraph", "Visual3DNavGraph"]:
            if obsv_mode == "partial":
                graph = convert_maze(
                    maze,
                    goal_map,
                    opt_policy,
                    mechanism,
                    opt_value,
                    cont_act=cont_act,
                    has_obstacle=has_obstacle,
                    pos2pano=pos2pano,
                    obsv_mode=obsv_mode,
                    maze_onehot=maze_onehot,
                    goal_onehot=goal_map_onehot,
                )
            else:
                graph = convert_maze(
                    maze,
                    goal_map,
                    opt_policy,
                    mechanism,
                    opt_value,
                    cont_act=cont_act,
                    has_obstacle=has_obstacle,
                    pos2pano=pos2pano,
                    obsv_mode=obsv_mode,
                )

            if graph is None:
                continue

            graphs.append(graph)

        elif env_name in ["Visual3DNavRandomGraph"]:
            graphs.append(pos2pano)

        if obsv_mode == "full":
            mazes[i, :, :] = maze
            goal_maps[i, :, :, :] = goal_map
            opt_policies[i, :, :, :, :] = opt_policy
            opt_values[i, :, :] = opt_value
            i += 1

        # print("\r%0.4f" % (float(i) / data_size * 100) + "%")
        sys.stdout.write("\r%0.4f" % (float(i) / data_size * 100) + "%")
        sys.stdout.flush()

    sys.stdout.write("\r100%\n")

    if env_name in ["RandomMazeGraph", "Visual3DNavGraph", "Visual3DNavRandomGraph"]:
        return graphs
    elif env_name == "Visual3DNav":
        env.close()
        return mazes, goal_maps, opt_policies, opt_values, pano_obs_array
    return mazes, goal_maps, opt_policies, opt_values


def create_graph_dataset(
    env_name,
    mechanism,
    maze_size,
    workspace_size,
    label,
    data_size,
    compare_mazes=None,
    cont_act=False,
    has_obstacle=True,
):
    graphs = []

    maze_hash = {}

    while len(graphs) < data_size:
        # randomly generate nodes
        nodes = torch.rand(maze_size, 2) * 15  # (x,y) => with in (0,15),(0,15)

        # radius graph
        radius_edge_index = torch_cluster.knn_graph(nodes, k=10)
        radius_data = torch_geometric.data.Data(x=nodes, edge_index=radius_edge_index)
        graph = torch_geometric.utils.to_networkx(radius_data)
        graph = graph.to_undirected()

        # randomly select obstacles
        obs_idc = random.sample(graph.nodes, maze_size // 10)
        obs = np.zeros((maze_size, 1))
        obs[obs_idc] = 1

        # randomly select goal that is not obstacle
        while True:
            goal_idx = random.sample(graph.nodes, 1)
            if goal_idx[0] not in obs_idc:
                break
            else:
                continue
        goal = np.zeros((maze_size, 1))
        goal[goal_idx] = 1

        # form the node feature (2D pos, goal, obstacle)
        graph.x = np.concatenate([nodes[:, :2].numpy(), goal, obs], axis=1)

        assign_edge_value(graph)
        length = nx.single_source_dijkstra_path_length(
            graph, goal_idx[0], weight="cost"
        )

        for _idx in length:
            if length[_idx] == np.inf:
                # we have node that couldn't reach the goal
                graph.remove_node(_idx)
                obs_idc.remove(_idx)

        length = nx.single_source_dijkstra_path_length(
            graph, goal_idx[0], weight="cost"
        )
        opt_v = np.zeros((len(graph.nodes), 1))
        for _idx, _key in enumerate(graph.nodes):
            opt_v[_idx] = -length[_key]
        opt_v[obs_idc] = -np.inf

        # construct the optimal policy
        for node in graph.nodes:
            best_neighbor = (-1, np.inf)
            for nei in graph.neighbors(node):
                if length[nei] < best_neighbor[1]:
                    best_neighbor = (nei, length[nei])
            relative_trans = graph.x[best_neighbor[0], :2] - graph.x[node, :2]
            graph.nodes[node]["labels"] = relative_trans

        graph_pyg = torch_geometric.utils.from_networkx(graph)
        graph_pyg.x = torch.tensor(graph.x, dtype=torch.float)
        graph_pyg.opt_v = torch.tensor(opt_v)
        graph_pyg.edge_index = torch_geometric.utils.add_self_loops(
            graph_pyg.edge_index
        )[0]
        graph_pyg.labels = graph_pyg.labels.float()
        graphs.append(graph_pyg)

        # print("\r%0.4f" % (float(len(graphs)) / data_size * 100) + "%")
        sys.stdout.write("\r%0.4f" % (float(len(graphs)) / data_size * 100) + "%")
        sys.stdout.flush()

    sys.stdout.write("\r100%\n")

    return graphs


def hash_maze_to_string(_maze):
    maze = np.array(_maze, dtype=np.uint8).reshape((-1))
    maze_key = ""
    for i in range(maze.shape[0]):
        maze_key += str(maze[i])
    return maze_key


def hashed_check_maze_exists(maze_key, maze_hash):
    if maze_hash is None:
        return False
    if maze_key in maze_hash:
        return True
    return False


def extract_goal(goal_map, mechanism, maze_size):
    for o in range(mechanism.num_orient):
        for y in range(maze_size):
            for x in range(maze_size):
                if goal_map[o][y][x] == 1.0:
                    return (o, y, x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="RandomMaze",
        help="Which environment to generate data.",
    )
    parser.add_argument(
        "--output-path", type=str, default="", help="Filename to save the dataset to."
    )
    parser.add_argument(
        "--train-size", type=int, default=10000, help="Number of training mazes."
    )
    parser.add_argument(
        "--valid-size", type=int, default=1000, help="Number of validation mazes."
    )
    parser.add_argument(
        "--test-size", type=int, default=1000, help="Number of test mazes."
    )
    parser.add_argument("--maze-size", type=int, default=9, help="Size of mazes.")
    parser.add_argument(
        "--workspace-size", type=int, default=96, help="Size of manipulator workspace."
    )
    parser.add_argument(
        "--label",
        type=str,
        default="one_hot",
        help="Optimal policy labeling. (one_hot|full)",
    )
    parser.add_argument(
        "--min-decimation",
        type=float,
        default=0.0,
        help="How likely a wall is to be destroyed (minimum).",
    )
    parser.add_argument(
        "--max-decimation",
        type=float,
        default=1.0,
        help="How likely a wall is to be destroyed (maximum).",
    )
    parser.add_argument(
        "--start-pos-x", type=int, default=1, help="Maze start X-axis position."
    )
    parser.add_argument(
        "--start-pos-y", type=int, default=1, help="Maze start Y-axis position."
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default="news",
        help="Maze transition mechanism. (news|news-wrap|4abs-cc|4abs-cc-wrap|diffdrive|moore)",
    )
    parser.add_argument(
        "--cont_action", action="store_true", help="Continous action (theta)"
    )
    parser.add_argument(
        "--has_obstacle", action="store_true", help="Include obstalce node in the graph"
    )
    parser.add_argument(
        "--obsv_mode",
        type=str,
        choices=["full", "partial"],
        default="full",
        help="Observation mode",
    )
    args = parser.parse_args()

    file_name = (
        args.env
        + "_"
        + str(args.workspace_size)
        + "_"
        + str(args.train_size)
        + "_"
        + str(args.maze_size)
        + "_"
        + args.label
        + "_"
        + args.mechanism
        + ".npz"
    )

    if args.output_path == "":
        # Note: path is relative to root directory, use `python -m envs.generate_dataset <arguments>`
        file_path = "../../data/"
        file_path += file_name

    else:
        if args.output_path.endswith(".npz") or args.output_path.endswith(".pth"):
            file_path = args.output_path
        else:
            file_path = os.path.join(args.output_path, file_name)

    mechanism = get_mechanism(args.mechanism)
    generate_data(
        file_path,
        args.train_size,
        args.valid_size,
        args.test_size,
        mechanism,
        args.maze_size,
        args.workspace_size,
        args.min_decimation,
        args.max_decimation,
        args.label,
        start_pos=(args.start_pos_y, args.start_pos_x),
        env_name=args.env,
        cont_act=args.cont_action,
        has_obstacle=args.has_obstacle,
        obsv_mode=args.obsv_mode,
    )


if __name__ == "__main__":
    main()
