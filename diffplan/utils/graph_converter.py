import torch_geometric
import numpy as np

# matplotlib.use("TkAgg")
import torch
from diffplan.envs.maze_utils import img_uint8_to_tensor


def discrete_to_cont_act(label):
    # convert one-hot vector to R^2 relative translation
    idices = torch.argmax(label, axis=1)
    action_order = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    delta = []
    for idx in idices:
        delta.append(action_order[idx])
    delta = np.array(delta, dtype=np.float32)
    return delta


def assign_edge_value(_graph):
    # assign value to each edge using the given node information
    # graph.x: (pos_x, pos_y, goal, obstacle)
    for edge in _graph.edges:
        pos_i = _graph.x[edge[0], :2]
        pos_j = _graph.x[edge[1], :2]
        obs_i = _graph.x[edge[0], 3]
        if obs_i == 1:
            _graph.edges[edge]["cost"] = 999999
        else:
            _graph.edges[edge]["cost"] = np.linalg.norm(pos_i - pos_j)


def get_partial_graph_obsv_mask(graph, rand_node_pos):
    # graph: networkx
    obsv_mask = list(graph.neighbors(rand_node_pos))
    for i in range(len(obsv_mask)):
        obsv_mask += list(graph.neighbors(obsv_mask[i]))
    obsv_mask = [*set(obsv_mask)]  # remove duplicates
    return obsv_mask


def convert_maze(
    maze,
    goal_map,
    opt_policy,
    mechanism,
    opt_value,
    cont_act=False,
    has_obstacle=False,
    pos2pano=None,
    obsv_mode="full",
    maze_onehot=None,
    goal_onehot=None,
):
    """
    Take in a maze (and the goal) and return a graph
    """
    nodes = []
    nodes_x_y = []  # store the x-y coordinate of each node
    nodes_hash = {}
    labels = []
    vis_labels = {}  # labels for visualization
    edges = []
    action2str = ["←", "↓", "→", "↑"]  # for visualization THIS IS THE INVERSE OF ACTUAL
    goals = []  # 0 for non-goal, 1 for goal
    obstacles = []  # 0 for non-obstacle 1 for obstacle
    goals_partial = []  # store partial observation
    obstacles_partial = []  # 0 store partial observation
    node_colors = []  # for visualization
    opt_v = []  # for the optimal values
    pano_obs = []  # for pano observation from Miniworld
    counter = 0

    height, width = maze.shape[0], maze.shape[1]

    for orient in range(mechanism.num_orient):
        for y in range(height):
            for x in range(width):
                # skip obstalce node if has_obstacle == True
                if (not has_obstacle) and (maze[y][x] == 0):
                    # obstacle
                    continue

                if has_obstacle:
                    neighbors = mechanism.neighbors_with_obstacle_func(
                        maze, orient, y, x
                    )
                else:
                    neighbors = mechanism.neighbors_func(maze, orient, y, x)

                if len(neighbors) == 0:
                    return None

                # record the hash of the node
                if (y, x) not in nodes_hash:
                    nodes.append(counter)  # store new node (id)
                    nodes_x_y.append((y, x))  # store x-y coordinate of new node
                    nodes_hash[y, x] = counter  # hash new node (id)
                    counter += 1
                    # store visualization label
                    vis_labels[nodes_hash[y, x]] = action2str[
                        np.argmax(opt_policy[:, orient, y, x])
                    ]
                    # store goal
                    if goal_map[orient][y][x] == 1.0:
                        # is goal
                        node_colors.append("red")
                        goals.append(1)
                    else:
                        node_colors.append("blue")
                        goals.append(0)
                    if obsv_mode == "partial":
                        if goal_onehot[0][y][x] == 1:
                            # is goal
                            node_colors.append("red")
                        elif goal_onehot[1][y][x] == 1:
                            # is not goal
                            node_colors.append("blue")
                        else:
                            # unknown
                            node_colors.append("gray")
                        goals_partial.append(goal_onehot[:, y, x])

                    # store obstacle: flipping original(0: no-obstacle, 1: obstacle)
                    obstacles.append(1 - maze[y][x])
                    if obsv_mode == "partial":
                        obstacles_partial.append(maze_onehot[:, y, x])

                    # store optimal value
                    opt_v.append(opt_value[orient][y][x])

                    # store pano RGB observation
                    if pos2pano is not None:
                        pano_obs.append(img_uint8_to_tensor(pos2pano[y, x]))

                for neighbor in neighbors:
                    py, px = neighbor[1], neighbor[2]

                    if py == y and px == x:
                        # the neighbor is node itself
                        continue
                    if (py, px) not in nodes_hash:
                        nodes.append(counter)  # store new node (id)
                        nodes_x_y.append((py, px))  # store x-y coordinate of new node
                        nodes_hash[py, px] = counter  # hash new node (id)
                        counter += 1
                        # store label
                        vis_labels[nodes_hash[py, px]] = action2str[
                            np.argmax(opt_policy[:, orient, py, px])
                        ]

                        # store goal
                        if goal_map[orient][py][px] == 1.0:
                            # is goal
                            node_colors.append("red")
                            goals.append(1)
                        else:
                            node_colors.append("blue")
                            goals.append(0)
                        if obsv_mode == "partial":
                            if goal_onehot[0][py][px] == 1:
                                # is goal
                                node_colors.append("red")
                            elif goal_onehot[1][py][px] == 1:
                                # is not goal
                                node_colors.append("blue")
                            else:
                                # unknown
                                node_colors.append("gray")
                            goals_partial.append(goal_onehot[:, py, px])

                        # store obstacle: flipping original(0: no-obstacle, 1: obstacle)
                        obstacles.append(1 - maze[py][px])
                        if obsv_mode == "partial":
                            obstacles_partial.append(maze_onehot[:, py, px])

                        # store optimal value
                        opt_v.append(opt_value[orient][py][px])
                        # store pano RGB observation
                        if pos2pano is not None:
                            pano_obs.append(img_uint8_to_tensor(pos2pano[py, px]))

                    edges.append((nodes_hash[y, x], nodes_hash[py, px]))

                # store NEWS label
                labels.append(opt_policy[:, orient, y, x])

    # convert to continous action if needed
    if cont_act:
        labels = discrete_to_cont_act(label=torch.tensor(labels))

    nodes_x_y = np.array(nodes_x_y)
    goals = np.expand_dims(np.array(goals), 1)
    goals_partial = np.array(goals_partial)
    obstacles = np.expand_dims(np.array(obstacles), 1)
    obstacles_partial = np.array(obstacles_partial)
    opt_v = torch.unsqueeze(torch.tensor(opt_v), 1)
    if pos2pano is not None:
        pano_obs = torch.stack(pano_obs)
    edges = torch.tensor(edges)

    feature = [nodes_x_y, goals]
    if has_obstacle:
        feature.append(obstacles)
    feature = torch.tensor(np.concatenate(feature, axis=1), dtype=torch.float)

    partial_feature = []
    if obsv_mode == "partial":
        partial_feature.append(goals_partial)
        if has_obstacle:
            partial_feature.append(obstacles_partial)
        partial_feature = torch.tensor(
            np.concatenate(partial_feature, axis=1), dtype=torch.float
        )

    data = torch_geometric.data.Data(
        x=feature,
        partial_x=partial_feature,
        labels=torch.tensor(labels),
        opt_v=opt_v,
        pano_obs=pano_obs,
        edge_index=edges.t().contiguous(),
    )
    data.edge_index = torch_geometric.utils.to_undirected(data.edge_index)
    data.edge_index = torch_geometric.utils.add_self_loops(data.edge_index)[0]

    # remove comment for visualization
    # g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    # positions = dict(zip(g.nodes, nodes_x_y))
    # v = opt_v.numpy()
    # v[:, 0][v[:, 0] == -225.0] = 225.0
    # v[:, 0][v[:, 0] == 225.0] = np.min(v[:, 0])
    # colors = plt.cm.Greys((v[:, 0] - np.min(v[:, 0])) / (np.max(v[:, 0]) - np.min(v[:, 0])))
    # node_colors = [colors[i] if goals[i, 0] == 0 else "red" for i in range(len(goals))]
    # nx.draw(g, positions, node_color=node_colors)
    # nx.draw_networkx_labels(g, positions, labels=vis_labels)
    # plt.show()

    return data
