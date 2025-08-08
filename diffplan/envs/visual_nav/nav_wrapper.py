import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_cluster
import torch_geometric
from miniworld.miniworld import MiniWorldEnv

from diffplan.envs.maze_utils import img_uint8_to_tensor
from diffplan.utils.vis_pano import plot_pano

matplotlib.use("TkAgg")


def assign_edge_value(_graph):
    # assign value to each edge using the given node information
    # graph.x: (pos_x, pos_y, goal, obstacle)
    for edge in _graph.edges:
        pos_i = _graph.x[edge[0], :2]
        pos_j = _graph.x[edge[1], :2]
        obs_i = _graph.obs[edge[0], 0]
        if obs_i == 1:
            _graph.edges[edge]["cost"] = 999999
        else:
            _graph.edges[edge]["cost"] = np.linalg.norm(pos_i - pos_j)


class MazeNavWorld(MiniWorldEnv):
    def __init__(self, maze_map=None, obs_width=80, obs_height=60, domain_rand=False):
        if maze_map is not None:
            self.num_rows, self.num_cols = maze_map.shape[0], maze_map.shape[1]
            self.maze_map = maze_map
            self.valid_pos = self.get_valid_pos(
                self.maze_map, valid_sym=1, return_np=False
            )

        self.room_size = 1
        self.gap_size = 0.01  # room gap size / wall thickness

        self.num_views = 4
        self.num_rgb = 3
        self.obs_dim = (obs_height, obs_width, self.num_rgb)

        self.obs_width = obs_width
        self.obs_height = obs_height

        super().__init__(
            obs_width=obs_width, obs_height=obs_height, domain_rand=domain_rand
        )

    def set_map(self, maze_map):
        self.num_rows, self.num_cols = maze_map.shape[0], maze_map.shape[1]
        self.maze_map = maze_map
        self.valid_pos = self.get_valid_pos(maze_map, valid_sym=1, return_np=False)

    def reset(self):
        # Note: `_gen_world()` is invoked
        return super().reset()

    def _gen_world(self):
        _rooms = {}

        # > Note: row is for x-axis, col is for z-axis, full position coordinate is like (x, -, z)
        for _row in range(self.num_rows):
            for _col in range(self.num_cols):
                # check if the current cell is a corridor cell
                if (_row, _col) not in self.valid_pos:
                    continue

                # compute the boundary
                min_x = _col * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = _row * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                # add the room
                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                )

                _rooms[_row, _col] = room

        visited = set()

        # > connect the neighbors rooms given map
        for _row, _col in self.valid_pos:
            room = _rooms[_row, _col]
            visited.add(room)

            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, left, right

            # loop valid neighbors
            for d_row, d_col in neighbors:
                n_row = _row + d_row
                n_col = _col + d_col

                if n_row < 0 or n_row >= self.num_rows:
                    continue
                if n_col < 0 or n_col >= self.num_cols:
                    continue

                # > don't render rooms for invalid positions
                if (n_row, n_col) not in self.valid_pos:
                    continue

                neighbor = _rooms[n_row, n_col]

                if neighbor in visited:
                    continue

                if d_col == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif d_row == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

        # > no need to place goal
        # self.box = self.place_entity(Box(color='red'))

        self.place_agent()

    def render_pano(self, pos, vis_top=False):
        """
        Helper for rendering egocentric panoramic images
        """

        pano_obs = np.empty(
            shape=(self.num_views, self.obs_height, self.obs_width, self.num_rgb),
            dtype=np.uint8,
        )

        # > set position; note: swap col and row outside!
        self.agent.pos = np.array([pos[0] + 0.5, 0.0, pos[1] + 0.5])

        # > (1) agent facing right, so (3/2)pi or (-1/2)pi should be north
        # > (2) should have counter-clockwise order
        directions = np.array([1.0 / 2, 0.0 / 2, 3.0 / 2, 2.0 / 2]) * np.pi
        # directions = np.arange(0, 2 * np.pi, np.pi / 2)
        # directions = np.roll(directions, -1)

        for _i, _dir in enumerate(directions):
            # > set direction/orientation
            self.agent.dir = _dir

            if vis_top:
                print(
                    f"(Debug) position = {self.agent.pos}, orientation = {self.agent.dir}"
                )

            # > retrieve observation
            pano_obs[_i] = self.render_obs()

        # > retrieve top-down observation for visualization
        if vis_top:
            top_obs = self.render_top_view()
            return pano_obs, top_obs
        else:
            return pano_obs

    def get_all_pano(self, vis_top=False, plot_top=False):
        # > panoramic (4-direction) RGB images (e.g., 32 x 32 x 3) for every location (e.g., 5 x 5)
        # > e.g. 5 x 5 x (4 x 32 x 32 x 3)
        pos2pano = np.zeros(
            (self.num_rows, self.num_cols, self.num_views) + self.obs_dim,
            dtype=np.uint8,
        )
        pos2top = {}

        # > fill actual images
        for pos in self.valid_pos:
            # > note: swap col and row!
            pos_agent = (pos[1], pos[0])

            if vis_top:
                pano_obs, top_obs = self.render_pano(pos=pos_agent, vis_top=True)
                pos2top[pos_agent] = top_obs

                if plot_top:
                    plot_pano(pano_obs, top_obs, title=f"Position = ${pos}$")

            else:
                pano_obs = self.render_pano(pos=pos_agent, vis_top=False)

            pos2pano[pos[1]][pos[0]] = pano_obs

        return (pos2pano, pos2top) if vis_top else pos2pano

    @staticmethod
    def get_valid_pos(maze_map, valid_sym=0, return_np=False):
        """return valid (0, empty) grid cells for goal/start position"""
        res = np.argwhere(maze_map == valid_sym)
        return res if return_np else list(map(tuple, res.tolist()))


class GraphNavWorld(MiniWorldEnv):
    def __init__(self, maze_map=None, obs_width=80, obs_height=60, domain_rand=False):
        if maze_map is not None:
            self.num_rows, self.num_cols = maze_map.shape[0], maze_map.shape[1]
            self.maze_map = maze_map
            self.valid_pos = self.get_valid_pos(
                self.maze_map, valid_sym=1, return_np=False
            )

        self.room_size = 1
        self.gap_size = 0.01  # room gap size / wall thickness

        self.num_views = 4
        self.num_rgb = 3
        self.obs_dim = (obs_height, obs_width, self.num_rgb)

        self.obs_width = obs_width
        self.obs_height = obs_height

        super().__init__(
            obs_width=obs_width, obs_height=obs_height, domain_rand=domain_rand
        )

    def graph_on_maze(self, maze_map, goal_map):
        H, W = maze_map.shape
        # randomly generate nodes
        nodes = np.random.rand(127, 2) * H  # (x,y) => with in (0,15),(0,15)
        goal_node = np.argwhere(goal_map == 1)[:, 1:]
        nodes = np.concatenate([nodes, goal_node])  # add goal node in random nodes
        nodes = torch.tensor(nodes)
        # radius graph
        radius_edge_index = torch_cluster.knn_graph(nodes, k=10)
        radius_data = torch_geometric.data.Data(x=nodes, edge_index=radius_edge_index)
        graph = torch_geometric.utils.to_networkx(radius_data)
        graph = graph.to_undirected()

        obs_idx = np.argwhere(maze_map == 0)
        distances = torch.cdist(torch.tensor(obs_idx).float(), nodes.float())
        closest_nodes = torch.argwhere(torch.sum(distances <= 0.5, dim=0) > 0)

        obs = np.zeros((128, 1))
        obs[closest_nodes] = 1

        maze_idx = torch.floor(nodes).int()
        goal = np.expand_dims(goal_map[0, maze_idx[:, 0], maze_idx[:, 1]], 1)

        # form the node feature (2D pos, goal, obstacle)
        graph.x = np.concatenate([nodes[:, :2].numpy(), goal], axis=1)
        graph.obs = obs  # only for get_valid_pos

        return graph

    def set_map(self, maze_map, goal_map):
        self.num_rows, self.num_cols = maze_map.shape[0], maze_map.shape[1]
        self.maze_map = maze_map
        self.graph_map = self.graph_on_maze(maze_map, goal_map)
        self.valid_pos = self.get_valid_pos(maze_map, valid_sym=1, return_np=False)
        self.valid_graph_pos = self.get_valid_pos(
            self.graph_map.obs, valid_sym=0, return_np=False
        )

    def reset(self):
        # Note: `_gen_world()` is invoked
        return super().reset()

    def _gen_world(self):
        _rooms = {}

        # > Note: row is for x-axis, col is for z-axis, full position coordinate is like (x, -, z)
        for _row in range(self.num_rows):
            for _col in range(self.num_cols):
                # check if the current cell is a corridor cell
                if (_row, _col) not in self.valid_pos:
                    continue

                # compute the boundary
                min_x = _col * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = _row * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                # add the room
                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                )

                _rooms[_row, _col] = room

        visited = set()

        # > connect the neighbors rooms given map
        for _row, _col in self.valid_pos:
            room = _rooms[_row, _col]
            visited.add(room)

            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, left, right

            # loop valid neighbors
            for d_row, d_col in neighbors:
                n_row = _row + d_row
                n_col = _col + d_col

                if n_row < 0 or n_row >= self.num_rows:
                    continue
                if n_col < 0 or n_col >= self.num_cols:
                    continue

                # > don't render rooms for invalid positions
                if (n_row, n_col) not in self.valid_pos:
                    continue

                neighbor = _rooms[n_row, n_col]

                if neighbor in visited:
                    continue

                if d_col == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif d_row == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

        # > no need to place goal
        # self.box = self.place_entity(Box(color='red'))

        self.place_agent()

    def render_pano(self, pos, vis_top=False):
        """
        Helper for rendering egocentric panoramic images
        """

        pano_obs = np.empty(
            shape=(self.num_views, self.obs_height, self.obs_width, self.num_rgb),
            dtype=np.uint8,
        )

        # > set position; note: swap col and row outside!
        self.agent.pos = np.array([pos[0], 0.0, pos[1]])

        # > (1) agent facing right, so (3/2)pi or (-1/2)pi should be north
        # > (2) should have counter-clockwise order
        directions = np.array([1.0 / 2, 0.0 / 2, 3.0 / 2, 2.0 / 2]) * np.pi
        # directions = np.arange(0, 2 * np.pi, np.pi / 2)
        # directions = np.roll(directions, -1)

        for _i, _dir in enumerate(directions):
            # > set direction/orientation
            self.agent.dir = _dir

            if vis_top:
                print(
                    f"(Debug) position = {self.agent.pos}, orientation = {self.agent.dir}"
                )

            # > retrieve observation
            pano_obs[_i] = self.render_obs()

        # > retrieve top-down observation for visualization
        if vis_top:
            top_obs = self.render_top_view()
            return pano_obs, top_obs
        else:
            return pano_obs

    def plot_graph_pano(
        self,
        node,
        pano_obs,
        title=None,
        size=(8, 8),
        reversed_order=True,
        node_sizes=[100, 10, 10],
    ):
        fig, axs = plt.subplots(3, 3, constrained_layout=True, figsize=size)

        # > disable unused subplots
        for ax_i in [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]:
            axs[ax_i].axis("off")

        front = axs[0][1]
        back = axs[2][1]
        left = axs[1][0]
        right = axs[1][2]

        front.set_title("Front")
        front.axis("off")
        left.set_title("Left")
        left.axis("off")
        back.set_title("Back")
        back.axis("off")
        right.set_title("Right")
        right.axis("off")

        # > counter-clockwise order
        if reversed_order:
            front.imshow(pano_obs[0])
            left.imshow(pano_obs[-1])
            back.imshow(pano_obs[-2])
            right.imshow(pano_obs[-3])
        else:
            front.imshow(pano_obs[0])
            left.imshow(pano_obs[1])
            back.imshow(pano_obs[2])
            right.imshow(pano_obs[3])

        # > render top-down obs
        top = axs[1, 1]
        top.axis("off")
        top.set_title("Top")
        # add spatial feature for drawing
        features = self.graph_map.x
        positions = dict(zip(self.graph_map.nodes, features[:, :2]))
        # plt.title(title)
        nx.draw(self.graph_map, positions, ax=top, node_size=node_sizes[0])
        # plot agent location
        curr_pos = self.graph_map.x[node, [0, 1]]
        top.plot(curr_pos[0], curr_pos[1], "r*", markersize=node_sizes[1])
        # plot obstacles
        obs = np.argwhere(self.graph_map.obs == 1)
        obs_pos = self.graph_map.x[obs][:, 0, :2]
        top.plot(obs_pos[:, 0], obs_pos[:, 1], "ko", markersize=node_sizes[2])

        if title is not None:
            fig.suptitle(title)

        return fig

    def get_all_pano(self, vis_top=False, plot_top=False):
        # > panoramic (4-direction) RGB images (e.g., 32 x 32 x 3) for every location (e.g., 5 x 5)
        # > e.g. 5 x 5 x (4 x 32 x 32 x 3)
        pos2pano = np.zeros((128, self.num_views) + self.obs_dim, dtype=np.uint8)
        pos2top = {}

        # > fill actual images
        for node in self.valid_graph_pos:
            node = node[0]
            # > note: swap col and row!
            pos = self.graph_map.x[node, [0, 1]]
            pos_agent = (pos[1], pos[0])

            if vis_top:
                pano_obs, top_obs = self.render_pano(pos=pos_agent, vis_top=True)
                pos2top[pos_agent] = top_obs

                if plot_top:
                    plot_pano(pano_obs, top_obs, title=f"Position = ${pos}$")
                    fig = self.plot_graph_pano(
                        node, pano_obs, title=f"Position = ${pos}$"
                    )

            else:
                pano_obs = self.render_pano(pos=pos_agent, vis_top=False)

            pos2pano[node] = pano_obs

        assign_edge_value(self.graph_map)

        goal_idx = np.argwhere(self.graph_map.x[:, 2] == 1)[0][0]
        obs_idx = np.argwhere(self.graph_map.obs[:, 0] == 1)

        length = nx.single_source_dijkstra_path_length(
            self.graph_map, goal_idx, weight="cost"
        )

        for _idx in length:
            if length[_idx] == np.inf:
                # we have node that couldn't reach the goal
                return None

        opt_v = np.zeros((len(self.graph_map.nodes), 1))
        for _idx, _key in enumerate(self.graph_map.nodes):
            opt_v[_idx] = -length[_key]
        opt_v[obs_idx] = -np.inf

        # construct the optimal policy
        for node in self.graph_map.nodes:
            best_neighbor = (-1, np.inf)
            for nei in self.graph_map.neighbors(node):
                if length[nei] < best_neighbor[1]:
                    best_neighbor = (nei, length[nei])
            relative_trans = (
                self.graph_map.x[best_neighbor[0], :2] - self.graph_map.x[node, :2]
            )
            self.graph_map.nodes[node]["labels"] = relative_trans

        graph_pyg = torch_geometric.utils.from_networkx(self.graph_map)
        graph_pyg.x = torch.tensor(self.graph_map.x, dtype=torch.float)
        graph_pyg.opt_v = torch.tensor(opt_v)
        graph_pyg.edge_index = torch_geometric.utils.add_self_loops(
            graph_pyg.edge_index
        )[0]
        graph_pyg.labels = graph_pyg.labels.float()
        graph_pyg.pano_obs = img_uint8_to_tensor(pos2pano)

        return (pos2pano, pos2top) if vis_top else self.graph_map

    @staticmethod
    def get_valid_pos(maze_map, valid_sym=0, return_np=False):
        """return valid (0, empty) grid cells for goal/start position"""
        res = np.argwhere(maze_map == valid_sym)
        return res if return_np else list(map(tuple, res.tolist()))
