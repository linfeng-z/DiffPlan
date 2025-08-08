from typing import List, Tuple

import habitat_sim.registry as registry
import numpy as np
from habitat_sim.utils.data import PoseExtractor
from habitat_sim.utils.data.pose_extractor import TopdownView
from numpy import float32, ndarray


@registry.register_pose_extractor(name="custom_panorama_extractor")
class CustomPanoramaExtractor(PoseExtractor):
    def __init__(
        self,
        topdown_views: List[Tuple[TopdownView, str, Tuple[float32, float32, float32]]],
        meters_per_pixel: float = 0.1,
    ) -> None:
        super().__init__(topdown_views, meters_per_pixel)

    def extract_poses(
        self, view: ndarray, fp: str
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        # Determine the physical spacing between each camera position
        height, width = view.shape
        dist = min(height, width) // 20  # We can modify this to be user-defined later

        # Create a grid of camera positions
        n_gridpoints_width, n_gridpoints_height = (
            width // dist - 1,
            height // dist - 1,
        )

        # Exclude camera positions at invalid positions
        num_random_points = 128
        gridpoints = []

        while len(gridpoints) < num_random_points:
            # Get the row and column of a random point on the topdown view
            row, col = np.random.randint(0, height), np.random.randint(0, width)

            # Convenient method in the PoseExtractor class to check if a point
            # is navigable
            if self._valid_point(row, col, view) and (row, col) not in gridpoints:
                gridpoints.append((row, col))

        # Find the closest point of the target class to each gridpoint
        poses = []
        for point in gridpoints:
            point_label_pairs = self._panorama_extraction(point, view, dist)
            poses.extend([(point, point_, fp) for point_, label in point_label_pairs])

        # Returns poses in the coordinate system of the topdown view
        self.gridpoints = gridpoints
        return poses

    def _panorama_extraction(
        self, point: Tuple[int, int], view: ndarray, dist: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        in_bounds_of_topdown_view = lambda row, col: 0 <= row < len(
            view
        ) and 0 <= col < len(view[0])
        point_label_pairs = []
        r, c = point
        neighbor_dist = dist // 2
        neighbors = [
            (r - neighbor_dist, c - neighbor_dist),
            (r - neighbor_dist, c + neighbor_dist),
            (r + neighbor_dist, c - neighbor_dist),
            (r + neighbor_dist, c + neighbor_dist),
        ]

        for n in neighbors:
            # Only add the neighbor point if it is navigable. This prevents camera poses that
            # are just really close-up photos of some object
            #             if in_bounds_of_topdown_view(*n) and self._valid_point(*n, view):
            point_label_pairs.append((n, 0.0))

        return point_label_pairs


def remove_edges_on_walls(data_nx, view):
    # remove edges crossing the wall
    for edge in data_nx.edges:
        if data_nx.nodes[edge[0]]["x"][0] < data_nx.nodes[edge[1]]["x"][0]:
            node_i = data_nx.nodes[edge[0]]["x"]
            node_j = data_nx.nodes[edge[1]]["x"]
        else:
            node_i = data_nx.nodes[edge[1]]["x"]
            node_j = data_nx.nodes[edge[0]]["x"]

        interp_x = np.arange(node_i[0], node_j[0]).astype(int)
        interp_y = np.interp(
            np.arange(node_i[0], node_j[0]),
            [node_i[0], node_j[0]],
            [node_i[1], node_j[1]],
        ).astype(int)

        for i in range(len(interp_x)):
            row = interp_x[i]
            col = interp_y[i]
            if view[row, col] != 1.0:
                # edge cross walls
                data_nx.remove_edge(edge[0], edge[1])
                break

    for edge in data_nx.edges:
        if data_nx.nodes[edge[0]]["x"][1] < data_nx.nodes[edge[1]]["x"][1]:
            node_i = data_nx.nodes[edge[0]]["x"]
            node_j = data_nx.nodes[edge[1]]["x"]
        else:
            node_i = data_nx.nodes[edge[1]]["x"]
            node_j = data_nx.nodes[edge[0]]["x"]

        interp_y = np.arange(node_i[1], node_j[1]).astype(int)
        interp_x = np.interp(
            np.arange(node_i[1], node_j[1]),
            [node_i[1], node_j[1]],
            [node_i[0], node_j[0]],
        ).astype(int)

        for i in range(len(interp_x)):
            row = interp_x[i]
            col = interp_y[i]
            if view[row, col] != 1.0:
                # edge cross walls
                data_nx.remove_edge(edge[0], edge[1])
                break

    return data_nx
