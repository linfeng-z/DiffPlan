import sys

sys.path.append("/home/chris/Desktop/corl")
sys.path.append("/work/riverlab/hongyu/corl")

import numpy as np
import torch
import torch_geometric as pyg
from diffplan.envs.habitat.pose_extractor import remove_edges_on_walls
from diffplan.envs.habitat.data_extractor import ImageExtractor
import torch_cluster
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import Pool


def extract_scene(test_scene):
    try:
        extractor = ImageExtractor(
            test_scene,
            img_size=(128, 128),
            output=["rgba", "depth", "semantic"],
            pose_extractor_name="custom_panorama_extractor",
            meters_per_pixel=0.1,
            shuffle=False,
        )
    except:
        return None

    # collect randomly sampled points
    view = extractor.tdv_fp_ref_triples[0][0].topdown_view
    fp = extractor.tdv_fp_ref_triples[0][1]
    pos = set()
    for pose in extractor.pose_extractor.gridpoints:
        pos.add(pose)

    # Consturct networkx graph
    x = torch.tensor(list(pos), dtype=torch.float32)

    # extract the goal node (refrigerator)
    ids = []
    for _id, _item in extractor.instance_id_to_name.items():
        if _item == "refrigerator":
            ids.append(_id)

    # distance of the object in each node's observation
    dist = {}
    for idx, sample in enumerate(extractor):
        # extract four orientations
        pose = extractor.pose_extractor.gridpoints[idx // 4]

        # extract object in depth map (if seen)
        mask = np.zeros(sample["depth"].shape)
        if pose not in dist:
            dist[pose] = np.inf
        for _id in ids:
            mask = np.logical_or(mask, sample["semantic"] == _id)
        if len(np.nonzero(mask)[0]) > 30:  # threshold to filter noisy image
            new_dist = min(dist[pose], sample["depth"][mask].mean())
            if new_dist != 0.0:
                dist[pose] = new_dist

    pos_to_idx = {}  # map position to the graph node index
    dist_ordered = []  # dist in the order of node
    for idx, _pos in enumerate(pos):
        dist_ordered.append(dist[_pos])
        pos_to_idx[_pos] = idx

    # construct the graph based on sampled nodes
    radius_edge_index = torch_cluster.radius_graph(x, r=30)
    knn_edge_index = torch_cluster.knn_graph(x, k=15)
    radius_data = Data(x=x, edge_index=radius_edge_index)
    knn_data = Data(x=x, edge_index=knn_edge_index)
    G = to_networkx(radius_data, node_attrs=["x"], to_undirected=True)
    T = to_networkx(knn_data, node_attrs=["x"], to_undirected=True)
    # assign the distance as the weight
    for edge in G.edges:
        weight = np.linalg.norm(
            np.array(G.nodes[edge[0]]["x"]) - np.array(G.nodes[edge[1]]["x"])
        )
        G.edges[edge]["weight"] = weight
    for edge in T.edges:
        weight = np.linalg.norm(
            np.array(T.nodes[edge[0]]["x"]) - np.array(T.nodes[edge[1]]["x"])
        )
        T.edges[edge]["weight"] = weight

    G = remove_edges_on_walls(G, view)
    T = remove_edges_on_walls(T, view)
    data_nx = nx.minimum_spanning_tree(G, weight="weight")
    data_nx = nx.compose(data_nx, T)

    # label each node
    dijkstra_dist = nx.single_source_dijkstra_path_length(
        data_nx, np.argmin(dist_ordered), weight="weight"
    )
    length = np.array([np.inf] * len(data_nx))
    for _idx, _dist in dijkstra_dist.items():
        length[_idx] = _dist

    # construct the optimal policy
    for node in data_nx.nodes:
        best_neighbor = (-1, np.inf)
        for nei in data_nx.neighbors(node):
            if length[nei] < best_neighbor[1]:
                best_neighbor = (nei, length[nei])
        if best_neighbor[0] == -1:
            relative_trans = np.array((0, 0))
        else:
            relative_trans = np.array(data_nx.nodes[best_neighbor[0]]["x"]) - np.array(
                data_nx.nodes[node]["x"]
            )
        data_nx.nodes[node]["labels"] = relative_trans

    # add image observations
    for idx, sample in enumerate(extractor):
        pose = extractor.pose_extractor.gridpoints[idx // 4]
        node_idx = pos_to_idx[pose]
        data_nx.nodes[node_idx][f"img_{idx%4}"] = sample["rgba"]
        # when use images
        # follow the order 0->1->3->2
    extractor.close()

    # choose the largest subgraph due to isolated nodes exist
    data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len)).copy()

    graph_pyg = pyg.utils.from_networkx(data_nx)
    graph_pyg.opt_v = torch.tensor(length)
    graph_pyg.edge_index = pyg.utils.to_undirected(graph_pyg.edge_index)
    graph_pyg.edge_index = pyg.utils.add_self_loops(graph_pyg.edge_index)[0]
    return graph_pyg


def save_one_sample(i=0, _map="None"):
    test_scene = f"/work/riverlab/hongyu/dataset/habitat/gibson/{_map}.glb"
    while True:
        graph = extract_scene(test_scene)
        if graph is None:
            continue
        torch.save(graph, f"/work/riverlab/hongyu/corl/data/habitat/{_map}_{i}.pth")
        break


if __name__ == "__main__":
    map_name = [
        "Airport",
        "Ancor",
        "Andover",
        "Arkansaw",
        "Athens",
        "Bautista",
        "Bonesteel",
        "Chilhowie",
        "Clairton",
        "Emmaus",
        "Frankfort",
        "Goffs",
        "Goodfield",
        "Gravelly",
        "Highspire",
        "Hortense",
        "Irvine",
        "Kobuk",
        "Maida",
        "Neibert",
        "Newcomb",
        "Oyens",
        "Parole",
        "Pittsburg",
        "Scioto",
        "Soldier",
        "Springerville",
        "Sugarville",
        "Sussex",
        "Touhy",
        "Victorville",
    ]
    # for f in os.listdir("/work/riverlab/hongyu/dataset/habitat/3DSceneGraph_medium/automated_graph"):
    #     map_name.append(f.replace("3DSceneGraph_","").replace(".npz",""))

    for _map in tqdm(map_name):
        pool = Pool(12)
        pool.map(partial(save_one_sample, _map=_map), list(range(100)))
        # for i in range(100):
        #     save_one_sample(_map, i)
