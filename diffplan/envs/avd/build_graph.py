import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import torch
import torch_geometric


def get_nodes(scene_path):
    # load annotations
    annotations_path = os.path.join(scene_path, "annotations.json")
    with open(annotations_path) as ann_file:
        annotations = json.load(ann_file)

    image_structs_path = os.path.join(scene_path, "image_structs.mat")
    image_structs = sio.loadmat(image_structs_path)
    # a scale constant
    scale = image_structs["scale"][0][0]

    # meta-data for every image such as camera pose
    image_structs = image_structs["image_structs"][0]

    # node_dict
    # we combine node with different orientations here
    # if a node is parent node, the value is False
    # if a node is child node (different orientation), the value will link to the parent node (canonical orientation)
    node_dict = {}
    parent_node_count = 0  # for debugging

    # find parent and child nodes
    for camera in image_structs:
        image_name = camera[0][0]

        if image_name not in node_dict:
            # start a parent node
            parent_node = image_name
            parent_node_count += 1
            node_dict[parent_node] = False

            # iterate through other orientations clockwise
            curr_node = parent_node
            while True:
                child_node = annotations[curr_node]["rotate_cw"]
                if child_node in node_dict and node_dict[child_node] == False:
                    # get back to the parent node
                    break
                node_dict[child_node] = parent_node
                curr_node = child_node

    print(
        f"Total nodes {len(node_dict)}, parent nodes {parent_node_count}, orientations {len(node_dict)/parent_node_count}"
    )

    # find the position of parent nodes
    parent_nodes = []  # store names
    parent_nodes_hash = {}  # map name to indices
    counter = 0
    x = []
    y = []

    for camera in image_structs:
        image_name = camera[0][0]

        if node_dict[image_name] != False:
            continue
        world_pos = camera[3]  # as shown in official code
        world_pos *= scale  # as shown in official code

        parent_nodes.append(image_name)
        parent_nodes_hash[image_name] = counter
        counter += 1
        x.append(world_pos[0])
        y.append(world_pos[2])

    # uncomment this for nodes visualization
    plt.plot(x, y, "ro")
    plt.show()

    return x, y, parent_nodes, parent_nodes_hash, node_dict, annotations


def get_node_index(node, node_dict, parent_nodes_hash):
    """
    node: the node we are interested
    node_dict: key is image file name, value is either False (parent node) or parent node name
    parent_nodes_hash: map parent node name to a index numer
    """
    if node_dict[node] == False:
        # is parent node
        return parent_nodes_hash[node]
    else:
        parent_node = node_dict[node]
        return parent_nodes_hash[parent_node]


def get_edges(x, y, parent_nodes, parent_nodes_hash, node_dict, annotations):
    edges_set = {}

    for node in parent_nodes:
        curr_node = node
        curr_node_idx = get_node_index(curr_node, node_dict, parent_nodes_hash)
        edges_set[curr_node_idx] = set()
        while True:
            curr_node_idx = get_node_index(curr_node, node_dict, parent_nodes_hash)
            node_annot = annotations[curr_node]
            for neighbor_direction in ["forward", "backward", "left", "right"]:
                neighbor_node = node_annot[neighbor_direction]
                if neighbor_node == "":
                    continue

                neighbor_node_idx = get_node_index(
                    neighbor_node, node_dict, parent_nodes_hash
                )
                edges_set[curr_node_idx].add(neighbor_node_idx)

            curr_node = node_annot["rotate_cw"]
            if curr_node in parent_nodes:
                break

    # convert set into list
    edges = []
    for node in parent_nodes:
        node_idx = get_node_index(node, node_dict, parent_nodes_hash)
        for neighbor_node in edges_set[node_idx]:
            edges.append([node_idx, neighbor_node])

    return edges


if __name__ == "__main__":
    AVD_PATH = "/home/chris/Downloads/ActiveVisionDataset/example_scene/"
    scene_name = "Home_003_1"

    scene_path = os.path.join(AVD_PATH, scene_name)

    x, y, parent_nodes, parent_nodes_hash, node_dict, annotations = get_nodes(
        scene_path
    )
    edges = get_edges(x, y, parent_nodes, parent_nodes_hash, node_dict, annotations)

    edge_index = torch.tensor(edges, dtype=torch.long)
    nodes = torch.arange(len(parent_nodes))

    data = torch_geometric.data.Data(x=nodes, edge_index=edge_index.t().contiguous())
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)

    # add spatial feature for drawing
    coordinates = np.column_stack((x, y))
    positions = dict(zip(g.nodes, coordinates))

    nx.draw(g, positions)
    plt.show()
