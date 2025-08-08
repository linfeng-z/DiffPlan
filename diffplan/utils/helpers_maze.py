import torch
import numpy as np


def convert_actions(logits):
    assert len(logits.shape) == 4

    # logits: actions x orient x width x height
    # actions_one_hot = logits.max(dim=0).values
    actions_one_hot = None

    if isinstance(logits, torch.Tensor):
        actions_int = logits.max(dim=0).indices
    elif isinstance(logits, np.ndarray):
        actions_int = logits.argmax(axis=0)
    else:
        raise ValueError

    return actions_one_hot, actions_int


def convert_goal(goal_map, to_tuple=True):
    if len(goal_map.shape) == 2:
        goal_map = goal_map.unsqueeze(
            0
        )  # one dim for orientation, might lose in concat
    else:
        assert len(goal_map.shape) == 3

    goal_pos = goal_map.nonzero()

    if to_tuple and not isinstance(goal_pos, tuple):
        goal_pos = goal_pos.squeeze()
        goal_pos = tuple(goal_pos.numpy().tolist())

    return goal_pos


def get_partial_obsv_mask(maze, rand_node_pos):
    maze_size, _ = maze.shape
    obsv_mask = np.zeros(maze.shape)
    for i in range(-2, 2):
        for j in range(-2, 2):
            _new_i = rand_node_pos[0] + i
            _new_j = rand_node_pos[1] + j
            if _new_i < 0 or _new_i >= maze_size or _new_j < 0 or _new_j >= maze_size:
                continue
            obsv_mask[_new_i, _new_j] = 1
    return obsv_mask
