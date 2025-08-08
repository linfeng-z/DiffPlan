# -*- coding: utf-8 -*-
from __future__ import print_function

import sys

import numpy as np
import torch
import torch.utils.data as data


class MazeDataset(data.Dataset):
    def __init__(self, filename, dataset_type):
        """
        Args:
          filename (str): Dataset filename (must be .npz format).
          dataset_type (str): One of "train", "valid", or "test".
        """
        assert filename.endswith("npz")  # Must be .npz format
        self.filename = filename
        self.dataset_type = dataset_type  # train, valid, test

        self._process(filename)

        self.num_actions = self.opt_policies.shape[1]
        self.num_orient = self.opt_policies.shape[2]

    def _process(self, filename):
        """
        Data format: list, [train data, test data]
        """

        with np.load(filename) as f:
            dataset2idx = {"train": 0, "valid": 3, "test": 6}
            idx = dataset2idx[self.dataset_type]
            mazes = f["arr_" + str(idx)]
            goal_maps = f["arr_" + str(idx + 1)]
            opt_policies = f["arr_" + str(idx + 2)]

            # Set proper datatypes
            self.mazes = mazes.astype(np.float32)
            self.goal_maps = goal_maps.astype(np.float32)
            self.opt_policies = opt_policies.astype(np.float32)

            # > Process for obs
            if "Visual3DNav" in self.filename or "WorkSpaceEnv" in self.filename:
                print("> Note: loading 3D nav data")
                assert len(f.keys()) == 12  # it has additional 3 arrays for obs

                dataset2idx_pano = {"train": 9, "valid": 10, "test": 11}
                pano_idx = dataset2idx_pano[self.dataset_type]
                pano_obs = f["arr_" + str(pano_idx)]
                # > keep numpy lazily load array; convert to float tensor in getitem
                self.pano_obs = pano_obs

        # Print number of samples
        if self.dataset_type == "train":
            print("Number of Train Samples: {0}".format(mazes.shape[0]))
        elif self.dataset_type == "valid":
            print("Number of Validation Samples: {0}".format(mazes.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(mazes.shape[0]))
        print("\tSize: {}x{}".format(mazes.shape[1], mazes.shape[2]))

    def __getitem__(self, index):
        maze = self.mazes[index]
        goal_map = self.goal_maps[index]
        opt_policy = self.opt_policies[index]

        maze = torch.from_numpy(maze)
        goal_map = torch.from_numpy(goal_map)
        opt_policy = torch.from_numpy(opt_policy)

        if "Visual3DNav" in self.filename or "WorkSpaceEnv" in self.filename:
            # > only convert uint8 (smaller) to float (larger, ready to train) when loading
            pano_obs = img_uint8_to_tensor(self.pano_obs[index])
            return dict(
                maze=maze, goal_map=goal_map, opt_policy=opt_policy, pano_obs=pano_obs
            )
        else:
            return dict(maze=maze, goal_map=goal_map, opt_policy=opt_policy)

    def __len__(self):
        return self.mazes.shape[0]


def img_uint8_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    convert uint8 image array to torch tensor, with normalization
    """
    img = torch.from_numpy(img)

    if isinstance(img, torch.ByteTensor):
        img = img.float().div(255)
    if isinstance(img, torch.DoubleTensor):
        # print('>>> Warning! Data is in Double 64, convert to Float 32')
        img = img.float().div(255)

    return img


def extract_policy(maze, mechanism, value, is_full_policy=False):
    """Extracts the policy from the given values."""
    policy = np.zeros(
        (mechanism.num_actions, value.shape[0], value.shape[1], value.shape[2])
    )
    for p_orient in range(value.shape[0]):
        for p_y in range(value.shape[1]):
            for p_x in range(value.shape[2]):
                # Find the neighbor w/ max value (assuming deterministic
                # transitions)
                max_val = -sys.maxsize
                max_acts = [0]
                neighbors = mechanism.neighbors_func(maze, p_orient, p_y, p_x)
                for i in range(len(neighbors)):
                    n = neighbors[i]
                    nval = value[n[0]][n[1]][n[2]]
                    if nval > max_val:
                        max_val = nval
                        max_acts = [i]
                    elif nval == max_val:
                        max_acts.append(i)

                # Choose max actions if several w/ same value
                if is_full_policy:
                    policy[max_acts, p_orient, p_y, p_x] = 1.0
                else:
                    max_act = max_acts[np.random.randint(len(max_acts))]
                    policy[max_act][p_orient][p_y][p_x] = 1.0
    return policy


def extract_goal(goal_map):
    """Returns the goal location."""
    for o in range(goal_map.shape[0]):
        for y in range(goal_map.shape[1]):
            for x in range(goal_map.shape[2]):
                if goal_map[o][y][x] == 1.0:
                    return (o, y, x)
    assert False
