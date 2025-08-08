import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data.lightning import LightningDataset


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


class MazeDataset(Dataset):
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
            # regular grid world
            dataset2idx = {"train": 0, "valid": 4, "test": 8}
            idx = dataset2idx[self.dataset_type]
            mazes = f["arr_" + str(idx)]
            goal_maps = f["arr_" + str(idx + 1)]
            opt_policies = f["arr_" + str(idx + 2)]
            opt_values = f["arr_" + str(idx + 3)]

            # Set proper datatypes
            self.mazes = mazes.astype(np.float32)
            self.goal_maps = goal_maps.astype(np.float32)
            self.opt_policies = opt_policies.astype(np.float32)
            self.opt_values = opt_values.astype(np.float32)

            # > Process for obs
            if "Visual3DNav" in self.filename or "WorkSpaceEnv" in self.filename:
                print("> Note: loading 3D nav data")
                # it has additional 3 arrays for obs
                assert len(f.keys()) == 15
                dataset2idx_pano = {"train": 12, "valid": 13, "test": 14}
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

        opt_values = self.opt_values[index]
        opt_values = torch.from_numpy(opt_values)

        if "Visual3DNav" in self.filename or "WorkSpaceEnv" in self.filename:
            # > only convert uint8 (smaller) to float (larger, ready to train) when loading
            pano_obs = img_uint8_to_tensor(self.pano_obs[index])
            return dict(
                maze=maze,
                goal_map=goal_map,
                opt_policy=opt_policy,
                opt_values=opt_values,
                pano_obs=pano_obs,
            )
        else:
            return dict(
                maze=maze,
                goal_map=goal_map,
                opt_policy=opt_policy,
                opt_values=opt_values,
            )

    def __len__(self):
        return self.mazes.shape[0]


class GridDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            # train+val
            self.maze_train = MazeDataset(self.data_dir, "train")
            self.maze_val = MazeDataset(self.data_dir, "valid")
        elif stage == "test":
            self.maze_test = MazeDataset(self.data_dir, "test")

    def train_dataloader(self):
        return DataLoader(self.maze_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.maze_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.maze_test, batch_size=self.batch_size)


class HabitatDataset(Dataset):
    def __init__(self, file_names) -> None:
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        return torch.load(self.file_names[index])


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        train_set, val_set, test_set = torch.load(data_dir)
        self.dataset = LightningDataset(
            train_set, val_set, test_set, batch_size=batch_size
        )

    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def val_dataloader(self):
        return self.dataset.val_dataloader()

    def test_dataloader(self):
        return self.dataset.test_dataloader()


class HabitatDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_scenes = [
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
        ]
        self.test_scenes = [
            "Scioto",
            "Soldier",
            "Springerville",
            "Sugarville",
            "Sussex",
            "Touhy",
            "Victorville",
        ]

        train_set = []
        test_set = []

        for file in os.listdir(data_dir):
            full_path = os.path.join(data_dir, file)
            if file.split("_")[0] in self.train_scenes:
                train_set.append(full_path)
            elif file.split("_")[0] in self.test_scenes:
                test_set.append(full_path)
            else:
                raise Exception("Unknown file category")

        train_dataset = HabitatDataset(train_set)
        test_dataset = HabitatDataset(test_set)
        self.dataset = LightningDataset(
            train_dataset, test_dataset, test_dataset, batch_size=batch_size
        )

    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def val_dataloader(self):
        return self.dataset.val_dataloader()

    def test_dataloader(self):
        return self.dataset.test_dataloader()
