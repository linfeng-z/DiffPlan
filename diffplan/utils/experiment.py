import os
import numpy as np

import torch
from torch import optim as optim

from diffplan.envs.maze_utils import MazeDataset
from diffplan.utils.mechanism import (
    DifferentialDrive,
    NorthEastWestSouth,
    Moore,
    FourAbsClockwise,
    FourAbsCounterClockwise,
    FourAbsCounterClockwiseWrap,
    MooreCompatibleCounterClockwise,
    NorthEastWestSouthWrap,
)


def get_optimizer(args, parameters):
    if args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(
            parameters, lr=args.lr, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(
            parameters, lr=args.lr, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(
            parameters, lr=args.lr, momentum=0.95, weight_decay=args.weight_decay
        )
    else:
        raise ValueError("Unsupported optimizer: %s" % args.optimizer)

    if args.scheduler is None:
        scheduler = None
    elif args.scheduler == "cosine":
        # TODO hardcode the max iters to be 30
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=1e-6, verbose=True
        )
    else:
        raise ValueError("Unsupported scheduler")

    return optimizer, scheduler


def get_mechanism(mechanism):
    str2class = {
        "news": NorthEastWestSouth(),
        "news-wrap": NorthEastWestSouthWrap(),
        "diffdrive": DifferentialDrive(),
        "moore": Moore(),
        "4abs": FourAbsClockwise(),
        "4abs-cc": FourAbsCounterClockwise(),
        "4abs-cc-wrap": FourAbsCounterClockwiseWrap(),
        "8abs": MooreCompatibleCounterClockwise(),
    }

    return str2class[mechanism]


def create_dataloader(datafile, dataset_type, batch_size, mechanism, shuffle=False):
    """
    Creates a maze DataLoader.
    Args:
      datafile (string): Path to the dataset
      dataset_type (string): One of "train", "valid", or "test"
      batch_size (int): The batch size
      shuffle (bool): Whether to shuffle the data
    """
    dataset = MazeDataset(datafile, dataset_type)
    assert dataset.num_actions == mechanism.num_actions
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=(16 if "Visual3DNav" in datafile else 0),
    )


def create_save_dir(save_directory, save_timestamp):
    """
    Creates and returns path to the save directory.
    """

    # > Use the same timestamp as W&B experiment logging for saving model
    save_directory += str(save_timestamp)

    try:
        os.makedirs(save_directory)
    except OSError:
        if not os.path.isdir(save_directory):
            raise

    return save_directory + "/planner"


def print_row(width, items):
    """
    Prints the given items.
    Args:
      width (int): Character length for each column.
      items (list): List of items to print.
    """

    def fmt_item(x):
        if isinstance(x, np.ndarray):
            assert x.ndim == 0
            x = x.item()
        if isinstance(x, float):
            rep = "%.3f" % x
        else:
            rep = str(x)
        return rep.ljust(width)

    print(" | ".join(fmt_item(item) for item in items))


def print_stats(info):
    """Prints performance statistics output from Runner."""
    print_row(10, ["Loss", "Err", "% Optimal", "% Success"])
    print_row(
        10,
        [info["avg_loss"], info["avg_error"], info["avg_optimal"], info["avg_success"]],
    )
    return info
