import importlib

import torch

from diffplan.utils.experiment import create_dataloader, get_mechanism


def load_data(args, dataset_type="train", use_loader=False, idx=0, mechanism=None):
    if mechanism is None:
        mechanism = get_mechanism(args.mechanism)

    # load data
    train_loader = create_dataloader(
        args.datafile, dataset_type, args.batch_size, mechanism, shuffle=True
    )

    if use_loader:
        # > Method 1 - use loader
        data = next(iter(train_loader))
        maze_map, goal_map, opt_policy = (
            data["maze"],
            data["goal_map"],
            data["opt_policy"],
        )
        maze_map, goal_map, opt_policy = maze_map[idx], goal_map[idx], opt_policy[idx]

    else:
        # > Method 2 - directly use dataset
        data = train_loader.dataset[idx]
        maze_map, goal_map, opt_policy = (
            data["maze"],
            data["goal_map"],
            data["opt_policy"],
        )

    # > take only one sample and restore the dim
    maze_map, goal_map, opt_policy = (
        maze_map.unsqueeze(0),
        goal_map.unsqueeze(0),
        opt_policy.unsqueeze(0),
    )

    if maze_map.dim() == 3:
        maze_map = maze_map.unsqueeze(1)

    return maze_map, goal_map


def load_model(args, load_path=None):
    mechanism = get_mechanism(args.mechanism)

    model_module = importlib.import_module(args.model)
    model = model_module.Planner(mechanism.num_orient, mechanism.num_actions, args)

    # > load model
    if load_path is not None:
        if not load_path.endswith(".pth"):
            load_path += "/planner.final.pth"
        model.load_state_dict(state_dict=torch.load(load_path)["model"])

    return model


def inference_model(
    args, transformed_out=False, debug_out=True, use_loader=False, idx=0, load_path=None
):
    mechanism = get_mechanism(args.mechanism)

    model_module = importlib.import_module(args.model)
    model = model_module.Planner(mechanism.num_orient, mechanism.num_actions, args)

    # > load model
    if load_path is not None:
        if not load_path.endswith(".pth"):
            load_path += "/planner.final.pth"
        model.load_state_dict(state_dict=torch.load(load_path)["model"])

    # > get data
    maze_map, goal_map = load_data(
        args, mechanism=mechanism, use_loader=use_loader, idx=idx
    )

    if transformed_out:
        # > get transformed output
        out = model.get_transformed_output(maze_map, goal_map)
    else:
        out = model.forward(maze_map, goal_map, debug=debug_out)

    return out
