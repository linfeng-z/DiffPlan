import argparse
import importlib
import os
import matplotlib.pyplot as plt

import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from escnn.nn.init import generalized_he_init
from escnn.nn.modules.conv.r2convolution import R2Conv
from escnn.nn.modules.linear import Linear
from lightning.fabric import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from rich import pretty
from rich.console import Console

from diffplan.modules.Base import LitGraphPlanner, LitGridPlanner
from diffplan.modules.Dataset import GridDataModule, GraphDataModule, HabitatDataModule
from diffplan.utils.equiv_model_checkpoint import ModelCheckpoint

# Configure rich's pretty printing
pretty.install()
console = Console()

# For local machine, use TkAgg, otherwise Agg for headless
matplotlib.use("Agg")
# Set matplotlib figure limit
matplotlib.rcParams['figure.max_open_warning'] = 50

# Enable tensor cores for better performance
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument("--enable_sweep", action="store_true", help="Enable wandb sweep")
parser.add_argument("--sweep_id", type=str, default="nu-team/corl/xa5a259w")
parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logger")
parser.add_argument("--test", action="store_true", help="Testing mode")
parser.add_argument("--data_path", type=str, default="data/m15_graph-cc_10k.pth")
parser.add_argument("--notes", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--task",
    type=str,
    default="GridWorld",
    choices=["GridWorld", "GraphWorld", "VisNav", "Habitat"],
)
parser.add_argument(
    "--has_obstacle", action="store_true", help="Include obstalce node in the graph"
)
parser.add_argument(
    "--obsv_mode",
    type=str,
    default="full",
    choices=["full", "partial"],
    help="Observation mode",
)
parser.add_argument("--maze_size", type=int, default=15)

parser.add_argument_group("Training", "training configurations")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--optimizer", type=str, choices=["RMSprop", "Adam", "SGD"], default="RMSprop"
)
parser.add_argument("--algorithm", default="MP-VIN-v3")
parser.add_argument(
    "--cont_action", action="store_true", help="Continous action (relative translation)"
)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument(
    "--no_weight_init",
    action="store_true",
    help="Disable model weight xavier initialization",
)

parser.add_argument_group("Model", "model configurations")
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--l_h", type=int, default=150)
parser.add_argument("--l_q", type=int, default=40)
parser.add_argument(
    "--mp_h", type=int, default=256, help="hidden size of Message Passing Layer self.h"
)
parser.add_argument(
    "--mp_q", type=int, default=256, help="hidden size of Message Passing Layer self.q"
)
parser.add_argument(
    "--mp_agg",
    type=str,
    default="add",
    choices=["add", "mean", "max"],
    help="Message Passing Layer aggregation method",
)
parser.add_argument(
    "--v_repr", type=str, default="regular", choices=["regular", "trivial"]
)
parser.add_argument(
    "--q_repr", type=str, default="regular", choices=["regular", "trivial"]
)
parser.add_argument(
    "--visual_feat", type=int, default=128, help="Output dimension of ResNet"
)

# Equivariance configuration parameters
parser.add_argument("--f", type=int, default=3, help="Number of feature channels for equivariant models")
parser.add_argument("--group", type=str, default="d4", help="Symmetry group (d4, d8, etc.)")
parser.add_argument("--latent_dim_factor", type=str, default="sqrt", help="Latent dimension scaling factor")
parser.add_argument("--latent_repr", type=str, default="regular", help="Latent representation type")
parser.add_argument(
    "--no_equiv_policy", action="store_true", help="Disable Equivariant Layer of pi"
)
parser.add_argument(
    "--test_equiv_err", action="store_true", help="Test Equivariance Error mode"
)
parser.add_argument(
    "--equiv_err_theta",
    type=int,
    help="Testing equivariance error given theta rotation",
)
parser.add_argument("--num_orient", type=int, default=1)
parser.add_argument("--num_actions", type=int, default=4)

args = parser.parse_args()
seed_everything(args.seed)


def kaiming_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, R2Conv):
            generalized_he_init(m.weights.data, m.basisexpansion)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, Linear):
            generalized_he_init(m.weights.data, m.basisexpansion)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


def get_run_name(args):
    name = args.algorithm
    name += f"_{args.task}"
    name += f"_optimizer_{args.optimizer}"
    name += f"_lr_{args.lr}"
    name += f"_k_{args.k}"
    name += f"_l_h_{args.l_h}"
    name += f"_l_q_{args.l_q}"
    name += f"_mp_h_{args.mp_h}"
    name += f"_mp_q_{args.mp_q}"
    name += f"_mp_agg_{args.mp_agg}"
    if args.algorithm == "Sym-MP-VIN":
        name += f"_group_{args.group}"
    if args.cont_action:
        name += "_cont_action"
    if args.no_equiv_policy:
        name += "_no_equiv_policy"
    name += f"_{args.task}"
    return name


def merge_args_config(args, config):
    args.loss = config.loss
    args.optimizer = config.optimizer
    return args


def main():
    global args
    if args.enable_sweep:
        wandb.init()
        config = wandb.config
        args = merge_args_config(args, config)

    console.rule("[bold green]Initializing model")
    if args.algorithm in ["VIN", "SymVIN", "SymVIN-new", "DE-VIN", "DE-SymVIN"] or "GPPN" in args.algorithm:
        model_module = importlib.import_module("diffplan.models_grid." + args.algorithm)
        net = model_module.Planner(num_orient=args.num_orient, num_actions=args.num_actions, args=args)
        model = LitGridPlanner(net, args=args)
    elif "GCN" in args.algorithm:
        model_module = importlib.import_module("diffplan.models_graph." + args.algorithm)
        net = model_module.GCNPlanningNetwork(args=args)
        model = LitGraphPlanner(net, args=args)
    elif "GAT" in args.algorithm:
        model_module = importlib.import_module("diffplan.models_graph." + args.algorithm)
        net = model_module.GATVINPlanningNetwork(args=args)
        model = LitGraphPlanner(net, args=args)
    elif "MP" in args.algorithm:
        console.rule(f"[bold cyan]Loading MP model: {args.algorithm}")
        model_module = importlib.import_module("diffplan.models_graph." + args.algorithm)
        net = model_module.MPPlanningNetwork(args=args)
        model = LitGraphPlanner(net, args=args)
    else:
        raise ValueError("Invalid algorithm")

    if not args.no_weight_init:
        console.rule("[yellow]Initializing model weights")
        net.apply(kaiming_init)

    run_name = get_run_name(args=args)
    console.rule(f"[bold blue]Run name: {run_name}")
    
    # choose logger
    loggers = []
    loggers.append(TensorBoardLogger(f"logs/{run_name}"))
    if not args.disable_wandb:
        loggers.append(
            WandbLogger(
                name=run_name, entity="nu-team", project="corl", notes=args.notes
            )
        )

    # model checkpoint
    if True:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/success", mode="max", filename="best"
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/acc", mode="max", filename="best"
        )

    console.rule("[bold green]Setting up trainer")
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy=None,
        log_every_n_steps=5,  # Reduced from 50 to match smaller batch size
    )

    # define different datamodule
    console.rule("[bold green]Setting up data module")
    if args.task == "Habitat":
        data_module = HabitatDataModule(args.data_path, args.batch_size)
    elif args.task == "GraphWorld":
        data_module = GraphDataModule(args.data_path, args.batch_size)
    elif args.algorithm in ["VIN", "SymVIN"]:
        data_module = GridDataModule(args.data_path, args.batch_size)
    else:
        data_module = GraphDataModule(args.data_path, args.batch_size)

    if args.test_equiv_err:
        console.rule("[bold red]Testing equivariance error")
        for theta in np.arange(0, 4, 1):
            args.equiv_err_theta = theta
            net = model_module.MPPlanningNetwork(args=args)
            model = LitGraphPlanner(net, args=args)
            trainer.test(model=model, datamodule=data_module)
            plt.close('all')  # Close figures after each test
    elif args.test:
        # testing
        console.rule("[bold yellow]Starting testing")
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=input("Please input checkpoint path: "),
        )
        plt.close('all')  # Close figures after testing
    else:
        # training
        console.rule("[bold green]Starting training")
        trainer.fit(model=model, datamodule=data_module)
        trainer.model.eval()  # LHY: fix escnn bug
        trainer.test(model=model, ckpt_path="best", datamodule=data_module)
        plt.close('all')  # Close figures after training


if __name__ == "__main__":
    if args.enable_sweep:
        wandb.agent(args.sweep_id, main)
    else:
        main()

