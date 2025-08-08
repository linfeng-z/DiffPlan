import wandb
import pandas as pd
import seaborn as sns
from torch_geometric.utils import remove_self_loops
import torch_geometric
import networkx as nx
import numpy as np
import io
import PIL
from matplotlib.colors import Normalize

# matplotlib.rc_file_defaults()

import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import to_networkx as torch_to_networkx

model2name = {
    "models.VIN": "VIN",
    "models.GPPN": "GPPN",
    # 'models.E2-VIN-v4': 'SymVIN',
    "models.SymVIN": "SymVIN",
    # 'models.Conv-GPPN': 'ConvGPPN',
    # 'models.E2-Conv-GPPN': 'SymGPPN',
    # 'models.Conv-GPPN': 'ConvGPPN',
    "models.DecoupledGPPN-ConvGRU": "ConvGPPN",
    "models.DE-VIN-v3": "DE-VIN",
    "models.DE-SymVIN": "DE-SymVIN",
    "models.DE-ConvGPPN": "DE-ConvGPPN",
}

datafile2name = {
    "data/m15_4abs-cc_10k.npz": "$15 \\times 15$",
    "data/m27_4abs-cc_10k.npz": "$27 \\times 27$",
    "data/m49_4abs-cc_10k.npz": "$49 \\times 49$",
    "/mnt_host/zlf-local-data/symplan/Visual3DNav_10k_15_4abs-cc.npz": "Visual Nav ($15 \\times 15$)",
    "data/m15_4abs-cc_1k.npz": "$15 \\times 15$ (1K)",
    "data/m27_4abs-cc_1k.npz": "$27 \\times 27$ (1K)",
    "data/m49_4abs-cc_1k.npz": "$49 \\times 49$ (1K)",
    "data/Arm2DoFsEnv_96_10000_18_one_hot_4abs-cc-wrap.npz": "$18 \\times 18$ Manipulation",
    "data/Arm2DoFsEnv_96_10000_36_one_hot_4abs-cc-wrap.npz": "$36 \\times 36$ Manipulation",
    "data/Arm2DoFsWorkSpaceEnv_96_10000_18_one_hot_4abs-cc-wrap.npz": "$18 \\times 18$ Workspace Manipulation",
}

implicit_names = ["DE-VIN", "DE-SymVIN", "DE-ConvGPPN", "DE-SPT"]
explicit_names = [
    "VIN",
    "SymVIN",
    # 'GPPN',
    "ConvGPPN",
    "SPT",
]

core_keys = (
    "f",
    "k",
    "l_q",
    "model",
    "datafile",
)


def retrieve_wandb(project="zhao0625/DiffPlanLib-DEPlan"):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(project)

    summary_list = []
    config_list = []
    name_list = []

    for run in runs:
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        config_list.append(config)

        # run.name is the name of the run.
        name_list.append(run.name)

    # > to dataframe
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list})
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)

    return {
        "runs": runs,
        "all_df": all_df,
        "summary_df": summary_df,
        "config_df": config_df,
        "name_df": name_df,
    }


def collect_learning_curves(
    runs,
    indices,
    key_plot="Train/avg_success",
    key_step="_step",
    config_keys=core_keys,
    skip_incomplete=False,
):
    # _aggregate_df = pd.DataFrame()
    _agg_list = []

    # for _i, _run in enumerate(runs):
    for _i in indices:
        _run = runs[_i]

        # > skip incomplete runs (if less than set epochs)
        if skip_incomplete and len(_run.history().index) < runs[3].config["epochs"]:
            print(f"Skipped: {_i}, length = {len(_run.history().index)}")
            continue

        # > if found illegal runs without stats, skip
        if (key_step not in _run.history().columns) or (
            key_plot not in _run.history().columns
        ):
            print(f"No curve stats: {_i}, columns: {_run.history().columns}")
            continue

        # > add values to plot
        curve_df = _run.history()[[key_step, key_plot]]

        # > add id
        curve_df["id"] = _run.id

        # > add all config
        for _config_key in _run.config.keys():  # or only in 'config_keys'
            curve_df[_config_key] = _run.config[_config_key]

        _agg_list.append(curve_df)
        # _aggregate_df = pd.concat([_aggregate_df, curve_df], axis=0)

        print(f"Collected: {_i}, length = {len(curve_df.index)}, run = {_run}")

    print("Concatenating...")
    _aggregate_df = pd.concat(_agg_list, axis=0)

    return _aggregate_df


def rename_model_df(
    df,
    y_plot="Train/avg_success",
    y_name="Successful Rate",
    x_plot="_step",
    x_name="Epochs",
):
    # > replace model name
    df = df.replace(model2name)

    # > rename columns
    df = df.rename(columns={x_plot: x_name, y_plot: y_name})

    # > add data map sizes
    df["Task"] = df["datafile"].apply(lambda f: datafile2name[f])

    return df


# def preprocess_plot_df(agg_df, all_df, filters, hue_plot, note, runs, x_name, x_plot, y_name, y_plot):
def preprocess_curve_df(
    y_plot="Train/avg_success",
    y_name="Successful Rate",
    x_plot="_step",
    x_name="Epochs",
    hue_plot="model",
    hue_order=None,
    runs=None,
    all_df=None,
    agg_df=None,
    note=None,
    filters=True,
):
    """
    Preprocessing for plotting learning curves
    """

    # > collect curve data
    if agg_df is None:
        aggregate_df = collect_learning_curves(
            runs=runs,
            skip_incomplete=True,
            key_plot=y_plot,
            indices=all_df.index[(all_df["note"] == note) & filters],
        )
    else:
        aggregate_df = agg_df

    # > renaming
    aggregate_df = aggregate_df.replace(model2name)

    # > grouping
    grouped_df = aggregate_df.groupby([x_plot, y_plot, hue_plot]).mean().reset_index()
    grouped_df = grouped_df.rename(columns={x_plot: x_name, y_plot: y_name})
    return grouped_df


def get_curves(
    grouped_df,
    y_name="Successful Rate",
    x_name="Epochs",
    hue_plot="model",
    hue_order=None,
):
    """
    key_plot = 'Train/avg_success' or 'Validation/avg_success'
    """

    plt.figure(figsize=(5, 3))
    g = sns.lineplot(
        data=grouped_df,
        x=x_name,
        y=y_name,
        hue=hue_plot,
        hue_order=hue_order,
    )

    plt.tight_layout()
    fig = g.get_figure()

    return fig


def get_latex(
    df,
    filters=None,
    y_plot="Train/avg_success",
    y_name="Successful Rate",
    x_plot="_step",
    x_name="Epochs",
):
    # TODO filter columns
    # TODO filter entries

    return df.to_latex()


def visualize_graph_v(
    graph,
    v,
    title="Untitled",
    ax=None,
):
    """
    Visualize the v values of the graph
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Process graph data
    if hasattr(graph, 'edge_index'):
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        g = torch_to_networkx(graph, to_undirected=True)
        g.remove_edges_from(nx.selfloop_edges(g))
        
        # Extract features and positions
        features = graph.x.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        v[:, 0][v[:, 0] == np.min(v[:, 0])] = -np.inf
        positions = dict(zip(g.nodes, features[:, :2]))
        
        # Draw graph
        nx.draw(
            g,
            positions,
            node_color=v,
            cmap='Greys',
            ax=ax
        )
        
        # Plot goal position
        goal_pos = features[features[:, 2] == 1][0, :2]
        ax.plot(goal_pos[0], goal_pos[1], "r*", markersize=15)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='Greys', norm=Normalize(vmin=v.min(), vmax=v.max()))
        plt.colorbar(sm, ax=ax)
    
    if title:
        ax.set_title(title)
    
    return ax
