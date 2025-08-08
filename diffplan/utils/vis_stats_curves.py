import pandas as pd
import seaborn as sns
import wandb


def retrieve_runs(project="zhao0625/SymPlan-Debug-GPPN"):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(project)

    return runs


def get_all_stats(runs):
    summary_list = []
    config_list = []
    name_list = []

    for run in runs:
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)  # noqa

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        config_list.append(config)

        # run.name is the name of the run.
        name_list.append(run.name)

    # > collect into one
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list})
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)

    return all_df


def collect_stats(
    runs,
    indices,
    key_plot="Train/avg_SPL",
    key_step="_step",
    config_keys=(
        "f",
        "k",
        "l_q",
        "model",
    ),
    skip_incomplete=False,
):
    _aggregate_df = pd.DataFrame()

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
        # > add other config
        for _config_key in config_keys:
            curve_df[_config_key] = _run.config[_config_key]

        _aggregate_df = pd.concat([_aggregate_df, curve_df], axis=0)

        print(f"Collected: {_i}, length = {len(curve_df.index)}, run = {_run}")

    return _aggregate_df


def average_grouped_stats(
    stats_df,
    grouped_run=False,
    key_hue="model",
    grouping_keys=("f", "k", "l_q"),
    key_step="_step",
):
    grouping_keys = [key_hue] + list(grouping_keys)
    # > if grouping over repeated runs (or other non-added config), then don't add id
    # > otherwise enable grouping over runs
    if not grouped_run:
        grouping_keys.append("id")
    # > average over steps! don't add step to grouping
    if key_step in grouping_keys:
        grouping_keys = grouping_keys.remove(key_step)

    # > grouped by runs but not
    avg_grouped_df = stats_df.groupby(grouping_keys).mean()

    return avg_grouped_df


def plot_curves(
    stats_df,
    key_value="Train/avg_SPL",
    key_step="_step",
    key_hue="model",
    grouping_keys=("f", "k", "l_q"),
):
    grouping_keys = list(grouping_keys)
    if key_step not in grouping_keys:
        grouping_keys.append(key_step)
    if key_hue not in grouping_keys:
        grouping_keys.append(key_hue)

    # > average over all configs of a model for grouping
    grouped_df = stats_df.groupby(list(grouping_keys)).mean().reset_index()

    g = sns.lineplot(data=grouped_df, x=key_step, y=key_value, hue=key_hue)

    return g
