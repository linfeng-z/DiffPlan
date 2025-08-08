import numpy as np
import torch
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

from diffplan.modules.helpers import TransformedOutput
from diffplan.utils.helpers_maze import convert_goal, convert_actions


def gen_policy_field(mechanism, maze, goal_pos, actions_int):
    if len(maze.shape) == 3:
        maze = maze.squeeze()

    policy_field = np.zeros(shape=list(actions_int.shape) + [2])
    for o in [0]:
        for x in range(actions_int.shape[-2]):
            for y in range(actions_int.shape[-1]):
                if (maze[x, y] == 0.0) or (o, x, y) == goal_pos:
                    policy_field[o, x, y] = np.array([0, 0])
                else:
                    policy_field[o, x, y] = mechanism.action2direction[
                        actions_int[o, x, y]
                    ]
    return policy_field


def prepare_policy_field(mechanism, maze_map, goal_map, logits=None, a_int=None, idx=0):
    goal_pos = convert_goal(goal_map[idx])

    if logits is not None:
        _, a_int = convert_actions(logits[idx])
    else:
        assert a_int is not None

    field = gen_policy_field(
        mechanism=mechanism, maze=maze_map[idx], goal_pos=goal_pos, actions_int=a_int
    )

    map_size = maze_map.shape[-1]
    line_x, line_y = np.meshgrid(
        np.linspace(0, map_size - 1, map_size), np.linspace(0, map_size - 1, map_size)
    )

    return line_x, line_y, field[idx, :, :, 0], field[idx, :, :, 1]


def visualize_policy_field(
    mechanism,
    maze_map,
    goal_map,
    logits=None,
    a_int=None,
    scale=2,
    idx=0,
    plotly=False,
    verbose=True,
    fig_size=None,
):
    assert maze_map.ndim == 4 and goal_map.ndim == 4

    x, y, u, v = prepare_policy_field(mechanism, maze_map, goal_map, logits, a_int, idx)

    if not plotly:
        fig, ax = plt.subplots(figsize=fig_size)
        # ax.quiver(x, y, u, v)
        ax.quiver(x, y, u, v, scale=scale, scale_units="xy")

        ax.imshow(
            maze_map[idx].squeeze(), cmap=plt.cm.get_cmap("Blues_r"), alpha=1.0
        )  # [idx][orient]
        ax.imshow(goal_map[idx].squeeze(), cmap=plt.cm.get_cmap("Reds"), alpha=0.5)
        if verbose:
            ax.set_title("Policy Field (using argmax actions)")
        else:
            ax.axis("off")
    else:
        fig = ff.create_quiver(x, y, u, v, scale=1, scaleratio=1)

    return fig


def visualize_value(maze_map, goal_map, out, group_size=None, fig_size=(15, 15), idx=0):
    """
    visualize Q-value fields
    """

    maze_map = maze_map.squeeze()
    goal_map = goal_map.squeeze()

    if group_size is None:
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2 + 1,
            figsize=fig_size,
        )
        for ax_row in [1]:
            for ax_col in [0, 1]:
                axs[ax_row, ax_col].axis("off")

        axs[0, 0].imshow(maze_map, cmap=plt.cm.get_cmap("Blues_r"))
        axs[0, 1].imshow(goal_map, cmap=plt.cm.get_cmap("Blues_r"))

        v_min, v_max = out.q.min(), out.q.max()

        ax = axs[0, 2]
        cax = ax.imshow(
            out.q.detach().numpy()[idx][0],
            cmap=plt.cm.get_cmap("Blues"),
            vmin=v_min,
            vmax=v_max,
        )
        fig.colorbar(cax, ax=ax)

    else:
        fig, axs = plt.subplots(nrows=2, ncols=2 + group_size, figsize=fig_size)
        for ax_row in [1]:
            for ax_col in [0, 1]:
                axs[ax_row, ax_col].axis("off")

        axs[0, 0].imshow(maze_map, cmap=plt.cm.get_cmap("Blues_r"))
        axs[0, 1].imshow(goal_map, cmap=plt.cm.get_cmap("Blues_r"))

        v_min, v_max = out.q_geo.tensor.min(), out.q_geo.tensor.max()

        for g_channel in range(group_size):
            ax = axs[0, g_channel + 2]
            cax = ax.imshow(
                out.q_geo.tensor.detach().numpy()[idx][g_channel],
                cmap=plt.cm.get_cmap("Blues"),
                vmin=v_min,
                vmax=v_max,
            )
            fig.colorbar(cax, ax=ax)


def visualize_value_plotly(maze_map, goal_map, out, idx=0):
    fig = make_subplots(rows=2, cols=3)

    fig.add_trace(
        go.Heatmap(
            z=maze_map.squeeze(),
            # color_continuous_scale='Blues_r',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=goal_map.squeeze(),
            # color_continuous_scale='Blues_r',
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=out.q.detach().numpy()[idx][0],
            # color_continuous_scale='Blues',
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Heatmap(
            z=out.r.detach().numpy()[idx][0],
            # color_continuous_scale='Blues',
        ),
        row=2,
        col=3,
    )

    fig.show()


def run_visualize_value_equivariance(out: TransformedOutput, idx=0):
    # > get group channel from repr size
    n_g_channel = out.v_geo.tensor.size(1)

    # > init
    fig, axs = plt.subplots(nrows=8, ncols=2 + n_g_channel, figsize=(15, 15))

    obstacle_map, goal_map = out.x_geo.tensor[idx]
    rotated_obstacle_map, rotated_goal_map = out.x_geo_gx.tensor[idx]
    # > note: removed [idx].squeeze() afterwards

    axs[0, 0].imshow(obstacle_map, cmap=plt.cm.get_cmap("Blues_r"))
    axs[0, 1].imshow(goal_map, cmap=plt.cm.get_cmap("Blues_r"))

    axs[2, 0].imshow(rotated_obstacle_map, cmap=plt.cm.get_cmap("Blues_r"))
    axs[2, 1].imshow(rotated_goal_map, cmap=plt.cm.get_cmap("Blues_r"))

    for ax_row in [1, 3, 4, 5, 6, 7]:
        for ax_col in [0, 1]:
            axs[ax_row, ax_col].axis("off")

    r_min, r_max = out.r_geo.tensor.min(), out.r_geo.tensor.max()
    v_min, v_max = out.v_geo.tensor.min(), out.v_geo.tensor.max()

    # original version for value and reward
    for g_channel in range(n_g_channel):
        ax = axs[0, g_channel + 2]
        cax = ax.imshow(
            out.v_geo.tensor.detach().numpy()[idx][g_channel],
            cmap=plt.cm.get_cmap("Blues"),
            vmin=v_min,
            vmax=v_max,
        )
        fig.colorbar(cax, ax=ax)

    for g_channel in range(n_g_channel):
        ax = axs[1, g_channel + 2]
        cax = ax.imshow(
            out.r_geo.tensor.detach().numpy()[idx][g_channel],
            cmap=plt.cm.get_cmap("Blues"),
            vmin=r_min,
            vmax=r_max,
        )
        fig.colorbar(cax, ax=ax)

    # input transformed/rotated version - f(g.x)
    plot_v_fgx = [
        out.v_geo_fgx.tensor.detach().numpy()[idx][g_channel]
        for g_channel in range(n_g_channel)
    ]
    plot_r_fgx = [
        out.r_geo_fgx.tensor.detach().numpy()[idx][g_channel]
        for g_channel in range(n_g_channel)
    ]

    for g_channel in range(n_g_channel):
        ax = axs[2, g_channel + 2]
        cax = ax.imshow(
            plot_v_fgx[g_channel], cmap=plt.cm.get_cmap("Blues"), vmin=v_min, vmax=v_max
        )
        fig.colorbar(cax, ax=ax)

    for g_channel in range(n_g_channel):
        ax = axs[3, g_channel + 2]
        cax = ax.imshow(
            plot_r_fgx[g_channel], cmap=plt.cm.get_cmap("Blues"), vmin=r_min, vmax=r_max
        )
        fig.colorbar(cax, ax=ax)

    # generate
    plot_v_gfx = [
        out.v_geo_fgx.tensor.detach().numpy()[idx][g_channel]
        for g_channel in range(n_g_channel)
    ]
    plot_r_gfx = [
        out.r_geo_gfx.tensor.detach().numpy()[idx][g_channel]
        for g_channel in range(n_g_channel)
    ]

    # output transformed/rotated version - g.f(x)
    for g_channel in range(n_g_channel):
        ax = axs[4, g_channel + 2]
        cax = ax.imshow(
            plot_v_gfx[g_channel], cmap=plt.cm.get_cmap("Blues"), vmin=v_min, vmax=v_max
        )
        fig.colorbar(cax, ax=ax)

    for g_channel in range(n_g_channel):
        ax = axs[5, g_channel + 2]
        cax = ax.imshow(
            plot_r_gfx[g_channel], cmap=plt.cm.get_cmap("Blues"), vmin=r_min, vmax=r_max
        )
        fig.colorbar(cax, ax=ax)

    # visualize difference
    plot_v_ee = [
        plot_v_fgx[g_channel] - plot_v_gfx[g_channel]
        for g_channel in range(n_g_channel)
    ]
    plot_r_ee = [
        plot_r_fgx[g_channel] - plot_r_gfx[g_channel]
        for g_channel in range(n_g_channel)
    ]

    for g_channel in range(n_g_channel):
        ax = axs[6, g_channel + 2]
        cax = ax.imshow(
            plot_v_ee[g_channel],
            cmap=plt.cm.get_cmap("RdBu"),
            vmin=v_min - v_max,
            vmax=v_max - v_min,
        )
        fig.colorbar(cax, ax=ax)

    for g_channel in range(n_g_channel):
        ax = axs[7, g_channel + 2]
        cax = ax.imshow(
            plot_r_ee[g_channel],
            cmap=plt.cm.get_cmap("RdBu"),
            vmin=r_min - r_max,
            vmax=r_max - r_min,
        )
        fig.colorbar(cax, ax=ax)


def run_visualize_policy_field(mechanism, out: TransformedOutput, idx=0):
    # > prepare map
    obstacle_map, goal_map = out.x_geo.tensor[idx]
    obstacle_map_gx, goal_map_gx = out.x_geo_gx.tensor[idx]

    goal_pos = convert_goal(goal_map)
    goal_pos_gx = convert_goal(goal_map_gx)

    # > prepare action
    a_onehot, a_int = convert_actions(out.logits_raw[idx])
    a_onehot_fgx, a_int_fgx = convert_actions(out.logits_fgx[idx])

    a_field_fx = gen_policy_field(
        mechanism=mechanism, maze=obstacle_map, goal_pos=goal_pos, actions_int=a_int
    )

    a_field_fgx = gen_policy_field(
        mechanism=mechanism,
        maze=obstacle_map_gx,
        goal_pos=goal_pos_gx,
        actions_int=a_int_fgx,
    )

    # > prepare
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    line_x, line_y = np.meshgrid(np.linspace(0, 14, 15), np.linspace(0, 14, 15))

    # > plot f(x)
    ax = axs[0]
    ax.quiver(line_x, line_y, a_field_fx[idx, :, :, 0], a_field_fx[idx, :, :, 1])
    ax.imshow(obstacle_map, cmap=plt.cm.get_cmap("Blues_r"), alpha=1.0)  # [idx][orient]
    ax.imshow(goal_map, cmap=plt.cm.get_cmap("Reds"), alpha=0.5)
    ax.set_title("$g.x$")

    # > plot f(g.x)
    ax = axs[1]
    ax.quiver(line_x, line_y, a_field_fgx[idx, :, :, 0], a_field_fgx[idx, :, :, 1])
    ax.imshow(obstacle_map_gx, cmap=plt.cm.get_cmap("Blues_r"), alpha=1.0)
    ax.imshow(goal_map_gx, cmap=plt.cm.get_cmap("Reds"), alpha=0.5)
    ax.set_title("$f(g.x)$")


def run_visualize_policy_ee_matrix(out: TransformedOutput):
    # > squeeze out batch dim
    # > logits: (batch size) x actions x orientations x width x height
    if len(out.logits_fgx.shape) == 4:  # no batch dim
        out.logits_fgx = out.logits_fgx.unsqueeze(0)
        out.logits_gfx = out.logits_gfx.unsqueeze(0)

    comp1 = out.logits_fgx
    comp2 = out.logits_gfx
    ee_table = np.zeros((comp1.size(1), comp2.size(1)))

    print("logits shape", comp1.shape)

    for i in range(comp1.size(1)):
        for j in range(comp2.size(1)):
            # > EE between action i & j: logits from (batch dim) x action (i/j) x orient x width x height
            ee_table[i, j] = (comp1[:, i] - comp2[:, j]).pow(2).mean()  # .norm()

    plt.imshow(ee_table, cmap=plt.cm.get_cmap("Blues"))
    plt.colorbar()

    print("ee_table shape", ee_table.shape)
    print(ee_table)


def run_visualize_policy_equivariance(out: TransformedOutput, idx=0):
    # > get group channel from repr size
    n_g_channel = out.logits_geo.tensor.size(1)

    # > init
    fig, axs = plt.subplots(nrows=4, ncols=2 + n_g_channel, figsize=(15, 15))

    obstacle_map, goal_map = out.x_geo.tensor[idx]
    rotated_obstacle_map, rotated_goal_map = out.x_geo_gx.tensor[idx]
    # > note: removed [idx].squeeze() afterwards

    axs[0, 0].imshow(obstacle_map, cmap=plt.cm.get_cmap("Blues_r"))
    axs[0, 1].imshow(goal_map, cmap=plt.cm.get_cmap("Blues_r"))

    axs[1, 0].imshow(rotated_obstacle_map, cmap=plt.cm.get_cmap("Blues_r"))
    axs[1, 1].imshow(rotated_goal_map, cmap=plt.cm.get_cmap("Blues_r"))

    for ax_row in [2, 3]:
        for ax_col in [0, 1]:
            axs[ax_row, ax_col].axis("off")

    v_min, v_max = out.logits_geo.tensor.min(), out.logits_geo.tensor.max()

    # original version for value and reward
    for g_channel in range(n_g_channel):
        ax = axs[0, g_channel + 2]
        cax = ax.imshow(
            out.logits_geo.tensor.detach().numpy()[idx][g_channel],
            cmap=plt.cm.get_cmap("Blues"),
            vmin=v_min,
            vmax=v_max,
        )
        fig.colorbar(cax, ax=ax)

    # input transformed/rotated version - f(g.x)
    plot_logits_fgx = [
        out.logits_geo_fgx.tensor.detach().numpy()[idx][g_channel]
        for g_channel in range(n_g_channel)
    ]

    for g_channel in range(n_g_channel):
        ax = axs[1, g_channel + 2]
        cax = ax.imshow(
            plot_logits_fgx[g_channel],
            cmap=plt.cm.get_cmap("Blues"),
            vmin=v_min,
            vmax=v_max,
        )
        fig.colorbar(cax, ax=ax)

    # generate
    plot_logits_gfx = [
        out.logits_geo_gfx.tensor.detach().numpy()[idx][g_channel]
        for g_channel in range(n_g_channel)
    ]
    # rotate(v_trivial.tensor, 1).detach().numpy()[idx][g_channel],
    # rotate_p4(v_trivial.tensor, 1).detach().numpy()[idx][g_channel],

    # output transformed/rotated version - g.f(x)
    for g_channel in range(n_g_channel):
        ax = axs[2, g_channel + 2]
        cax = ax.imshow(
            plot_logits_gfx[g_channel],
            cmap=plt.cm.get_cmap("Blues"),
            vmin=v_min,
            vmax=v_max,
        )
        fig.colorbar(cax, ax=ax)

    # visualize difference
    plot_logits_ee = [
        plot_logits_fgx[g_channel] - plot_logits_gfx[g_channel]
        for g_channel in range(n_g_channel)
    ]
    ee_min, ee_max = (
        torch.tensor(plot_logits_ee).min(),
        torch.tensor(plot_logits_ee).max(),
    )

    for g_channel in range(n_g_channel):
        ax = axs[3, g_channel + 2]
        cax = ax.imshow(
            plot_logits_ee[g_channel],
            cmap=plt.cm.get_cmap("RdBu"),
            vmin=v_min - v_max,
            vmax=v_max - v_min
            # vmin=ee_min, vmax=ee_max
        )
        fig.colorbar(cax, ax=ax)

    # Note: Figure title can be added if needed
    # fig.suptitle('Visualization Title')
    # plt.title('Visualization Title')
