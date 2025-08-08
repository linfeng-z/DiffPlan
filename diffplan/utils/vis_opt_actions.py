import matplotlib.pyplot as plt

from diffplan.envs.generate_dataset import extract_goal
from diffplan.envs.maze_env import RandomMaze
from diffplan.envs.maze_utils import extract_policy
from diffplan.utils.dijkstra import dijkstra_dist
from diffplan.utils.experiment import get_mechanism
from diffplan.utils.vis_fields import visualize_policy_field


def get_map(mechanism_name="4abs-cc", size=5, goal_pos=(1, 3)):
    mechanism = get_mechanism(mechanism_name)

    env = RandomMaze(
        mechanism,
        min_maze_size=size,
        max_maze_size=size,
        min_decimation=0.0,
        max_decimation=0.0,
        start_pos=(1, 1),
        goal_pos=goal_pos,
    )

    maze_map, player_map, goal_map = env.reset()

    return maze_map, player_map, goal_map


def plot_maze_goal(maze_map, goal_map):
    fig, ax = plt.subplots()

    ax.imshow(maze_map, cmap=plt.cm.get_cmap("Blues_r"), alpha=1.0)
    ax.imshow(goal_map[0], cmap=plt.cm.get_cmap("Reds"), alpha=0.5)
    ax.axis("off")

    return fig


def plot_opt_policy(maze_map, goal_map, mechanism_name="4abs-cc"):
    mechanism = get_mechanism(mechanism_name)

    maze_size = maze_map.shape[0]

    opt_value = dijkstra_dist(
        maze_map, mechanism, extract_goal(goal_map, mechanism, maze_size=maze_size)
    )
    opt_policy = extract_policy(maze_map, mechanism, opt_value, is_full_policy=False)

    fig = visualize_policy_field(
        mechanism,
        maze_map.reshape(1, 1, maze_size, maze_size),
        goal_map.reshape(1, 1, maze_size, maze_size),
        logits=opt_policy.reshape(1, 4, 1, maze_size, maze_size),
        verbose=False,
    )

    return fig
