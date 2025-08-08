import numpy as np
import pyglet

from diffplan.envs.maze_env import RandomMaze
from diffplan.envs.visual_nav.nav_wrapper import MazeNavWorld, GraphNavWorld


class VisNavEnv(RandomMaze):
    # @ex.capture
    def __init__(
        self,
        mechanism,
        min_maze_size,
        max_maze_size,
        min_decimation,
        max_decimation,
        start_pos=(1, 1),
        obs_width=32,
        obs_height=32,
    ):
        super().__init__(
            mechanism,
            min_maze_size,
            max_maze_size,
            min_decimation,
            max_decimation,
            start_pos,
        )

        self.obs_width = obs_width
        self.obs_height = obs_height

        # > init mini world and only reset map; some init map is needed but not actually used
        self.nav_world = MazeNavWorld(
            maze_map=np.ones((5, 5)),
            obs_width=self.obs_width,
            obs_height=self.obs_height,
        )

    def reset(self):
        # > reset the RandomMaze - randomly generates map
        maze, player_map, goal_map = super().reset()

        # > set up new map in MiniWorld env & gen world using the new map
        self.nav_world.set_map(maze)
        self.nav_world.reset()
        # TODO check whether mechanism should affect the data gen?

        return maze, player_map, goal_map

    def close(self):
        """
        close mini-world env to save memory
        """
        if self.nav_world.window:
            self.nav_world.window.close()
        pyglet.app.exit()
        self.nav_world.close()
        # del self.nav_world

    def render_all_pano(self):
        return self.nav_world.get_all_pano(vis_top=False, plot_top=False)


class VisNavGraphEnv(RandomMaze):
    # @ex.capture
    def __init__(
        self,
        mechanism,
        min_maze_size,
        max_maze_size,
        min_decimation,
        max_decimation,
        start_pos=(1, 1),
        obs_width=32,
        obs_height=32,
    ):
        super().__init__(
            mechanism,
            min_maze_size,
            max_maze_size,
            min_decimation,
            max_decimation,
            start_pos,
        )

        self.obs_width = obs_width
        self.obs_height = obs_height

        # > init mini world and only reset map; some init map is needed but not actually used
        self.nav_world = GraphNavWorld(
            maze_map=np.ones((5, 5)),
            obs_width=self.obs_width,
            obs_height=self.obs_height,
        )

    def reset(self):
        # > reset the RandomMaze - randomly generates map
        maze, player_map, goal_map = super().reset()

        # > set up new map in MiniWorld env & gen world using the new map
        self.nav_world.set_map(maze, goal_map)
        self.nav_world.reset()
        # TODO check whether mechanism should affect the data gen?

        return maze, player_map, goal_map

    def close(self):
        """
        close mini-world env to save memory
        """
        if self.nav_world.window:
            self.nav_world.window.close()
        pyglet.app.exit()
        self.nav_world.close()
        # del self.nav_world

    def render_all_pano(self):
        return self.nav_world.get_all_pano(vis_top=False, plot_top=False)
