from __future__ import print_function

import math

import numpy as np
from numpy import random as npr


def generate_maze(maze_size, decimation, start_pos=(1, 1), manual_boundary=False):
    maze = np.zeros((maze_size, maze_size))

    stack = [((start_pos[0], start_pos[1]), (0, 0))]

    def add_stack(next_pos, next_dir):
        if (next_pos[0] < 0) or (next_pos[0] >= maze_size):
            return
        if (next_pos[1] < 0) or (next_pos[1] >= maze_size):
            return
        if maze[next_pos[0]][next_pos[1]] == 0.0:
            stack.append((next_pos, next_dir))

    while stack:
        pos, prev_dir = stack.pop()
        # Has this not been filled since being added?
        if maze[pos[0]][pos[1]] == 1.0:
            continue

        # Fill in this point + break down wall from previous position
        maze[pos[0]][pos[1]] = 1.0
        maze[pos[0] - prev_dir[0]][pos[1] - prev_dir[1]] = 1.0

        choices = []
        choices.append(((pos[0] - 2, pos[1]), (-1, 0)))
        choices.append(((pos[0], pos[1] + 2), (0, 1)))
        choices.append(((pos[0], pos[1] - 2), (0, -1)))
        choices.append(((pos[0] + 2, pos[1]), (1, 0)))

        perm = np.random.permutation(np.array(range(4)))
        for i in range(4):
            choice = choices[perm[i]]
            add_stack(choice[0], choice[1])

    for y in range(1, maze_size - 1):
        for x in range(1, maze_size - 1):
            if np.random.uniform() < decimation:
                maze[y][x] = 1.0

    # TODO manually add boundary walls
    if manual_boundary:
        for i in range(maze_size):
            maze[i][maze_size - 1] = 0.0
            maze[maze_size - 1][i] = 0.0

    return maze


class RandomMaze:
    def __init__(
        self,
        mechanism,
        min_maze_size,
        max_maze_size,
        min_decimation,
        max_decimation,
        start_pos=(1, 1),
        goal_pos=None,
    ):
        self.mechanism = mechanism
        self.min_maze_size = min_maze_size
        self.max_maze_size = max_maze_size
        self.min_decimation = min_decimation
        self.max_decimation = max_decimation
        self.start_pos = start_pos

        self.input_goal_pos = goal_pos
        if self.input_goal_pos is not None:
            self.goal_pos = self.input_goal_pos

    def _is_goal_pos(self, pos):
        """Returns true if pos is equal to the goal position."""
        return pos[0] == self.goal_pos[0] and pos[1] == self.goal_pos[1]

    def _get_state(self):
        """Returns the current state."""
        goal_map = np.zeros((self.mechanism.num_orient, self.maze_size, self.maze_size))
        goal_map[self.goal_orient, self.goal_pos[0], self.goal_pos[1]] = 1.0

        player_map = np.zeros(
            (self.mechanism.num_orient, self.maze_size, self.maze_size)
        )
        player_map[self.player_orient, self.player_pos[0], self.player_pos[1]] = 1.0

        # Check if agent has reached the goal state
        reward = 0
        terminal = False
        if (self.player_orient == self.goal_orient) and self._is_goal_pos(
            self.player_pos
        ):
            reward = 1
            terminal = True

        return np.copy(self.maze), player_map, goal_map, reward, terminal

    def reset(self):
        """Resets the maze."""
        if self.min_maze_size == self.max_maze_size:
            self.maze_size = self.min_maze_size
        else:
            self.maze_size = self.min_maze_size + 2 * npr.randint(
                math.floor((self.max_maze_size - self.min_maze_size) / 2)
            )
        if self.min_decimation == self.max_decimation:
            self.decimation = self.min_decimation
        else:
            self.decimation = npr.uniform(self.min_decimation, self.max_decimation)
        self.maze = generate_maze(
            self.maze_size, self.decimation, start_pos=self.start_pos
        )

        # Randomly sample a goal location
        if self.input_goal_pos is None:
            self.goal_pos = (
                npr.randint(1, self.maze_size - 1),
                npr.randint(1, self.maze_size - 1),
            )
        # > Or use specified goal
        else:
            self.goal_pos = self.input_goal_pos

        while self._is_goal_pos(self.start_pos):
            self.goal_pos = (
                npr.randint(1, self.maze_size - 1),
                npr.randint(1, self.maze_size - 1),
            )
        self.goal_orient = npr.randint(self.mechanism.num_orient)

        # Free the maze at the goal location
        self.maze[self.goal_pos[0]][self.goal_pos[1]] = 1.0

        # Player start position
        self.player_pos = (self.start_pos[0], self.start_pos[1])

        # Sample player orientation
        self.player_orient = npr.randint(self.mechanism.num_orient)

        maze, player_map, goal_map, _, _ = self._get_state()
        return maze, player_map, goal_map

    def step(self, action):
        # Compute neighbors for the current state.
        neighbors = self.neighbors_func(
            self.maze, self.player_orient, self.player_pos[0], self.player_pos[1]
        )
        assert (action > 0) and (action < len(neighbors))
        self.player_orient, self.player_pos[0], self.player_pos[1] = neighbors[action]
        return self._get_state()
