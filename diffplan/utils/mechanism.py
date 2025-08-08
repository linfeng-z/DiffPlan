# -*- coding: utf-8 -*-
from __future__ import print_function
import abc

import numpy as np

name2direction = {
    "up": [0, 1],
    "right": [1, 0],
    "left": [-1, 0],
    "down": [0, -1],
    "n": [0, 1],
    "e": [1, 0],
    "w": [-1, 0],
    "s": [0, -1],
}

name2direction.update("")


def names2directions(*names):
    res = []
    for name in names:
        res.append(name2direction[name])
    return np.array(res)


def _north(maze, p_orient, p_y, p_x):
    if (p_y > 0) and (maze[p_y - 1][p_x] != 0.0):
        return p_orient, p_y - 1, p_x
    return p_orient, p_y, p_x


def _south(maze, p_orient, p_y, p_x):
    if (p_y < (maze.shape[0] - 1)) and (maze[p_y + 1][p_x] != 0.0):
        return p_orient, p_y + 1, p_x
    return p_orient, p_y, p_x


def _west(maze, p_orient, p_y, p_x):
    if (p_x > 0) and (maze[p_y][p_x - 1] != 0.0):
        return p_orient, p_y, p_x - 1
    return p_orient, p_y, p_x


def _east(maze, p_orient, p_y, p_x):
    if (p_x < (maze.shape[1] - 1)) and (maze[p_y][p_x + 1] != 0.0):
        return p_orient, p_y, p_x + 1
    return p_orient, p_y, p_x


def _north_with_obstacle(maze, p_orient, p_y, p_x):
    if p_y > 0:
        return p_orient, p_y - 1, p_x
    return p_orient, p_y, p_x


def _south_with_obstacle(maze, p_orient, p_y, p_x):
    if p_y < (maze.shape[0] - 1):
        return p_orient, p_y + 1, p_x
    return p_orient, p_y, p_x


def _west_with_obstacle(maze, p_orient, p_y, p_x):
    if p_x > 0:
        return p_orient, p_y, p_x - 1
    return p_orient, p_y, p_x


def _east_with_obstacle(maze, p_orient, p_y, p_x):
    if p_x < (maze.shape[1] - 1):
        return p_orient, p_y, p_x + 1
    return p_orient, p_y, p_x


def _north_wrap(maze, p_orient, p_y, p_x):
    p_y_next = (p_y - 1) % maze.shape[0]
    if maze[p_y_next][p_x] != 0.0:
        return p_orient, p_y_next, p_x
    return p_orient, p_y, p_x


def _south_wrap(maze, p_orient, p_y, p_x):
    p_y_next = (p_y + 1) % maze.shape[0]
    if maze[p_y_next][p_x] != 0.0:
        return p_orient, p_y_next, p_x
    return p_orient, p_y, p_x


def _west_wrap(maze, p_orient, p_y, p_x):
    p_x_next = (p_x - 1) % maze.shape[0]
    if maze[p_y][p_x_next] != 0.0:
        return p_orient, p_y, p_x_next
    return p_orient, p_y, p_x


def _east_wrap(maze, p_orient, p_y, p_x):
    p_x_next = (p_x + 1) % maze.shape[0]
    if maze[p_y][p_x_next] != 0.0:
        return p_orient, p_y, p_x_next
    return p_orient, p_y, p_x


def _northwest(maze, p_orient, p_y, p_x):
    if (p_y > 0) and (p_x > 0) and (maze[p_y - 1][p_x - 1] != 0.0):
        return p_orient, p_y - 1, p_x - 1
    return p_orient, p_y, p_x


def _northeast(maze, p_orient, p_y, p_x):
    if (p_y > 0) and (p_x < (maze.shape[1] - 1)) and (maze[p_y - 1][p_x + 1] != 0.0):
        return p_orient, p_y - 1, p_x + 1
    return p_orient, p_y, p_x


def _southwest(maze, p_orient, p_y, p_x):
    if (p_y < (maze.shape[0] - 1)) and (p_x > 0) and (maze[p_y + 1][p_x - 1] != 0.0):
        return p_orient, p_y + 1, p_x - 1
    return p_orient, p_y, p_x


def _southeast(maze, p_orient, p_y, p_x):
    if (
        (p_y < (maze.shape[0] - 1))
        and (p_x < (maze.shape[1] - 1))
        and (maze[p_y + 1][p_x + 1] != 0.0)
    ):
        return p_orient, p_y + 1, p_x + 1
    return p_orient, p_y, p_x


def _is_out_of_bounds(maze, p_y, p_x):
    return p_x < 0 or p_x >= maze.shape[1] or p_y < 0 or p_y >= maze.shape[0]


class Mechanism(abc.ABC):
    """Base class for maze transition mechanisms."""

    def __init__(self, num_actions, num_orient):
        self.num_actions = num_actions
        self.num_orient = num_orient

    @abc.abstractmethod
    def neighbors_func(self, maze, p_orient, p_y, p_x):
        """Computes next states for each action."""

    @abc.abstractmethod
    def inv_neighbors_func(self, maze, p_orient, p_y, p_x):
        """Computes previous states for each action."""

    @abc.abstractmethod
    def print_policy(self, maze, goal, policy):
        """Prints the given policy."""

    def gen_policy_field(self, maze, goal, policy):
        pass

    def plot_policy_field(self):
        pass


class DifferentialDrive(Mechanism):
    """
    In Differential Drive, the agent can move forward along its current
    orientation, or turn left/right by 90 degrees.
    """

    def __init__(self):
        super(DifferentialDrive, self).__init__(num_actions=3, num_orient=4)
        self.clockwise = [1, 3, 0, 2]  # E S N W
        self.cclockwise = [2, 0, 3, 1]  # W N S E

    def _forward(self, maze, p_orient, p_y, p_x):
        assert p_orient < self.num_orient, p_orient

        next_p_y, next_p_x = p_y, p_x
        if p_orient == 0:  # North
            next_p_y -= 1
        elif p_orient == 1:  # East
            next_p_x += 1
        elif p_orient == 2:  # West
            next_p_x -= 1
        else:  # South
            next_p_y += 1

        # If position is out of bounds, simply return the current state.
        if _is_out_of_bounds(maze, next_p_y, next_p_x) or maze[p_y][p_x] == 0.0:
            return p_orient, p_y, p_x

        return p_orient, next_p_y, next_p_x

    def _turn_right(self, p_orient, p_y, p_x):
        assert p_orient < self.num_orient, p_orient
        return self.clockwise[p_orient], p_y, p_x

    def _turn_left(self, p_orient, p_y, p_x):
        assert p_orient < self.num_orient, p_orient
        return self.cclockwise[p_orient], p_y, p_x

    def _backward(self, maze, p_orient, p_y, p_x):
        assert p_orient < self.num_orient, p_orient

        next_p_y, next_p_x = p_y, p_x
        if p_orient == 0:  # North
            next_p_y += 1
        elif p_orient == 1:  # East
            next_p_x -= 1
        elif p_orient == 2:  # West
            next_p_x += 1
        else:  # South
            next_p_y -= 1

        # If position is out of bounds, simply return the current state.
        if _is_out_of_bounds(maze, next_p_y, next_p_x) or maze[p_y][p_x] == 0.0:
            return p_orient, p_y, p_x

        return p_orient, next_p_y, next_p_x

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._forward(maze, p_orient, p_y, p_x),
            self._turn_right(p_orient, p_y, p_x),
            self._turn_left(p_orient, p_y, p_x),
        ]

    def inv_neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._backward(maze, p_orient, p_y, p_x),
            self._turn_left(p_orient, p_y, p_x),
            self._turn_right(p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        orient2str = ["↑", "→", "←", "↓"]
        action2str = ["F", "R", "L"]
        for o in range(self.num_orient):
            print(orient2str[o])
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print("\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(action2str[a], end="")
                print("")


class NorthEastWestSouth(Mechanism):
    """
    In NEWS, the agent can move North, East, West, or South.
    """

    action2str = ["↑", "→", "←", "↓"]
    action2direction = names2directions("n", "e", "w", "s")

    def __init__(self):
        super(NorthEastWestSouth, self).__init__(num_actions=4, num_orient=1)

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _north(maze, p_orient, p_y, p_x),
            _east(maze, p_orient, p_y, p_x),
            _west(maze, p_orient, p_y, p_x),
            _south(maze, p_orient, p_y, p_x),
        ]

    def inv_neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _south(maze, p_orient, p_y, p_x),
            _west(maze, p_orient, p_y, p_x),
            _east(maze, p_orient, p_y, p_x),
            _north(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print("\u2588", end="")
                        # print('.', end="")
                    else:
                        a = policy[o][y][x]
                        print(self.action2str[a], end="")
                print("\n")
            print("\n")

    def plot_policy_field(self, maze, goal, policy):
        raise NotImplementedError


class NorthEastWestSouthWrap(Mechanism):
    """
    In NEWS, the agent can move North, East, West, or South.
    """

    action2str = ["↑", "→", "←", "↓"]
    action2direction = names2directions("n", "e", "w", "s")

    def __init__(self):
        super(NorthEastWestSouthWrap, self).__init__(num_actions=4, num_orient=1)

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _north_wrap(maze, p_orient, p_y, p_x),
            _east_wrap(maze, p_orient, p_y, p_x),
            _west_wrap(maze, p_orient, p_y, p_x),
            _south_wrap(maze, p_orient, p_y, p_x),
        ]

    def inv_neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _south_wrap(maze, p_orient, p_y, p_x),
            _west_wrap(maze, p_orient, p_y, p_x),
            _east_wrap(maze, p_orient, p_y, p_x),
            _north_wrap(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print("\u2588", end="")
                        # print('.', end="")
                    else:
                        a = policy[o][y][x]
                        print(self.action2str[a], end="")
                print("\n")
            print("\n")

    def plot_policy_field(self, maze, goal, policy):
        raise NotImplementedError


class Moore(Mechanism):
    """
    In Moore, the agent can move to any of the eight cells in its Moore
    neighborhood.
    """

    action2str = ["↑", "→", "←", "↓", "↗", "↖", "↘", "↙"]

    def __init__(self):
        super(Moore, self).__init__(num_actions=8, num_orient=1)

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _north(maze, p_orient, p_y, p_x),
            _east(maze, p_orient, p_y, p_x),
            _west(maze, p_orient, p_y, p_x),
            _south(maze, p_orient, p_y, p_x),
            _northeast(maze, p_orient, p_y, p_x),
            _northwest(maze, p_orient, p_y, p_x),
            _southeast(maze, p_orient, p_y, p_x),
            _southwest(maze, p_orient, p_y, p_x),
        ]

    def inv_neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _south(maze, p_orient, p_y, p_x),
            _west(maze, p_orient, p_y, p_x),
            _east(maze, p_orient, p_y, p_x),
            _north(maze, p_orient, p_y, p_x),
            _southwest(maze, p_orient, p_y, p_x),
            _southeast(maze, p_orient, p_y, p_x),
            _northwest(maze, p_orient, p_y, p_x),
            _northeast(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print("\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(self.action2str[a], end="")
                print("")
            print("")


class FourAbsClockwise(Mechanism):
    """
    In NEWS, the agent can move North, East, West, or South.
    """

    action2str = ["↑", "→", "↓", "←"]
    action2direction = names2directions("n", "e", "s", "w")

    def __init__(self):
        super(FourAbsClockwise, self).__init__(num_actions=4, num_orient=1)

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _north(maze, p_orient, p_y, p_x),
            _east(maze, p_orient, p_y, p_x),
            _south(maze, p_orient, p_y, p_x),  # TODO swap
            _west(maze, p_orient, p_y, p_x),
        ]

    def inv_neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _south(maze, p_orient, p_y, p_x),
            _west(maze, p_orient, p_y, p_x),
            _north(maze, p_orient, p_y, p_x),  # TODO swap
            _east(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print("\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(self.action2str[a], end="")
                print("")
            print("")


class FourAbsCounterClockwise(Mechanism):
    """
    In NEWS, the agent can move North, East, West, or South.
    """

    action2str = ["↑", "←", "↓", "→"]
    action2direction = names2directions("n", "w", "s", "e")

    def __init__(self):
        super(FourAbsCounterClockwise, self).__init__(num_actions=4, num_orient=1)

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _north(maze, p_orient, p_y, p_x),
            _west(maze, p_orient, p_y, p_x),
            _south(maze, p_orient, p_y, p_x),
            _east(maze, p_orient, p_y, p_x),
        ]

    def neighbors_with_obstacle_func(self, maze, p_orient, p_y, p_x):
        return [
            _north_with_obstacle(maze, p_orient, p_y, p_x),
            _west_with_obstacle(maze, p_orient, p_y, p_x),
            _south_with_obstacle(maze, p_orient, p_y, p_x),
            _east_with_obstacle(maze, p_orient, p_y, p_x),
        ]

    def inv_neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _south(maze, p_orient, p_y, p_x),
            _east(maze, p_orient, p_y, p_x),
            _north(maze, p_orient, p_y, p_x),
            _west(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print("\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(self.action2str[a], end="")
                print("")
            print("")


class FourAbsCounterClockwiseWrap(Mechanism):
    """
    In NEWS, the agent can move North, East, West, or South.
    """

    action2str = ["↑", "←", "↓", "→"]
    action2direction = names2directions("n", "w", "s", "e")

    def __init__(self):
        super(FourAbsCounterClockwiseWrap, self).__init__(num_actions=4, num_orient=1)

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _north_wrap(maze, p_orient, p_y, p_x),
            _west_wrap(maze, p_orient, p_y, p_x),
            _south_wrap(maze, p_orient, p_y, p_x),
            _east_wrap(maze, p_orient, p_y, p_x),
        ]

    def inv_neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            _south_wrap(maze, p_orient, p_y, p_x),
            _east_wrap(maze, p_orient, p_y, p_x),
            _north_wrap(maze, p_orient, p_y, p_x),
            _west_wrap(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print("\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(self.action2str[a], end="")
                print("")
            print("")


class FourRelativeClockwise:
    """
    TODO to achieve relative action space, need to consider orientation (see differential drive)
    """

    pass


class MooreCompatibleCounterClockwise(Mechanism):
    """
    In Moore, the agent can move to any of the eight cells in its Moore
    neighborhood.
    """

    action2str = ["↑", "→", "↓", "←", "↗", "↘", "↙", "↖"]

    def __init__(self):
        super(MooreCompatibleCounterClockwise, self).__init__(
            num_actions=8, num_orient=1
        )

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            # > orbit 1
            _north(maze, p_orient, p_y, p_x),  # "↑"
            _east(maze, p_orient, p_y, p_x),  # "→"
            _south(maze, p_orient, p_y, p_x),  # "↓"
            _west(maze, p_orient, p_y, p_x),  # "←"
            # > orbit 2
            _northeast(maze, p_orient, p_y, p_x),  # "↗"
            _southeast(maze, p_orient, p_y, p_x),  # "↘"
            _southwest(maze, p_orient, p_y, p_x),  # "↙"
            _northwest(maze, p_orient, p_y, p_x),  # "↖"
        ]

    def inv_neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            # > orbit 1
            _south(maze, p_orient, p_y, p_x),  # "↓"
            _west(maze, p_orient, p_y, p_x),  # "←"
            _north(maze, p_orient, p_y, p_x),  # "↑"
            _east(maze, p_orient, p_y, p_x),  # "→"
            # > orbit 2
            _southwest(maze, p_orient, p_y, p_x),  # "↙"
            _northwest(maze, p_orient, p_y, p_x),  # "↖"
            _northeast(maze, p_orient, p_y, p_x),  # "↗"
            _southeast(maze, p_orient, p_y, p_x),  # "↘"
        ]

    def print_policy(self, maze, goal, policy):
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print("\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(self.action2str[a], end="")
                print("")
            print("")
