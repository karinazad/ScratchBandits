import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class SlotMachine:
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

    def pull(self):
        return np.random.normal(self.mean, self.std_dev)


class SlotMachines(gym.Env):
    """
    Slot Machine environment.

    Arguments:
        n_machines - (int)
        mean_range - (tuple)
        std_range - (tuple)
    """

    def __init__(self, n_machines=10, mean_range=(-10, 10), std_range=(5, 10)):
        means = np.random.uniform(mean_range[0], mean_range[1], n_machines)
        std_devs = np.random.uniform(std_range[0], std_range[1], n_machines)
        self.machines = [SlotMachine(m, s) for (m, s) in zip(means, std_devs)]

        self.action_space = spaces.Discrete(n_machines)
        self.observation_space = spaces.Discrete(1)

    def seed(self, seed=None):
        """
        Arguments:
          seed - (int)
        """
        _, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Arguments:
          action - (int)

        Returns:
          observation - (int)
          reward - (float) T
          done - (bool)
          info - (dict)
        """
        assert self.action_space.contains(action)
        return 0, self.machines[action].pull(), True, {}
