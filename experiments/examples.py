import gym
from src import MultiArmedBandit
from src import QLearning
import numpy as np
import matplotlib.pyplot as plt


def run_FrozenLake():
    env = gym.make('FrozenLake-v0')
    d = {} #dictionary for averaged rewards for each epsilon

    # EXAMPLE
    for eps in [0.01,0.5]:
        averaged_rewards = []
        for _ in range(10):
            agent = QLearning(eps)
            action_values, rewards = agent.fit(env,steps=50000)
            averaged_rewards.append(rewards)
        averaged_rewards = np.array(averaged_rewards)
        np.reshape(averaged_rewards,newshape=(rewards.shape[0],10))
        averaged_rewards = np.mean(averaged_rewards,axis=0)
        d[eps] = averaged_rewards

    one = d[0.01]
    two = d[0.5]

    init_eps = 0.5
    averaged_rewards = []
    for _ in range(10):
        agent = QLearning(init_eps,adaptive=True)
        action_values, rewards = agent.fit(env, steps=50000)
        averaged_rewards.append(rewards)

    averaged_rewards = np.array(averaged_rewards)
    np.reshape(averaged_rewards, newshape=(rewards.shape[0], 10))
    averaged_rewards = np.mean(averaged_rewards, axis=0)


    plt.figure()
    plt.xlabel("step buckets")
    plt.ylabel(" average reward")
    plt.plot(range(len(one)), one, '-g',label="e = 0.01")
    plt.plot(range(len(two)), two, '-r',label="e = 0.5")
    plt.plot(range(len(averaged_rewards)),averaged_rewards, label="adaptive learning")
    plt.legend()
    plt.show()

    print('Finished example experiment')


# EXAMPLE 2
def run_SlotMachines():
    reward_first = []
    reward_five = []
    reward_ten = []
    for i in range(10):
        env = gym.make('SlotMachines-v0')
        bandit = MultiArmedBandit()
        action_values, rewards = bandit.fit(env, 10000)

        if i == 0:
            reward_first = rewards

        if i < 5:
            reward_five.append(rewards)

        reward_ten.append(rewards)

    reward_first = np.array(reward_first)
    reward_five = np.mean(np.array(reward_five), axis=0)
    reward_ten = np.mean(np.array(reward_ten), axis = 0)

    plt.plot(np.arange(100), reward_first, label="1 trial")
    plt.plot(np.arange(100), reward_five, label="5 trial")
    plt.plot(np.arange(100), reward_ten, label="10 trial")
    plt.legend()
    plt.ylim(0,8)
    plt.xlabel('step bucket')
    plt.ylabel('reward')
    plt.show()
