import numpy as np
import gym


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000):
        """
        Trains the MultiArmedBandit on an OpenAI Gym environment.
        """

        action_count = env.action_space.n
        states_count = env.observation_space.n
        s = np.floor(steps / 100)
        rewards = []
        avg_rewards = []

        Q = np.zeros((states_count, action_count))
        n = np.zeros(action_count)  # to keep count

        state = env.reset()

        for i_episode in range(steps):
            prev_state = state
            if np.random.random() < self.epsilon:
                action = np.random.randint(action_count)
            else:
                action = np.argmax(Q[state,:])

            state, reward, done, info = env.step(action)

            n[action] += 1
            Q[state, action] += (reward - Q[prev_state, action]) / n[action]
            rewards.append(reward)

            if i_episode % s == 0:
                avg = np.mean(rewards)
                avg_rewards.append(avg)
                rewards = []

        env.close()

        return Q, np.array(avg_rewards)

    def predict(self, env, state_action_values):
        """
        Arguments:
          env - (Env)
          state_action_values - (np.array)

        Returns:
          states - (np.array)
          actions - (np.array)
        """

        states = []
        actions = []
        rewards = []

        state = env.reset()

        done = False
        while not done:
            action = np.argmax(state_action_values[state,:])
            state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

        env.close()

        return np.array(states),np.array(actions), np.array(rewards)