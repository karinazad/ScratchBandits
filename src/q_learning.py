import numpy as np


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float)
      discount - (float)
      adaptive - (bool)
    """

    def __init__(self, epsilon=0.2, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive
        self.current_steps = 0

    def fit(self, env, steps=1000):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.
        Arguments:
          env - (Env)
          steps - (int)

        Returns:
          state_action_values - (np.array)
          rewards - (np.array)
        """
        action_count = env.action_space.n
        states_count = env.observation_space.n

        Q = np.zeros((states_count, action_count))
        n = np.zeros(action_count)

        s = np.floor(steps / 100)
        step = 0
        rewards = []
        avg_rewards = []


        for i in range(steps):
            R = 0
            state = env.reset()
            done = False

            while not done:

                if np.random.random() < self.epsilon:
                    action = np.random.randint(action_count)
                else:
                    action = np.argmax(Q[state, :])

                next_state, next_reward, done, info = env.step(action)

                n[action] += 1
                Q[state, action] += (1 / n[action]) * \
                                    (next_reward + self.discount * np.max(Q[next_state,:]) - Q[state, action])
                state = next_state
                R += next_reward

            rewards.append(R)
            if step % s == 0:
                avg = np.mean(rewards)
                avg_rewards.append(avg)
                rewards = []

            step += 1
            self.epsilon = self._get_epsilon(step / steps)

        return Q, np.array(avg_rewards)



    def predict(self, env, state_action_values):
        """
        Arguments:
          env - (Env)
          state_action_values - (np.array) T

        Returns:
          states - (np.array)
          actions - (np.array)
          rewards - (np.array)
        """

        states = []
        actions = []
        rewards = []

        state = env.reset()

        done = False
        while not done:
            action = np.argmax(state_action_values[state, :])
            state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

        env.close()

        return np.array(states), np.array(actions), np.array(rewards)




    def _get_epsilon(self, progress):
        """
        Arguments:
            progress - (float) in [0,1]
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        Arguments:
            progress - (float) in [0,1]
        """
        epsilon = (1 - progress) * self.epsilon
        return epsilon
