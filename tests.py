from Evoriment import BanditEnv, Strategy, GreedyRandomChoiceStrategy, EpsilonGreedyStrategy, SoftMaxStrategy, UCBStrategy
import numpy as np

def test_strategy(env: BanditEnv, strategy: Strategy, n_episodes: int=100):
    total_reward = 0.0
    actions = [-1 for _ in range(n_episodes)]
    rewards = [0 for _ in range(n_episodes)]

    for episode in range(n_episodes):
        action = strategy.make_action()
        actions[episode] = action
        if action == None:
            break

        reward = env.step(action)
        rewards[episode] = reward
        strategy.update(action, reward)
        total_reward += reward
    
    return actions, rewards, total_reward

def run_n_times(env: BanditEnv, strategy: Strategy, n_episodes: int,\
                n_epoches: int):
    epoch_actions = [0.0] * n_episodes
    counter = [0.0] * n_episodes
    epoches_rewards = [0.0] * n_episodes
    sum_total_reward = 0.0

    for epoch in range(n_epoches):
        strategy.reset()
        actions, rewards, total_reward = test_strategy(env, strategy, n_episodes)

        sum_total_reward += total_reward

        for episode in range(n_episodes):
            if actions[episode] == None:
                break
            counter[episode] += 1
            epoch_actions[episode] += actions[episode]
            epoches_rewards[episode] += rewards[episode]
        
    for episode in range(n_episodes):
        epoch_actions[episode] /= counter[episode]
        epoches_rewards[episode] /= counter[episode]
    sum_total_reward /= n_epoches

    return epoch_actions, epoches_rewards, sum_total_reward


n_episodes = 100
n_epoches = 100

costs = [5, 7, 8, 5, 1]
money = 100
env = BanditEnv([0.3, 0.2, 0.1, 0.2, 0.9], [10, 2, 3, 4, 1], costs)
# strategy = GreedyRandomChoiceStrategy(n_arms=env.n_arms, n_explores=0, costs=costs, money=money)
# strategy = EpsilonGreedyStrategy(n_arms=env.n_arms, epsilon=0.00, costs=costs, money=money)
# strategy = SoftMaxStrategy(n_arms=env.n_arms, alpha=0.05, costs=costs, money=money)
strategy = UCBStrategy(n_arms=env.n_arms, epsilon=0.05, costs=costs, money=money)

actions, rewards, total_reward = run_n_times(env, strategy, n_episodes, n_epoches)

print(strategy)
print(actions)
print(total_reward)

