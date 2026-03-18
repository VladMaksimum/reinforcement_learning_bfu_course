import numpy as np
from abc import ABC, abstractmethod

class BanditEnv():
    def __init__(self, probs: list[float], reward: list[int], costs: list[int]):
        self.probs = probs
        self.reward = reward
        self.costs = costs

    def step(self, action: int) -> int:
        return self.reward[action] - self.costs[action] if np.random.rand(1).item() < self.probs[action] else -self.costs[action]

    @property
    def n_arms(self):
        return len(self.probs)

class Strategy(ABC):
    def __init__(self, n_arms, costs, money):
        self.Q = np.zeros(n_arms)
        self.n = [0 for _ in range(n_arms)]
        self.n_arms = n_arms
        self.costs = costs
        self.money = money
        self.start_money = money

    @abstractmethod
    def make_action(self):
        ...

    def update(self, action, reward):
        self.n[action] += 1
        self.Q[action] = (self.Q[action] * (self.n[action]-1) + reward)/self.n[action]
        self.money += reward

    @abstractmethod
    def reset(self):
        ...

class GreedyRandomChoiceStrategy(Strategy):

    def __init__(self, n_arms: int, n_explores: int, costs: list[int], money: int):
        super().__init__(n_arms, costs, money)
        self.n_explores = n_explores

    def make_action(self):
        is_possible = False
        for c in self.costs:
            if c <= self.money:
                is_possible = True
                break
        
        if not is_possible:
            return None

        if sum(self.n) < self.n_explores:
            a = int(np.random.randint(self.n_arms))

            while self.costs[a] > self.money:
                a = int(np.random.randint(self.n_arms))

        else:
            a = None
            for i in range(len(self.Q)):
                if self.costs[i] <= self.money and a == None:
                    a = i
                elif self.costs[i] <= self.money and self.Q[i] > self.Q[a]:
                    a = i
            
        return a

    def __str__(self) -> str:
        return f'GreedyRandomChoiceStrategy: n_explores={self.n_explores}'

    def reset(self):
        self.__init__(self.n_arms, self.n_explores, self.costs, self.start_money)
    

class EpsilonGreedyStrategy(Strategy):

    def __init__(self, n_arms: int, epsilon: float, costs: list[int], money: int):
        super().__init__(n_arms, costs, money)

        self.epsilon = epsilon

    def make_action(self):
        is_possible = False
        for c in self.costs:
            if c <= self.money:
                is_possible = True
                break
        
        if not is_possible:
            return None
        
        if np.random.rand(1) > self.epsilon:
            a = None
            for i in range(len(self.Q)):
                if self.costs[i] <= self.money and a == None:
                    a = i
                elif self.costs[i] <= self.money and self.Q[i] > self.Q[a]:
                    a = i
        else:
            a = int(np.random.randint(self.n_arms))

            while self.costs[a] > self.money:
                a = int(np.random.randint(self.n_arms))
    
        return a
    
    def __str__(self) -> str:
        return f'EpsilonGreedyStrategy: epsilon={self.epsilon}'

    def reset(self):
        self.__init__(self.n_arms, self.epsilon, self.costs, self.start_money)

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))

class SoftMaxStrategy(Strategy):
    def __init__(self, n_arms: int, alpha: float, costs: list[int], money: int):
        super().__init__(n_arms, costs, money)

        self.alpha = alpha
    
    def make_action(self):
        is_possible = False
        for c in self.costs:
            if c <= self.money:
                is_possible = True
                break
        
        if not is_possible:
            return None
        
        probs = softmax(self.Q / self.alpha)
        a = np.random.choice(self.n_arms, p=probs)

        while self.costs[a] > self.money:
            a = np.random.choice(self.n_arms, p=probs)

        return a
    
    def __str__(self) -> str:
      return f'SoftMaxStrategy: alpha={self.alpha}'

    def reset(self):
        self.__init__(self.n_arms, self.alpha, self.costs, self.start_money)

class UCBStrategy(Strategy):
    def __init__(self, n_arms: int, epsilon: float, costs: list[int], money: int):
        super().__init__(n_arms, costs, money)
        self.epsilon = epsilon

    def make_action(self):
        is_possible = False
        for cost in self.costs:
            if cost <= self.money:
                is_possible = True
                break
        
        if not is_possible:
            return None

        c = np.sqrt(self.epsilon *
                    np.log(np.sum(self.n) + 1) /
                    (np.array(self.n) + 1))
        upper_bounds = self.Q + c

        a = None
        for i in range(len(upper_bounds)):
            if self.costs[i] <= self.money and a == None:
                a = i
            elif self.costs[i] <= self.money and upper_bounds[i] > upper_bounds[a]:
                a = i
        
        return a

    def __str__(self) -> str:
        return f'UCBStrategy: epsilon={self.epsilon}'

    def reset(self):
        self.__init__(self.n_arms, self.epsilon, self.costs, self.start_money)

