import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from game2 import SnakeRobotEnv
from game2 import SnakeRobotGame
from plot import plot
#from helper import plot


MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.99
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet(7, 256, 5)     #asks for the inputs in the init
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self, state):
        return np.array(state, dtype= np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80-self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        return move
    
def train():
    env = SnakeRobotEnv()
    game = SnakeRobotGame()
    agent_inst = agent()
    scores = []
    mean_scores = []
    total_score = 0

    record = - float('inf')
    for episode in range(2000):
        state_old = env.reset()
        done = False
        score = 0

        while not done:
            action = agent_inst.get_action(state_old)
            state_new, reward, done, _ = env.step(action)

            #print(state_new)

            # game.sim.state = env. sim. state
            # game.target = env.game.target
            # game._update_ui(action)
            # game.clock.tick(60)
            pass

            agent_inst.train_short_memory(state_old, action, reward, state_new, done)
            agent_inst.remember(state_old, action, reward, state_new, done)

            state_old = state_new
            score += reward

        agent_inst.n_games += 1
        agent_inst.train_long_memory()

        if score > record:
            record = score
            agent_inst.model.save()
            #print(f"New record! episode: {agent_inst.n_games}, Score: {score:.2f}")
        #else:
            #print(f"Episode: {agent_inst.n_games}, Score: {score:.2f}, Record: {record:.2f}")

        scores.append(score)
        np.save('scores.npy', scores)
        total_score += score
        mean_score = total_score/ agent_inst.n_games
        mean_scores = total_score/ agent_inst.n_games
        #plot(scores, [np.mean(scores[:i+1]) for i in range(len(scores))])
        plot(scores, mean_scores)
if __name__ == '__main__':
    train()

