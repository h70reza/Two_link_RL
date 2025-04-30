from game2 import SnakeRobotGame, SnakeRobotEnv
from model import Linear_QNet
import torch
import numpy as np


model = Linear_QNet()
model.load()

env = SnakeRobotEnv()
game = SnakeRobotGame()

obs = env.reset()
done = False


while not done:
    state = torch.tensor(obs, dtype = torch.float32)
    with torch.no_grad():
        action = torch.argmax(model(state)).item()
    obs, _, done, _ = env.step(action)

    game.sim.state = env.sim.state
    game.target = env.game.target
    game._update_ui(action)
    game.clock.tick(30)

