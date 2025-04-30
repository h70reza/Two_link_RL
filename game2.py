import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import odeint, solve_ivp
from dataclasses import dataclass


# initialize pygame
pygame.init()

@dataclass
class State:
    x: float
    y: float
    θ: float
    α: float
    p: float


#constants
WIDTH, HEIGHT = 800, 600 # screen size
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
SPEED = 90 #game speed
time_step = 0.13

#font for displaying text
FONT = pygame.font.Font(None,24)

#robot parameters
m, λ, J = 1, 50, 1/12 * 1 * 50**2 



def ODEeq(t, state, dα):
    x, y, θ, α, p = state
    #α_safe = α
    α_safe = α if abs(α) > 0.0001 else α*(-1)


    Iα = 4 * J + m * λ**2 + (m * λ**2 - 4 * J) * np.cos(α)
    dp = -(m * λ**2 * p * dα) / (Iα * np.tan(α_safe / 2))
    dx = (λ / 4) * np.cos(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α_safe)) / Iα) + dα)
    dy = (λ / 4) * np.sin(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α_safe)) / Iα) + dα)
    dθ = (2 * p * (1 - np.cos(α)) / Iα) - (dα / 2)

    return [dx, dy, dθ, dα, dp]


def compute_next_state(state: State, dα, time_val):
    state_list = [state.x, state.y,state.θ, state.α, state.p]
    t1 = time_val[-1]
    t2 = t1  + time_step

    sol = solve_ivp(ODEeq, [t1, t2], state_list, args=(dα,), method='RK23')
    x_new, y_new, θ_new, α_new, p_new = sol.y[:, -1]

    dx, dy, dθ, dα_, dp = ODEeq(t2, [x_new, y_new, θ_new, α_new, p_new], dα)

    dx2c = dx - λ/2 * np.sin(θ_new) * dθ - λ/2 * np.sin(θ_new + α_new)*(dθ + dα)
    dy2c = dy + λ/2 * np.cos(θ_new) * dθ + λ/2 * np.cos(θ_new + α_new)*(dθ + dα)

    new_state = State(x_new, y_new, θ_new, α_new, p_new)
    return new_state, (dp, dx, dy, dx2c, dy2c)


class SnakeRobotSim:
    def __init__(self):
        self.state = State(WIDTH/2, HEIGHT/2, 0, 0.8, 11160)  #(x, y, θ, α, p)
        self.old_state = State(WIDTH/2, HEIGHT/2, 0, 0.8, 11160)  #(x, y, θ, α, p)
        self.time_step = time_step
        self.time_val = [0]
        self.counter = 0
        self.history = {
            "x": [],
            "y": [],
            "θ": [],
            "α": [],
            "p": [],
            "dx": [],
            "dy": [],
            "dx2c": [],
            "dy2c": [],
            "dp": []
        }

        self._log_current()

    def _log_current(self):     #helper function for q state replacement
        for key in ["x", "y" , "θ", "α", "p"]:
            self.history[key].append(getattr(self.state, key))

    def _update_state(self, new_state: State):  # helper for entire state replacement
        self.state = new_state
        self._log_current()
        
    
    def _log_velocities(self, dp, dx, dy, dx2c, dy2c):  #helper for vel history append
        for key, val in zip(["dp", "dx", "dy", "dx2c", "dy2c"], [dp, dx, dy, dx2c, dy2c]):
            self.history[key].append(val)

    def reset(self):
        self.state = State(WIDTH/2, HEIGHT/2, 0, 0.8, 11160)
        self.time_val = [0]
        self.counter = 0
        for key in self.history:
            self.history[key] = []
        self._log_current()

            
    def step(self,dα):
        self.old_state = self.state
        #self._log_current()
        # constraint1 = (dx*np.sin(θ_2last)-dy*np.cos(θ_2last))
        # constraint2 = (dx2c*np.sin(θ_2last+α_2last)-dy2c*np.cos(θ_2last+α_2last))
        new_state, (dp, dx, dy, dx2c, dy2c) = compute_next_state(self.state, dα, self.time_val)
        self._log_velocities(dp, dx, dy, dx2c, dy2c)

        self._update_state(new_state)

        self.counter +=1
        self.time_val.append(self.time_val[-1]+self.time_step)
        return new_state



class SnakeRobotGame:
    def __init__(self):
        self.sim = SnakeRobotSim()
        self.reset()
        self.display = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("Two-Link Snake Robot")
        self.clock = pygame.time.Clock()

        self.constraint1, self.constraint2  = [], []
        self.time_span = np.arange(0,500,0.13)
        self.time_val = [0]
        self.counter = 0

        # plt.ion()
        # self.fig, self.axs = plt.subplots(2,1)

    def reset(self): # only rests the target and a new input

        while True:
            target_x = random.randint(10, WIDTH-10)
            target_y = random.randint(10, HEIGHT-10)
            if np.linalg.norm(np.array([target_x, target_y])-np.array([self.sim.state.x, self.sim.state.y])) > 5:
                break

        self.target = (target_x, target_y)
        self.dα = 0.3* np.cos(0)

    def play_step(self):


        for event in pygame.event.get():
            if event.type == pygame.QUIT:      # when quiting by user
                pygame.quit()
                for key in self.sim.history:
                    self.sim.history[key] = np.array(self.sim.history[key])
                    if key in ["x", "y", "θ", "α", "p"]:
                        self.sim.history[key] = self.sim.history[key][:-1]   #adjusting the length of configuration variables to be similar to the valocity

                self.time_val.pop()
                plt.figure()
                plt.plot(self.sim.history["x"], self.sim.history["y"],label = 'x-y')
                # plt.plot(self.sim.time_val[:-1], self.sim.history["p"]* 1e7,label = 'p-t')
                # plt.plot(self.sim.time_val[:-1], self.sim.history["α"]* 5e12,label = 'α-t')
                # plt.plot(self.sim.time_val[:-1], self.sim.history["dp"]* 1e5,label = 'dp-t')
                # plt.plot(self.sim.time_val[:-1], self.sim.history["dx"]* 1e11,label = 'dx-t')
                #plt.plot(self.sim.time_val[:-1], (self.sim.history["x"]-WIDTH/2)* 1e11,label = 'x-t')
                # plt.plot(self.sim.time_val[:-1], self.constraint1, label='Constraint 1')
                # plt.plot(self.sim.time_val[:-1], self.constraint2, label='Constraint 2')
                plt.xlabel("Time")
                plt.ylabel("Constraint Value")
                plt.legend()
                plt.title("Nonholonomic Constraints Over Time")
                plt.grid(True)
                plt.show()
                plt.grid()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.dα += 0.1
                elif event.key == pygame.K_DOWN:
                    self.dα -= 0.1


        self.dα =  0.3* np.cos(self.time_val[-1])

        self.sim.step(self.dα)

        dx = self.sim.history["dx"][-1]
        dy = self.sim.history["dy"][-1]
        dx2c = self.sim.history["dx2c"][-1]
        dy2c = self.sim.history["dy2c"][-1]

        θ_2last = self.sim.history["θ"][-1]  
        α_2last = self.sim.history["α"][-1]
        test = self.sim.history["dx"]
        #print(f"Time = {self.time_val[-1]}, dx: {test[-1]}")
        self.constraint1.append(dx*np.sin(θ_2last)-dy*np.cos(θ_2last))
        self.constraint2.append(dx2c*np.sin(θ_2last+α_2last)-dy2c*np.cos(θ_2last+α_2last))


        if np.linalg.norm(np.array([self.sim.history["x"][-1],self.sim.history["y"][-1]])-np.array(self.target)) < 1:
            while True:
                target_x = random.randint(10, WIDTH-10)
                target_y = random.randint(10, HEIGHT-10)
                if np.linalg.norm(np.array([target_x, target_y])-np.array([self.sim.history["x"][-1],self.sim.history["y"][-1]])) > 5:
                    break

            self.target = (target_x, target_y)

        #update ui
        self._update_ui()
        #self._update_plots()
        self.clock.tick(SPEED)
        
        self.counter += 1
        self.time_val.append(self.time_span[self.counter])


    
    Actions = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]

    def _update_ui(self, action_idx):
        self.display.fill(BLACK)
        pygame.draw.circle(self.display, RED, self.target, 5)

        state = self.sim.history
        #print(f"Time = {self.time_val[-1]}, x: {state["x"][-1]}, θ: {state["θ"][-1]}, α: {state["α"][-1]}")
        x0 = state["x"][-1] - λ/2 * np.cos(state["θ"][-1])
        y0 = state["y"][-1] - λ/2 * np.sin(state["θ"][-1])
        x1 = state["x"][-1] + λ/2 * np.cos(state["θ"][-1])
        y1 = state["y"][-1] + λ/2 * np.sin(state["θ"][-1])
        x2 = x1 + λ * np.cos(state["θ"][-1] + state["α"][-1])
        y2 = y1 + λ * np.sin(state["θ"][-1] + state["α"][-1])

        pygame.draw.line(self.display, BLUE, (x0,y0), (x1,y1), 5)
        pygame.draw.line(self.display, BLUE, (x1,y1), (x2,y2), 5)
        dα = Actions[action_idx]
        text_surface = FONT.render(f"dα: {dα: .2f}", True, WHITE)
        self.display.blit(text_surface,(10,10))

        pygame.display.flip()

    def _update_plots(self):
        #print(f"Time = {self.time_val[-1]}, dx: {self.sim.history["dx"][-1]}, dy: {self.sim.history["dy"][-1]}")
        self.axs[0].cla()
        self.axs[0].plot(self.sim.time_val,  self.sim.history["p"], label='p[t]')
        self.axs[0].legend()
        
        self.axs[1].cla()
        self.axs[1].plot(self.sim.time_val, self.constraint1, label='dx*sin(θ) - dy*cos(θ)')
        self.axs[1].legend()

        # self.axs[2].cla()
        # self.axs[2].plot(self.time_val, self.constraint2, label='dx2*sin(θ+α) - dy2*cos(θ+α)')
        # self.axs[2].legend()

        self.axs[2].cla()
        self.axs[2].plot(self.x_val, self.y_val, label='x-y')
        self.axs[2].legend()
        
        plt.pause(0.01)

Actions = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]

class SnakeRobotEnv:
    def __init__(self):
        self.sim = SnakeRobotSim()
        self.game = SnakeRobotGame()
        self.max_steps = 300
        self.counter = 0

    def reset(self):
        self.sim.reset()
        self._reset_target()
        self.counter = 0
        return self._get_state()

    def step(self, action_idx): # based on an idx it will apply a sim.step
        dα = Actions[action_idx]
        self.sim.step(dα)
        self.counter +=1

        next_state = self._get_state()
        reward = self._compute_reward()
        #if reward == -40:
        #    s = self.sim.old_state
        #    target_x, target_y = self.game.target 
        #    return np.array([s.x, s.y, s.θ, s.α, s.p, target_x, target_y], dtype = np.float32), reward, False, {}
        done = self._check_done()

        return next_state, reward, done, {}

    def _get_state(self):
        s = self.sim.state
        target_x, target_y = self.game.target
        return np.array([s.x, s.y, s.θ, s.α, s.p, target_x, target_y], dtype = np.float32)
    
    def _reset_target(self):
        while True:
            target_x = np.random.randint(10, WIDTH-10)
            target_y = np.random.randint(10, HEIGHT-10)
            if np.linalg.norm([target_x-self.sim.state.x, target_y-self.sim.state.y])>5:
                break
        self.game.target = (target_x, target_y)

    def _compute_reward(self):
        s = self.sim.state
        target = self.game.target
        """
        if abs(s.α) > 2.5 :
            return -40
        
        dist = np.linalg.norm([s.x-target[0], s.y-target[1]])

        if dist<10:
            return 100
        
        return -1

        """
        dist = np.linalg.norm([s.x-target[0], s.y-target[1]])
        reward = -1e-5*dist

        if dist<10:
            reward += 40

        if abs(abs(s.α)-np.pi) < 0.2:
            reward -= 5*np.exp(1*abs(s.α))

        return reward
        
    
    def _check_done(self):
        s = self.sim.state
        target_x, target_y = self.game.target
        if self.counter > self.max_steps:
            return True
        dist = np.linalg.norm([s.x - target_x, s.y - target_y])
        return dist < 5
    
# env = SnakeRobotEnv()
# obs = env.reset()

# done = False
# while not done:
#     action = np.random.randint(0, len(Actions))
#     obs, reward, done, _ = env.step(action)
#     print(f"step {env.counter}: Reward = {reward: .2f}, state = {obs}")

# if __name__=="__main__":
#     game = SnakeRobotGame()
#     while True:
#         game.play_step()

