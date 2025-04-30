import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import odeint, solve_ivp

#constants
time_step = 0.1
global counter1
counter1 = 0
time_span = np.arange(0,400,0.1)
time_val = [0]
state_vars_old = [400, 300, 1, 0.025, 80] # Initial (x, y, θ, α, p)
x_val, y_val, θ_val, α_val, p_val, dp_val, dx_val, dy_val, dx2c_val, dy2c_val, = [400],[300],[1],[0.025],[80],[],[],[],[],[]


#robot parameters
m, λ, J = 1, 50, 1/12 * 1 * 50**2 

def ODEeq(state, dα):
    x, y, θ, α, p = state
    dα = dα
    Iα = 4 * J + m * λ**2 + (m * λ**2 - 4 * J) * np.cos(α)
    dp = -(m * λ**2 * p * dα) / (Iα * np.tan(α / 2))
    dx = (λ / 4) * np.cos(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α)) / Iα) + dα)
    dy = (λ / 4) * np.sin(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α)) / Iα) + dα)
    dθ = (2 * p * (1 - np.cos(α)) / Iα) - (dα / 2)

    return [dx, dy, dθ, dα, dp]


def compute_next_state(state, dα, timeVal):
    x, y, θ, α, p = state
    #dα = dα
    t1 = timeVal
    t2 = timeVal  + time_step
    
    sol1 = solve_ivp(lambda t, y:ODEeq(y, dα), [t1,t2], np.array(state), method = 'LSODA')
    x_new, y_new, θ_new, α_new, p_new = sol1.y[:, -1]

    Iα = 4 * J + m * λ**2 + (m * λ**2 - 4 * J) * np.cos(α)
    dp = -(m * λ**2 * p * dα) / (Iα * np.tan(α / 2))
    dx = (λ / 4) * np.cos(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α)) / Iα) + dα)
    dy = (λ / 4) * np.sin(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α)) / Iα) + dα)
    dθ = (2 * p * (1 - np.cos(α)) / Iα) - (dα / 2)
    dx2c = dx - λ/2 * np.sin(θ) * dθ - λ/2 * np.sin(θ + α)*(dθ+dα)
    dy2c = dy + λ/2 * np.cos(θ) * dθ + λ/2 * np.cos(θ + α)*(dθ+dα)

    
    return (x_new, y_new, θ_new, α_new, p_new), (dp, dx, dy, dx2c, dy2c)


for counter1 in range(len(time_span)-1):
    dα = -2* np.sin(time_span[counter1])
    state_vars_new, extra_vars = compute_next_state(state_vars_old, dα, time_val[-1])
    x, y, θ, α, p = state_vars_new
    dp, dx, dy, dx2c, dy2c = extra_vars
    #print(f"t: {time_val[-1]}, θ: {θ_val[-1]}, α: {α_val[-1]}, x: {x_val[-1]}, y: {y_val[-1]}, p: {p_val[-1]}")
    x_val.append(x)
    y_val.append(y)
    θ_val.append(θ) 
    α_val.append(α) 
    p_val.append(p) 
    dx_val.append(dx)
    dy_val.append(dy)
    dx2c_val.append(dx2c)
    dy2c_val.append(dy2c)
    #print(f"dx: {dx_val[-1]}, dy: {dy_val[-1]}, dx2c: {dx2c_val[-1]}, dy2c: {dy2c_val[-1]}")
    state_vars_old = state_vars_new
    time_val.append(time_span[counter1 + 1])
    


x_val = np.array(x_val)
y_val = np.array(y_val)
θ_val = np.array(θ_val)
α_val = np.array(α_val)
p_val = np.array(p_val)
dx_val = np.array(dx_val)
dy_val = np.array(dy_val)
dx2c_val = np.array(dx2c_val)
dy2c_val = np.array(dy2c_val)
x_val = x_val[:-1]
y_val = y_val[:-1]
θ_val = θ_val[:-1]
α_val = α_val[:-1]
p_val = p_val[:-1]
time_val.pop()
#print(f"Time = {time_val}, θ: {θ_val}, α: {α_val}, x: {x_val}, y: {y_val} dx: {dx_val}, dy: {dy_val}, dx2c: {dx2c_val}, dy2c: {dy2c_val}, p: {p_val}")
               
# plt.plot(time_val,dx_val*np.sin(θ_val)-dy_val*np.cos(θ_val),label = 'const1')
# plt.plot(time_val,dx2c_val*np.sin(θ_val+α_val)-dy2c_val*np.cos(θ_val+α_val),label = 'const2') 
plt.plot(x_val,y_val, label = 'x-y')
plt.legend()
plt.show()
