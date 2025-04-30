import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


m, λ, J = 50, 50, 50
t_initial, t_final = 0,5
dα_dθ_d = []

# Initial conditions
α0_values = np.linspace(0.05, 0.1, 6)  # Varying α0
p0_values = np.linspace(-500, 500, 100)  # Varying p0

def dynamic(t, state):
    α, p = state
    dα = np.cos(t) 
    Iα = 4 * J + m * λ**2 + (-4 * J + m * λ**2) * np.cos(α)
    dp = (m * p * λ**2 * dα) / (Iα * np.tan(α / 2) + 1e-6)
    return [dα, dp]


""" def reconstruct(t, state, p_values,α_values):
    x, y, θ = state
    dα = np.cos(t) 
    p = np.interp(t, p_valuesf[:, 0], p_values[:, 1])  # Interpolating p(t)
    α = np.interp(t, α_valuesf[:, 0], α_values[:, 1])  # Interpolating α(t)
    Iα = 4 * J + m * λ**2 + (-4 * J + m * λ**2) * np.cos(α)

    dx = (λ / 4) * np.cos(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α)) / Iα) + dα)
    dy = (λ / 4) * np.sin(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α)) / Iα) + dα)
    dθ = (2 * p * (1 - np.cos(α)) / Iα) - (dα / 2)

    return [dx, dy, dθ] """


# calculate store α and p as functions that can be used in solving reconstruction
for α0 in α0_values:
    for p0 in p0_values:
        dα_dθ = []
        sol1 = solve_ivp(dynamic, [t_initial, t_final],[α0,p0], method='RK45', max_step=0.01)

        # check divergance
        if np.any(np.abs(sol1.y[1])>1e3):
            dα_dθ_d.append(1)
            continue
        α_values = sol1.y[0]
        p_values = sol1.y[1]
        dα_values = np.cos(sol1.t)
        Iα_values = 4 * J + m * λ**2 + (-4 * J + m * λ**2) * np.cos(α_values)
        dθ_values = (2 * p_values * (1 - np.cos(α_values)) / Iα_values) - (dα_values / 2)
        dα_dθ = (dα_values/dθ_values)        

        plt.plot(sol1.t,dα_dθ)

# Add legend and labels
plt.xlabel("Time (s)")
plt.ylabel("dα/dθ")
plt.title("Evolution of dα/dθ Over Time for Different Initial Conditions")
plt.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1))  # Position legend outside
plt.grid()

# Display the length of dα_dθ_d (number of divergent cases)
plt.text(0.5, 0.5, f"Divergent Cases: {len(dα_dθ_d)}", transform=plt.gca().transAxes, fontsize=12, color='red')

# Show the plot
plt.show()

    
