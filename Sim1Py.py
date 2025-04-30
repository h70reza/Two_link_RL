import numpy as np
from scipy.integrate import solve_ivp

#Define the Constants
material_density = 1
plate_length = 15
plate_width = 15
plate_thickness = 1/900000
plate_mass = material_density * plate_length * plate_width * plate_thickness

link_length = 1
link_thickness = (1/12) * link_length
link_mass = material_density * link_length * link_thickness**2
link_rotational_inertia = (1/12) * link_mass * (link_length**2 + link_thickness**2)

wheel_radius = (1/8) * link_length
wheel_thickness = (2/3) * link_thickness
wheel_mass = material_density * np.pi * wheel_radius**2 * wheel_thickness
wheel_rolling_rotational_inertia = (1/2) * wheel_mass * wheel_radius**2
wheel_pivoting_rotational_inertia = (1/4) * wheel_mass * wheel_radius**2

hinge_radius = (5/6) * link_thickness
hinge_height = (3/2) * link_thickness

params = {
    "M" : plate_mass,
    "m" : link_mass + wheel_mass,
    "J" : link_rotational_inertia + wheel_pivoting_rotational_inertia,
    "R" : wheel_radius,
    "Jw" : wheel_rolling_rotational_inertia,
    "L" : link_length,
    "K" : 1/1100
}

# Define positions
def compute_positions(state):
    X, Y, x, y, θ, α = state[:6]
    plate_position = np.array([X,Y])
    first_link_position = np.array([X+x, Y+y])
    second_link_position = first_link_position + np.array([
        (params["L"]/2) * np.cos(θ),(params["L"]/2) * np.sin(θ)
    ]) + np.array([
        (params["L"]/2) * np.cos(θ+α),(params["L"]/2) * np.sin(θ+α)
    ])
    return plate_position, first_link_position, second_link_position

# Define velocities and constraints
def compute_constraints(state,derivatives):
    dX, dY, dx, dy, dθ, dα, = derivatives[6:]

    plate_velocity = np.array([dX, dY])
    first_link_velocity = np.array([dX + dx, dY + dy])
    second_link_velocity = first_link_velocity + np.array([
        (params["L"] / 2) * dθ * -np.sin(θ), (params["L"] / 2) * dθ * np.cos(θ)
    ]) + np.array([
        (params["L"] / 2) * (dθ + dα) * -np.sin(θ + α), (params["L"] / 2) * (dθ + dα) * np.cos(θ + α)
    ])

    first_link_constraint = np.dot((first_link_velocity - plate_velocity),[-np.sin(θ), np.cos(θ)])
    second_link_constraint = np.dot((second_link_velocity - plate_velocity),[-np.sin(θ +α), np.cos(θ + α)])

    return first_link_constraint, second_link_constraint


# Lagrangian of the system
def lagrangian(state, params):
    X, Y, x, y, θ, a, dX, dY, dx, dy, dθ, dα = state
    M, m, J, R, Jw, L, K = params["M"], params["m"], params["J"], params["R"], params["Jw"], params["L"], params["K"]

    # compute kinetic energy
    KE = (1/2) * M * (dX**2 + dY**2) + (1/2) * m * (dx**2 + dy**2) \
         + (1/2) * m * ((dx + (L/2) * dθ)**2 + (dy + (L/2) * dθ)**2) \
         + (1/2) * J * dθ**2 + (1/2) * J * (dθ + dα)**2 \
         + (1/2) * Jw/R**2 * ((dx - dX)**2 + (dy - dY)**2) 
