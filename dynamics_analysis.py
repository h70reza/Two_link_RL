import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sympy as sm

#define the state variables

t = sm.symbols("t")
m, J , λ = sm.symbols("m J λ")
x, y, θ, α, p = [sm.Function(s)(t) for s in ["x", "y", "θ", "α", "p"]]
dx, dy, dθ, dα, dp = [sm.diff(i,t) for i in [x, y, θ, α, p]]
ddx, ddy, ddθ, ddα, ddp = [sm.diff(k,t) for k in [dx, dy, dθ, dα, dp]]
q = sm.Matrix([x, y, θ, α, p])
dq = sm.Matrix([dx, dy, dθ, dα, dp])
ddq = sm.Matrix([ddx, ddy, ddθ, ddα, ddp])

# Trying to find equilibrium points for the MECC system

Iα = 4 * J + m * λ**2 + (m * λ**2 - 4 * J) * sm.cos(α)
eq_dp = -(m * λ**2 * p * dα) / (Iα * sm.tan(α / 2))
eq_dx = (λ / 4) * sm.cos(θ) * sm.tan(α / 2) * ((4 * p * (1 + sm.cos(α)) / Iα) + dα)
eq_dy = (λ / 4) * sm.sin(θ) * sm.tan(α / 2) * ((4 * p * (1 + sm.cos(α)) / Iα) + dα)
eq_dθ = (2 * p * (1 - sm.cos(α)) / Iα) - (dα / 2)
subs = {dx: 0, dy:0, dθ: 0, dα: 0, dp: 0}
sol1 = sm.solve([eq_dx.subs(subs), eq_dy.subs(subs), eq_dθ.subs(subs), eq_dp.subs(subs)],[x, y, θ, α, p],dict = True)

for i, sol in enumerate(sol1):
    print(f"solution {i+1}:")
    print(sol)

# checking the stability at the equilibrium point
dz = sm.Matrix([eq_dx, eq_dy, eq_dθ, dα, eq_dp])
eq_point = {α: 0, p: 0}
jz = dz.jacobian(q)
jz_eq = jz.subs(eq_point)
print(jz_eq)