import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Parameters
L = 4 * np.pi
T = 5  # 5 periods
Nx = 1000  # Number of spatial points
Nt = 100000  # Number of time steps
u = -1  # Advection velocity u

# Create a non-uniform grid
x = np.linspace(0, L, Nx)
dx = np.diff(x)
dx = np.append(dx, dx[-1])  # Extend the last dx for boundary conditions

# Ensure CFL condition is met
dt = T / Nt
CFL = np.abs(u) * dt / np.min(dx)  # Use the smallest dx for the CFL condition
print(f"CFL condition: {CFL}")
if CFL > 1:
    raise ValueError("CFL condition is not met. The simulation may be unstable.")

# Initial condition function
def f(x):
    return np.piecewise(x,
                        [x < np.pi/2, (np.pi/2 <= x) & (x <= 3*np.pi/2), (5*np.pi/2 <= x) & (x <= 7*np.pi/2), x > 7*np.pi/2],
                        [0, 1, lambda x: (1 + np.cos(2*x))/2, 0])

# Initial condition
q0 = f(x)

@jit(nopython=True)
def apply_periodic_bc(q):
    q[0] = q[-2]
    q[-1] = q[1]

# First-order upwind method
@jit(nopython=True)
def first_order_upwind(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        for i in range(1, len(q) - 1):
            q_new[i] = q[i] + u * dt / dx[i] * (q[i] - q[i-1])
        apply_periodic_bc(q_new)
        q[:] = q_new[:]
    return q

# Lax-Wendroff method
@jit(nopython=True)
def lax_wendroff(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        q_new[1:-1] = q[1:-1] + u * dt / (2*dx[1:-1]) * (q[2:] - q[:-2]) + (u**2 * dt**2) / (2*dx[1:-1] * (dx[1:-1] + dx[:-2])) * (q[2:] - 2*q[1:-1] + q[:-2])
        apply_periodic_bc(q_new)
        q[:] = q_new[:]
    return q

# MC flux limiter
@jit(nopython=True)
def mc_flux_limiter(r):
    return np.maximum(0, np.minimum(np.minimum(2*r, (1 + r) / 2), 2))

# High-Resolution method with MC flux limiter (MUSCL)
@jit(nopython=True)
def muscl_mc(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        dq = np.zeros_like(q)
        r = np.zeros_like(q)
        
        dq[1:-1] = (q[2:] - q[:-2]) / 2
        r[1:-1] = dq[:-2] / (dq[1:-1] + 1e-6)  # Add a small value to avoid division by zero
        phi = mc_flux_limiter(r)
        
        qL = q[:-1] + phi[:-1] * (q[1:] - q[:-1]) / 2
        qR = q[1:] - phi[:-1] * (q[1:] - q[:-1]) / 2
        
        flux = -0.5 * u * (qL + qR) - 0.5 * np.abs(u) * (qR - qL)
        
        q_new[1:-1] = q[1:-1] - dt / dx[1:-1] * (flux[1:] - flux[:-1])
        apply_periodic_bc(q_new)
        q[:] = q_new[:]
    return q

# Run simulations
q_upwind = first_order_upwind(np.copy(q0), u, dt, dx, Nt)
q_lax_wendroff = lax_wendroff(np.copy(q0), u, dt, dx, Nt)
q_muscl_mc = muscl_mc(np.copy(q0), u, dt, dx, Nt)

# Exact solution after time T
q_exact = f((x + T * u) % L)

# Plotting results
plt.figure(figsize=(12, 8))

plt.plot(x, q_upwind, label='First-order Upwind')
plt.plot(x, q_lax_wendroff, label='Lax-Wendroff')
plt.plot(x, q_muscl_mc, label='High-Resolution MUSCL MC')
plt.plot(x, q_exact, label='Exact Solution',
         linestyle='-', marker = None, linewidth = 1)

plt.legend()
plt.xlabel('x')
plt.ylabel('q')
plt.title('Comparison of Numerical Methods for Advection Equation on Uniform Grid')
plt.grid()
plt.show()
