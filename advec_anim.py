import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.animation import FuncAnimation, PillowWriter

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

# Initialize the data
q_upwind = np.copy(q0)
q_lax_wendroff = np.copy(q0)
q_muscl_mc = np.copy(q0)

# Prepare the figure
fig, ax = plt.subplots(figsize=(12, 8))

line_upwind, = ax.plot(x, q_upwind, label='First-order Upwind')
line_lax_wendroff, = ax.plot(x, q_lax_wendroff, label='Lax-Wendroff')
line_muscl_mc, = ax.plot(x, q_muscl_mc, label='High-Resolution MUSCL MC')
line_exact, = ax.plot(x, q0, label='Exact Solution', linestyle='-', linewidth=1)

ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('q')
ax.set_title('Comparison of Numerical Methods for Advection Equation on Uniform Grid')
ax.grid()


def init():
    line_upwind.set_ydata(np.copy(q0))
    line_lax_wendroff.set_ydata(np.copy(q0))
    line_muscl_mc.set_ydata(np.copy(q0))
    line_exact.set_ydata(np.copy(q0))
    return line_upwind, line_lax_wendroff, line_muscl_mc, line_exact

def update(frame):
    global q_upwind, q_lax_wendroff, q_muscl_mc
    q_upwind = first_order_upwind(np.copy(q_upwind), u, dt, dx, 1)
    q_lax_wendroff = lax_wendroff(np.copy(q_lax_wendroff), u, dt, dx, 1)
    q_muscl_mc = muscl_mc(np.copy(q_muscl_mc), u, dt, dx, 1)
    q_exact = f((x + frame * dt * u) % L)
    
    line_upwind.set_ydata(q_upwind)
    line_lax_wendroff.set_ydata(q_lax_wendroff)
    line_muscl_mc.set_ydata(q_muscl_mc)
    line_exact.set_ydata(q_exact)
    return line_upwind, line_lax_wendroff, line_muscl_mc, line_exact

anim = FuncAnimation(fig, update, frames=range(0, Nt, Nt // 100), init_func=init)

# Save the animation
anim.save('advection_simulation.mp4', writer='ffmpeg', fps=30)
# anim.save('advection_simulation.gif', writer=PillowWriter(fps=30))

plt.show()

#STILL NEED TO FIC
