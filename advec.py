import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit

non_uniform = False


# Parameters
L = 4 * np.pi
T = 5  # 5 periods

Nx = 200  # Number of spatial points
# Nt = 100  # Number of time steps
u = -1  # Advection velocity u
CFL = 1

if non_uniform is True:
    x_1 = np.linspace(0, L/2, int(Nx * 2/3), endpoint=False)
    x_2 = np.linspace(L/2,L, Nx - int(Nx * 2/3))

    x = np.concatenate([x_1,x_2])
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])  # Extend the last dx for boundary conditions


else:
    # # Create a non-uniform grid
    x = np.linspace(0, L, Nx)
    # x *= np.exp(x)
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])  # Extend the last dx for boundary conditions

# Chebyshev nodes in [-1, 1]
# chebyshev_nodes = np.cos(np.pi * (2 * np.arange(1, Nx + 1) - 1) / (2 * Nx))


# x = 0.5 * L * (chebyshev_nodes + 1)


# dx = np.diff(x)
# dx = np.append(dx, dx[-1])  # Extend the last dx for boundary conditions


# plt.plot(x,dx)
# plt.show()

# Ensure CFL condition is met
dt = CFL * np.min(np.abs(dx)) /np.abs(u)
Nt = int(np.round(T/dt))

# CFL = np.abs(u) * dt / np.min(dx)  # Use the smallest dx for the CFL condition

print(f"Max CFL condition found: {CFL:.2f}")
print(f"Number of timesteps: {Nt}")
print(f"Timestep dt: {dt}")

if CFL > 1:
    print("CFL condition is not met. The simulation may be unstable.")

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

@jit(nopython=True)
def apply_periodic_bc_upwind(q):
    q[-2] = q[0]
    q[-1] = q[1]

# First-order upwind method
@jit(nopython=True)
def first_order_upwind(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        q_new[:-2] = q[:-2] - u * dt / dx[:-2] * (q[1:-1] - q[:-2])
        apply_periodic_bc_upwind(q_new)
        q[:] = q_new[:]
    return q

# Lax-Wendroff method
@jit(nopython=True)
def lax_wendroff(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        c = u * dt / dx
        q_new[1:-1] = c[:-2]/2.0 * (1 + c[:-2]) * q[:-2] + (1 - c[1:-1]**2) * q[1:-1] - c[2:]/2.0 * (1 - c[2:]) * q[2:]
        q_new[0] = q_new[-2]
        q_new[-1] = q_new[1]
        q[:] = q_new[:]
    return q

@jit(nopython=True)
def minmod(a, b):
    if a * b <= 0:
        return 0
    else:
        return np.sign(a) * min(abs(a), abs(b))



# MUSCL with MC limiter method
# @jit(nopython=True)
# def muscl_mc(q, u, dt, dx, Nt):
#     q_new = np.copy(q)
#     for n in range(Nt):
#         qim1 = np.roll(q, 1)   # q_{j-1}
#         qip1 = np.roll(q, -1)  # q_{j+1}
        
#         dqR = qip1 - q
#         dqL = q - qim1
#         dqC = (qip1 - qim1) / 2.0
        
#         dq = np.zeros_like(q)
#         for j in range(len(q)):
#             dq[j] = minmod(minmod(2 * dqR[j], 2 * dqL[j]), dqC[j])
        
#         # Left and Right extrapolated q-values at the boundary j+1/2
#         qiph_M = q + dq / 2.0  # q_{j+1/2}^{-} from cell j
#         qimh_M = q - dq / 2.0  # q_{j+1/2}^{+} from cell j

#         qL = qiph_M[:-1]
#         qR = qimh_M[1:]

#         flux = 0.5 * u * (qL + qR) - 0.5 * np.abs(u) * (qR - qL)
        
#         q_new[1:-1] = q[1:-1] - dt / dx[1:-1] * (flux[1:] - flux[:-1])
#         q_new[0] = q_new[-2]
#         q_new[-1] = q_new[1]
#         q[:] = q_new[:]
#     return q

# @jit(nopython=True)
def muscl_mc_trial(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        v = u * dt / dx[1:-1]

        dQ_inh = q - np.roll(q, 1)
        dQ_iph = np.roll(q, -1) - q
        dQ_ipf = np.roll(q, -2) - np.roll(q, -1)

        theta_in = np.where(np.abs(dQ_inh) > 1e-6, dQ_iph / dQ_inh, 0)
        theta_ip = np.where(np.abs(dQ_iph) > 1e-6, dQ_ipf / dQ_iph, 0)

        psi_in = np.zeros_like(q)
        psi_ip = np.zeros_like(q)
        # print(np.array([(1 + theta_in) / 2, 2*np.ones_like(theta_in), 2 * theta_in]).min(axis=0))
        # for j in range(len(q)):
        psi_in = np.max([psi_in, np.array([(1 + theta_in) / 2, 2*np.ones_like(theta_in), 2 * theta_in]).min(axis=0)],axis=0)
        psi_ip = np.max([psi_ip, np.array([(1 + theta_ip) / 2, 2*np.ones_like(theta_in), 2 * theta_ip]).min(axis=0)],axis=0)

        q_new[1:-1] = (
            q[1:-1]
            - v * (q[2:] - q[1:-1])
            + 0.5 * v * (1 + v) * (psi_ip[1:-1] * (q[2:] - q[1:-1]) - psi_in[1:-1] * (q[1:-1] - q[:-2]))
        )
        q_new[0] = q_new[-2]
        q_new[-1] = q_new[1]
        q[:] = q_new[:]
    return q


# Exact solution after time T
q_exact = f((x - T * u) % L)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, L)
ax.set_ylim(-0.2, 2)
ax.set_xlabel('x')
ax.set_ylabel('q')


if non_uniform:
    ax.set_title(f'1D Advection (Non-Uniform), CFL = {CFL:.2f}, ' + r'$\bar{u}$ =' + f'{u}')
    ax.axvspan(0, L/2, alpha=0.1, color='red',label = 'fine')
    ax.axvspan(L/2, L, alpha=0.1, color='green', label = 'coarse')
else:
    ax.set_title(f'1D Advection (Uniform), CFL = {CFL:.2f}, ' + r'$\bar{u}$ =' + f'{u}')

q_upwind = first_order_upwind(np.copy(q0), u, dt, dx, Nt)
q_lax_wendroff = lax_wendroff(np.copy(q0), u, dt, dx, Nt)
q_muscl_mc = muscl_mc_trial(np.copy(q0), u, dt, dx, Nt)

line0, = ax.plot(x, q0, '-.', c='blue', label='Initial Condition', linewidth=1.5, alpha=0.2)
line1, = ax.plot(x, q_upwind, '-', c='purple', label='Upwind')
line2, = ax.plot(x, q_lax_wendroff, '-', c='orange', label='Lax-Wendroff')
line3, = ax.plot(x, q_muscl_mc, '-', c='green', label='MUSCL w/ MC')
line4, = ax.plot(x, q_exact, '-.', c='blue', label='Exact Solution', linewidth=1.5)

title = ax.text(0.925,0.03, f"", bbox={'facecolor':'w', 'alpha':0.5, 'pad':6},
                transform=ax.transAxes, ha="center")

ax.legend(loc="upper right", ncol=2)
ax.grid()
fig.tight_layout()

if non_uniform:
    fig.savefig(f"figures/advec_T_{T}_CFL_{CFL:.2e}_nx_{Nx:.1e}_NU.pdf")
else:
    fig.savefig(f"figures/advec_T_{T}_CFL_{CFL:.2e}_nx_{Nx:.1e}_U.pdf")

# plt.show()
