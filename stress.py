import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Parameters
L = 4 * np.pi
T = 5  # 5 periods

Nx = 1000  # Number of spatial points
u = -1  
CFL = 0.1
mu = 1
rho = 0.25
c = np.sqrt(mu / rho)

# Create a non-uniform grid
x = np.linspace(0, L, Nx)
dx = np.diff(x)
dx = np.append(dx, dx[-1])  # Extend the last dx for boundary conditions

# Ensure CFL condition is met
dt = CFL * np.min(np.abs(dx))

Nt = int(np.round(T / dt))
print(f"CFL condition: {CFL:.2f}")
if CFL > 1:
    print("CFL condition is not met. The simulation may be unstable.")

# Initial condition function
def f(x):
    return np.piecewise(x,
                        [x < np.pi/2, (np.pi/2 <= x) & (x <= 3*np.pi/2), (5*np.pi/2 <= x) & (x <= 7*np.pi/2), x > 7*np.pi/2],
                        [0, 1, lambda x: (1 + np.cos(2*x))/2, 0])

# Initial conditions
sigma0 = np.ones_like(x)
v0 = f(x)

# Characteristics
w0 = sigma0 + c * v0
z0 = sigma0 - c * v0

@jit(nopython=True)
def apply_periodic_bc(q):
    q[0] = q[-2]
    q[-1] = q[1]



# First-order upwind method for advection
@jit(nopython=True)
def first_order_upwind(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    if u < 0:
        for n in range(Nt):
            q_new[:-1] = q[:-1] - u * dt / dx[:-1] * (q[1:] - q[:-1])
            q_new[-1] = q_new[0]
            q[:] = q_new[:]
    else:
        for n in range(Nt):
            q_new[1:] = q[1:] - u * dt / dx[1:] * (q[1:] - q[:-1])
            q_new[0] = q_new[-1]
            q[:] = q_new[:]
    return q

# Lax-Wendroff method for advection
@jit(nopython=True)
def lax_wendroff(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        cf = u * dt / dx
        q_new[1:-1] = cf[:-2]/2.0 * (1 + cf[:-2]) * q[:-2] + (1 - cf[1:-1]**2) * q[1:-1] - cf[2:]/2.0 * (1 - cf[2:]) * q[2:]
        apply_periodic_bc(q_new)
        q[:] = q_new[:]
    return q

@jit(nopython=True)
def minmod(a, b):
    if a * b <= 0:
        return 0
    else:
        return np.sign(a) * min(abs(a), abs(b))

# MUSCL with MC limiter method
def muscl_mc(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        qim1 = np.roll(q, 1)   # q_{j-1}
        qip1 = np.roll(q, -1)  # q_{j+1}
        
        dqR = qip1 - q
        dqL = q - qim1
        dqC = (qip1 - qim1) / 2.0
        
        dq = np.zeros_like(q)
        for j in range(len(q)):
            dq[j] = minmod(minmod(2 * dqR[j], 2 * dqL[j]), dqC[j])
        
        qiph_M = q + dq / 2.0
        qimh_M = q - dq / 2.0

        qL = qiph_M[:-1]
        qR = qimh_M[1:]

        flux = 0.5 * u * (qL + qR) - 0.5 * np.abs(u) * (qR - qL)
        
        q_new[1:-1] = q[1:-1] - dt / dx[1:-1] * (flux[1:] - flux[:-1])
        apply_periodic_bc(q_new)
        q[:] = q_new[:]
    return q

# Exact solutions for characteristic variables
w_exact = sigma0 + c * f((x - T * c) % L)
z_exact = sigma0 - c * f((x + T * c) % L)


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, L)
# ax.set_ylim(-0.2, 2.0)
ax.set_xlabel('x')
ax.set_ylabel('w')
ax.set_title(f'Characteristic Variable w, CFL = {CFL:.2f}')

w_upwind = first_order_upwind(np.copy(w0), c, dt, dx, Nt)
w_lax_wendroff = lax_wendroff(np.copy(w0), c, dt, dx, Nt)
w_muscl_mc = muscl_mc(np.copy(w0), c, dt, dx, Nt)

line0, = ax.plot(x, w0, '-.', c='blue', label='Initial Condition (w)', linewidth=1.5, alpha=0.2)
line1, = ax.plot(x, w_upwind, '-', c='purple', label='Upwind (w)')
line2, = ax.plot(x, w_lax_wendroff, '-', c='orange', label='Lax-Wendroff (w)')
line3, = ax.plot(x, w_muscl_mc, '-', c='green', label='MUSCL w/ MC (w)')
line4, = ax.plot(x, w_exact, '-.', c='blue', label='Exact Solution (w)', linewidth=1.5)

ax.legend(loc="upper right")
ax.grid()
fig.tight_layout()

plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, L)
# ax.set_ylim(-0.2, 2.0)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title(f'Characteristic Variable z, CFL = {CFL:.2f}')

z_upwind = first_order_upwind(np.copy(z0), -c, dt, dx, Nt)
z_lax_wendroff = lax_wendroff(np.copy(z0), -c, dt, dx, Nt)
z_muscl_mc = muscl_mc(np.copy(z0), -c, dt, dx, Nt)

line0, = ax.plot(x, z0, '-.', c='blue', label='Initial Condition (z)', linewidth=1.5, alpha=0.2)
line1, = ax.plot(x, z_upwind, '-', c='purple', label='Upwind (z)')
line2, = ax.plot(x, z_lax_wendroff, '-', c='orange', label='Lax-Wendroff (z)')
line3, = ax.plot(x, z_muscl_mc, '-', c='green', label='MUSCL w/ MC (z)')
line4, = ax.plot(x, z_exact, '-.', c='blue', label='Exact Solution (z)', linewidth=1.5)

ax.legend(loc="upper right")
ax.grid()
fig.tight_layout()

plt.show()

# Transform 
sigma_upwind = 0.5 * (w_upwind + z_upwind)
v_upwind = 0.5 * (w_upwind - z_upwind) / c

sigma_lax_wendroff = 0.5 * (w_lax_wendroff + z_lax_wendroff)
v_lax_wendroff = 0.5 * (w_lax_wendroff - z_lax_wendroff) / c

sigma_muscl_mc = 0.5 * (w_muscl_mc + z_muscl_mc)
v_muscl_mc = 0.5 * (w_muscl_mc - z_muscl_mc) / c

# Exact solutions for sigma and v
sigma_exact = 0.5 * (w_exact + z_exact)
v_exact = 0.5 * (w_exact - z_exact) / c


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, L)
# ax.set_ylim(-0.2, 2.0)
ax.set_xlabel('x')
ax.set_ylabel('sigma')
ax.set_title(f'Stress Equations, CFL = {CFL:.2f}')

line0, = ax.plot(x, sigma0, '-.', c='blue', label='Initial Condition (sigma)', linewidth=1.5, alpha=0.2)
line1, = ax.plot(x, sigma_upwind, '-', c='purple', label='Upwind (sigma)')
line2, = ax.plot(x, sigma_lax_wendroff, '-', c='orange', label='Lax-Wendroff (sigma)')
line3, = ax.plot(x, sigma_muscl_mc, '-', c='green', label='MUSCL w/ MC (sigma)')
line4, = ax.plot(x, sigma_exact, '-.', c='blue', label='Exact Solution (sigma)', linewidth=1.5)

ax.legend(loc="upper right")
ax.grid()
fig.tight_layout()

plt.show()



fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, L)
# ax.set_ylim(-0.2, 2.0)
ax.set_xlabel('x')
ax.set_ylabel('sigma')
ax.set_title(f'Stress Equations, CFL = {CFL:.2f}')

line0, = ax.plot(x, v_upwind, '-.', c='blue', label='Initial Condition (v)', linewidth=1.5, alpha=0.2)
line1, = ax.plot(x, v_upwind, '-', c='purple', label='Upwind (v)')
line2, = ax.plot(x, v_lax_wendroff, '-', c='orange', label='Lax-Wendroff (v)')
line3, = ax.plot(x, v_muscl_mc, '-', c='green', label='MUSCL w/ MC (v)')
line4, = ax.plot(x, v_exact, '-.', c='blue', label='Exact Solution (v)', linewidth=1.5)

ax.legend(loc="upper right")
ax.grid()
fig.tight_layout()

plt.show()

