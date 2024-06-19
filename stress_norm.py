import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Parameters
L = 4 * np.pi
T = 5 * 2 * np.pi

Nx = 500  # Number of spatial points
CFL = 0.5
mu = 1
rho = 0.25
c = np.sqrt(mu / rho)

# Create a non-uniform grid
x = np.linspace(0, L, Nx)
dx = np.diff(x)
dx = np.append(dx, dx[-1])  # Extend the last dx for boundary conditions

# Ensure CFL condition is met
dt = CFL * np.min(np.abs(dx)) / c

Nt = int(np.round(T / dt))
print(f"CFL condition: {CFL:.1e}")
if CFL > 1:
    print("CFL condition is not met. The simulation may be unstable.")

# Initial condition function
def f(x):
    return np.piecewise(x,
                        [x < np.pi / 2, (np.pi / 2 <= x) & (x <= 3 * np.pi / 2), (5 * np.pi / 2 <= x) & (x <= 7 * np.pi / 2), x > 7 * np.pi / 2],
                        [0, 1, lambda x: (1 + np.cos(2 * x)) / 2, 0])

# Initial conditions
sigma0 = np.ones_like(x)
v0 = f(x)

# Characteristics
w0 = 0.5 * sigma0 + 0.25 * v0
z0 = -0.5 * sigma0 + 0.25 * v0

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
        q_new[1:-1] = cf[:-2] / 2.0 * (1 + cf[:-2]) * q[:-2] + (1 - cf[1:-1]**2) * q[1:-1] - cf[2:] / 2.0 * (1 - cf[2:]) * q[2:]
        apply_periodic_bc(q_new)
        q[:] = q_new[:]
    return q

# MUSCL with MC limiter method
def muscl_mc(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        v = u * dt / dx[1:-1]

        if u > 0:
            dQ_inh = q - np.roll(q, 1)
            dQ_iph = np.roll(q, -1) - q
            dQ_inf = np.roll(q, 1) - np.roll(q, 2)

            theta_in = np.where(np.abs(dQ_inh) > 1e-6, dQ_inf / dQ_inh, 0)
            theta_ip = np.where(np.abs(dQ_iph) > 1e-6, dQ_inh / dQ_iph, 0)

            psi_in = np.zeros_like(q)
            psi_ip = np.zeros_like(q)
            psi_in = np.max([psi_in, np.array([(1 + theta_in) / 2, 2 * np.ones_like(theta_in), 2 * theta_in]).min(axis=0)], axis=0)
            psi_ip = np.max([psi_ip, np.array([(1 + theta_ip) / 2, 2 * np.ones_like(theta_in), 2 * theta_ip]).min(axis=0)], axis=0)

            q_new[1:-1] = (
                q[1:-1]
                - v * (q[1:-1] - q[:-2])
                - 0.5 * v * (1 - v) * (psi_ip[1:-1] * (q[2:] - q[1:-1]) - psi_in[1:-1] * (q[1:-1] - q[:-2]))
            )

        else:
            dQ_inh = q - np.roll(q, 1)
            dQ_iph = np.roll(q, -1) - q
            dQ_ipf = np.roll(q, -2) - np.roll(q, -1)

            theta_in = np.where(np.abs(dQ_inh) > 1e-6, dQ_iph / dQ_inh, 0)
            theta_ip = np.where(np.abs(dQ_iph) > 1e-6, dQ_ipf / dQ_iph, 0)

            psi_in = np.zeros_like(q)
            psi_ip = np.zeros_like(q)
            psi_in = np.max([psi_in, np.array([(1 + theta_in) / 2, 2 * np.ones_like(theta_in), 2 * theta_in]).min(axis=0)], axis=0)
            psi_ip = np.max([psi_ip, np.array([(1 + theta_ip) / 2, 2 * np.ones_like(theta_in), 2 * theta_ip]).min(axis=0)], axis=0)

            q_new[1:-1] = (
                q[1:-1]
                - v * (q[2:] - q[1:-1])
                + 0.5 * v * (1 + v) * (psi_ip[1:-1] * (q[2:] - q[1:-1]) - psi_in[1:-1] * (q[1:-1] - q[:-2]))
            )

        q_new[0] = q_new[-2]
        q_new[-1] = q_new[1]
        q[:] = q_new[:]
    return q

@jit(nopython=True)
def get_L1(q, q_ex):
    return np.sum(np.abs(q - q_ex)) / np.sum(np.abs(q_ex))

@jit(nopython=True)
def get_L2(q, q_ex):
    return np.sum((q - q_ex)**2) / np.sum(q_ex**2)

# Exact solutions for characteristic variables
w_exact = 0.5 * sigma0 + 0.25 * f((x + T * 2) % L)
z_exact = -0.5 * sigma0 + 0.25 * f((x + T * -2) % L)

# Initialize arrays to store norms
time_steps = np.arange(Nt) * dt
L1_norms_w_upwind = np.zeros(Nt)
L2_norms_w_upwind = np.zeros(Nt)
L1_norms_w_lax_wendroff = np.zeros(Nt)
L2_norms_w_lax_wendroff = np.zeros(Nt)
L1_norms_w_muscl_mc = np.zeros(Nt)
L2_norms_w_muscl_mc = np.zeros(Nt)

L1_norms_z_upwind = np.zeros(Nt)
L2_norms_z_upwind = np.zeros(Nt)
L1_norms_z_lax_wendroff = np.zeros(Nt)
L2_norms_z_lax_wendroff = np.zeros(Nt)
L1_norms_z_muscl_mc = np.zeros(Nt)
L2_norms_z_muscl_mc = np.zeros(Nt)

L1_norms_sigma_upwind = np.zeros(Nt)
L2_norms_sigma_upwind = np.zeros(Nt)
L1_norms_sigma_lax_wendroff = np.zeros(Nt)
L2_norms_sigma_lax_wendroff = np.zeros(Nt)
L1_norms_sigma_muscl_mc = np.zeros(Nt)
L2_norms_sigma_muscl_mc = np.zeros(Nt)

L1_norms_v_upwind = np.zeros(Nt)
L2_norms_v_upwind = np.zeros(Nt)
L1_norms_v_lax_wendroff = np.zeros(Nt)
L2_norms_v_lax_wendroff = np.zeros(Nt)
L1_norms_v_muscl_mc = np.zeros(Nt)
L2_norms_v_muscl_mc = np.zeros(Nt)

# Simulation
w_upwind = np.copy(w0)
w_lax_wendroff = np.copy(w0)
w_muscl_mc = np.copy(w0)
z_upwind = np.copy(z0)
z_lax_wendroff = np.copy(z0)
z_muscl_mc = np.copy(z0)

for n in range(Nt):
    w_upwind = first_order_upwind(w_upwind, c, dt, dx, 1)
    w_lax_wendroff = lax_wendroff(w_lax_wendroff, c, dt, dx, 1)
    w_muscl_mc = muscl_mc(w_muscl_mc, c, dt, dx, 1)

    z_upwind = first_order_upwind(z_upwind, c, dt, dx, 1)
    z_lax_wendroff = lax_wendroff(z_lax_wendroff, c, dt, dx, 1)
    z_muscl_mc = muscl_mc(z_muscl_mc, c, dt, dx, 1)

    # Calculate norms for w and z
    L1_norms_w_upwind[n] = get_L1(w_upwind, w_exact)
    L2_norms_w_upwind[n] = get_L2(w_upwind, w_exact)
    L1_norms_w_lax_wendroff[n] = get_L1(w_lax_wendroff, w_exact)
    L2_norms_w_lax_wendroff[n] = get_L2(w_lax_wendroff, w_exact)
    L1_norms_w_muscl_mc[n] = get_L1(w_muscl_mc, w_exact)
    L2_norms_w_muscl_mc[n] = get_L2(w_muscl_mc, w_exact)

    L1_norms_z_upwind[n] = get_L1(z_upwind, z_exact)
    L2_norms_z_upwind[n] = get_L2(z_upwind, z_exact)
    L1_norms_z_lax_wendroff[n] = get_L1(z_lax_wendroff, z_exact)
    L2_norms_z_lax_wendroff[n] = get_L2(z_lax_wendroff, z_exact)
    L1_norms_z_muscl_mc[n] = get_L1(z_muscl_mc, z_exact)
    L2_norms_z_muscl_mc[n] = get_L2(z_muscl_mc, z_exact)

    # Transform
    sigma_upwind = (w_upwind - z_upwind)
    v_upwind = 2 * (w_upwind + z_upwind)

    sigma_lax_wendroff = (w_lax_wendroff - z_lax_wendroff)
    v_lax_wendroff = 2 * (w_lax_wendroff + z_lax_wendroff)

    sigma_muscl_mc = (w_muscl_mc - z_muscl_mc)
    v_muscl_mc = 2 * (w_muscl_mc + z_muscl_mc)

    # Exact solutions for sigma and v
    sigma_exact = (w_exact - z_exact)
    v_exact = 2 * (w_exact + z_exact)

    # Calculate norms for sigma and v
    L1_norms_sigma_upwind[n] = get_L1(sigma_upwind, sigma_exact)
    L2_norms_sigma_upwind[n] = get_L2(sigma_upwind, sigma_exact)
    L1_norms_sigma_lax_wendroff[n] = get_L1(sigma_lax_wendroff, sigma_exact)
    L2_norms_sigma_lax_wendroff[n] = get_L2(sigma_lax_wendroff, sigma_exact)
    L1_norms_sigma_muscl_mc[n] = get_L1(sigma_muscl_mc, sigma_exact)
    L2_norms_sigma_muscl_mc[n] = get_L2(sigma_muscl_mc, sigma_exact)

    L1_norms_v_upwind[n] = get_L1(v_upwind, v_exact)
    L2_norms_v_upwind[n] = get_L2(v_upwind, v_exact)
    L1_norms_v_lax_wendroff[n] = get_L1(v_lax_wendroff, v_exact)
    L2_norms_v_lax_wendroff[n] = get_L2(v_lax_wendroff, v_exact)
    L1_norms_v_muscl_mc[n] = get_L1(v_muscl_mc, v_exact)
    L2_norms_v_muscl_mc[n] = get_L2(v_muscl_mc, v_exact)

# Plot norms evolution over time for w
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_steps, L1_norms_w_upwind, label='Upwind (w) - L1')
ax.plot(time_steps, L2_norms_w_upwind, label='Upwind (w) - L2')
ax.plot(time_steps, L1_norms_w_lax_wendroff, label='Lax-Wendroff (w) - L1')
ax.plot(time_steps, L2_norms_w_lax_wendroff, label='Lax-Wendroff (w) - L2')
ax.plot(time_steps, L1_norms_w_muscl_mc, label='MUSCL w/ MC (w) - L1')
ax.plot(time_steps, L2_norms_w_muscl_mc, label='MUSCL w/ MC (w) - L2')
ax.set_xlabel('Time')
ax.set_ylabel('Norm')
ax.set_title('Evolution of L1 and L2 Norms for w')
ax.legend(loc='upper right')
ax.set_ylim(-0.1, 2.2)
ax.grid()
fig.savefig(f"stress_norm_w_CFL_{CFL}_Nx_{Nx}.pdf")

# Plot norms evolution over time for z
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_steps, L1_norms_z_upwind, label='Upwind (z) - L1')
ax.plot(time_steps, L2_norms_z_upwind, label='Upwind (z) - L2')
ax.plot(time_steps, L1_norms_z_lax_wendroff, label='Lax-Wendroff (z) - L1')
ax.plot(time_steps, L2_norms_z_lax_wendroff, label='Lax-Wendroff (z) - L2')
ax.plot(time_steps, L1_norms_z_muscl_mc, label='MUSCL w/ MC (z) - L1')
ax.plot(time_steps, L2_norms_z_muscl_mc, label='MUSCL w/ MC (z) - L2')
ax.set_xlabel('Time')
ax.set_ylabel('Norm')
ax.set_title('Evolution of L1 and L2 Norms for z')
ax.legend(loc='upper right')
ax.set_ylim(-0.1, 2.2)
ax.grid()
fig.savefig(f"stress_norm_z_CFL_{CFL}_Nx_{Nx}.pdf")

# Plot norms evolution over time for sigma
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_steps, L1_norms_sigma_upwind, label='Upwind (sigma) - L1')
ax.plot(time_steps, L2_norms_sigma_upwind, label='Upwind (sigma) - L2')
ax.plot(time_steps, L1_norms_sigma_lax_wendroff, label='Lax-Wendroff (sigma) - L1')
ax.plot(time_steps, L2_norms_sigma_lax_wendroff, label='Lax-Wendroff (sigma) - L2')
ax.plot(time_steps, L1_norms_sigma_muscl_mc, label='MUSCL w/ MC (sigma) - L1')
ax.plot(time_steps, L2_norms_sigma_muscl_mc, label='MUSCL w/ MC (sigma) - L2')
ax.set_xlabel('Time')
ax.set_ylabel('Norm')
ax.set_title('Evolution of L1 and L2 Norms for sigma')
ax.legend(loc='upper right')
ax.set_ylim(-0.1, 2.2)
ax.grid()
fig.savefig(f"stress_norm_sigma_CFL_{CFL}_Nx_{Nx}.pdf")

# Plot norms evolution over time for v
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_steps, L1_norms_v_upwind, label='Upwind (v) - L1')
ax.plot(time_steps, L2_norms_v_upwind, label='Upwind (v) - L2')
ax.plot(time_steps, L1_norms_v_lax_wendroff, label='Lax-Wendroff (v) - L1')
ax.plot(time_steps, L2_norms_v_lax_wendroff, label='Lax-Wendroff (v) - L2')
ax.plot(time_steps, L1_norms_v_muscl_mc, label='MUSCL w/ MC (v) - L1')
ax.plot(time_steps, L2_norms_v_muscl_mc, label='MUSCL w/ MC (v) - L2')
ax.set_xlabel('Time')
ax.set_ylabel('Norm')
ax.set_title('Evolution of L1 and L2 Norms for v')
ax.legend(loc='upper right')
ax.set_ylim(-0.1, 2.2)
ax.grid()
fig.savefig(f"stress_norm_v_CFL_{CFL}_Nx_{Nx}.pdf")

raise "error"

# Transform
sigma_upwind = (w_upwind - z_upwind)
v_upwind = 2 * (w_upwind + z_upwind)

sigma_lax_wendroff = (w_lax_wendroff - z_lax_wendroff)
v_lax_wendroff = 2 * (w_lax_wendroff + z_lax_wendroff)

sigma_muscl_mc = (w_muscl_mc - z_muscl_mc)
v_muscl_mc = 2 * (w_muscl_mc + z_muscl_mc)

# Exact solutions for sigma and v
sigma_exact = (w_exact - z_exact)
v_exact = 2 * (w_exact + z_exact)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(x, w_upwind, label='Numerical')
axs[0, 0].plot(x, w_exact, 'r--', label='Exact')
axs[0, 0].set_title("Characteristic variable $w$ using First-order Upwind Method")
axs[0, 0].set_xlabel('$x$')
axs[0, 0].set_ylabel('$w$')
axs[0, 0].legend()

axs[0, 1].plot(x, z_upwind, label='Numerical')
axs[0, 1].plot(x, z_exact, 'r--', label='Exact')
axs[0, 1].set_title("Characteristic variable $z$ using First-order Upwind Method")
axs[0, 1].set_xlabel('$x$')
axs[0, 1].set_ylabel('$z$')
axs[0, 1].legend()

axs[1, 0].plot(x, sigma_upwind, label='Numerical')
axs[1, 0].plot(x, sigma_exact, 'r--', label='Exact')
axs[1, 0].set_title(r"$\sigma$ using First-order Upwind Method")
axs[1, 0].set_xlabel('$x$')
axs[1, 0].set_ylabel(r"$\sigma$")
axs[1, 0].legend()

axs[1, 1].plot(x, v_upwind, label='Numerical')
axs[1, 1].plot(x, v_exact, 'r--', label='Exact')
axs[1, 1].set_title(r"$v$ using First-order Upwind Method")
axs[1, 1].set_xlabel('$x$')
axs[1, 1].set_ylabel('$v$')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(x, w_lax_wendroff, label='Numerical')
axs[0, 0].plot(x, w_exact, 'r--', label='Exact')
axs[0, 0].set_title("Characteristic variable $w$ using Lax-Wendroff Method")
axs[0, 0].set_xlabel('$x$')
axs[0, 0].set_ylabel('$w$')
axs[0, 0].legend()

axs[0, 1].plot(x, z_lax_wendroff, label='Numerical')
axs[0, 1].plot(x, z_exact, 'r--', label='Exact')
axs[0, 1].set_title("Characteristic variable $z$ using Lax-Wendroff Method")
axs[0, 1].set_xlabel('$x$')
axs[0, 1].set_ylabel('$z$')
axs[0, 1].legend()

axs[1, 0].plot(x, sigma_lax_wendroff, label='Numerical')
axs[1, 0].plot(x, sigma_exact, 'r--', label='Exact')
axs[1, 0].set_title(r"$\sigma$ using Lax-Wendroff Method")
axs[1, 0].set_xlabel('$x$')
axs[1, 0].set_ylabel(r"$\sigma$")
axs[1, 0].legend()

axs[1, 1].plot(x, v_lax_wendroff, label='Numerical')
axs[1, 1].plot(x, v_exact, 'r--', label='Exact')
axs[1, 1].set_title(r"$v$ using Lax-Wendroff Method")
axs[1, 1].set_xlabel('$x$')
axs[1, 1].set_ylabel('$v$')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(x, w_muscl_mc, label='Numerical')
axs[0, 0].plot(x, w_exact, 'r--', label='Exact')
axs[0, 0].set_title("Characteristic variable $w$ using MUSCL with MC Limiter Method")
axs[0, 0].set_xlabel('$x$')
axs[0, 0].set_ylabel('$w$')
axs[0, 0].legend()

axs[0, 1].plot(x, z_muscl_mc, label='Numerical')
axs[0, 1].plot(x, z_exact, 'r--', label='Exact')
axs[0, 1].set_title("Characteristic variable $z$ using MUSCL with MC Limiter Method")
axs[0, 1].set_xlabel('$x$')
axs[0, 1].set_ylabel('$z$')
axs[0, 1].legend()

axs[1, 0].plot(x, sigma_muscl_mc, label='Numerical')
axs[1, 0].plot(x, sigma_exact, 'r--', label='Exact')
axs[1, 0].set_title(r"$\sigma$ using MUSCL with MC Limiter Method")
axs[1, 0].set_xlabel('$x$')
axs[1, 0].set_ylabel(r"$\sigma$")
axs[1, 0].legend()

axs[1, 1].plot(x, v_muscl_mc, label='Numerical')
axs[1, 1].plot(x, v_exact, 'r--', label='Exact')
axs[1, 1].set_title(r"$v$ using MUSCL with MC Limiter Method")
axs[1, 1].set_xlabel('$x$')
axs[1, 1].set_ylabel('$v$')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
