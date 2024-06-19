import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit
from matplotlib import animation

non_uniform = False

# Parameters
L = 4 * np.pi
T = 2 *np.pi + np.pi/2  # 5 periods

Nx = 200  # Number of spatial points
# Nt = 100  # Number of time steps
u = -1  # Advection velocity u
CFL = 0.5

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

# # Transform nodes to the interval [0, L]
# x = 0.5 * L * (chebyshev_nodes + 1)

# # Compute the differences
# dx = np.diff(x)
# dx = np.append(dx, dx[-1])  # Extend the last dx for boundary conditions

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

@jit(nopython=True)
def get_L1(q,q_ex):
    return np.sum(np.abs(q-q_ex)) / np.sum(np.abs(q_ex))

@jit(nopython=True)
def get_L2(q,q_ex):
    return np.sum((q-q_ex)**2) / np.sum(q_ex**2)

def muscl_mc(q, u, dt, dx, Nt):
    q_new = np.copy(q)
    for n in range(Nt):
        v = u * dt / dx[1:-1]

        dQ_inh = q - np.roll(q, 1)
        dQ_iph = np.roll(q, -1) - q
        dQ_ipf = np.roll(q, -2) - np.roll(q, -1)

        theta_in = np.where(dQ_inh != 0, dQ_iph / dQ_inh, 0)
        theta_ip = np.where(dQ_iph != 0, dQ_ipf / dQ_iph, 0)

        psi_in = np.zeros_like(q)
        psi_ip = np.zeros_like(q)

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
fig, (ax, ax_norm) = plt.subplots(2, 1, figsize=(12, 7))
ax.set_xlim(0, L)
ax.set_ylim(-0.2, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('q')
ax.set_title(f'1D Advection, CFL = {CFL:.2f}, ' + r'$\bar{u}$ =' + f'{u}')

ax_norm.set_xlim(0, T)
ax_norm.set_ylim(0, 2)
ax_norm.set_xlabel('Time')
ax_norm.set_ylabel('Norm Value')

q_upwind = first_order_upwind(np.copy(q0), u, dt, dx, 1)
q_lax_wendroff = lax_wendroff(np.copy(q0), u, dt, dx, 1)
q_muscl_mc = muscl_mc(np.copy(q0), u, dt, dx, 1)

if non_uniform is True:
    ax.axvspan(0, L/2, alpha=0.1, color='red', label='fine')
    ax.axvspan(L/2, L, alpha=0.1, color='green', label='coarse')

line0, = ax.plot(x, q0, '-.', c='blue', label='Initial Solution', linewidth=1.5, alpha=0.25)
line1, = ax.plot(x, q0, '-', c='purple', label='Upwind')
line2, = ax.plot(x, q0, '-', c='orange', label='Lax-Wendroff')
line3, = ax.plot(x, q0, '-', c='green', label='MUSCL w/ MC')
line4, = ax.plot(x, q0, '-', c='blue', label='Exact Solution', linewidth=1.5)

title = ax.text(0.925, 0.03, f"", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 6},
                transform=ax.transAxes, ha="center")

ax.legend(loc="upper right", ncol=2)
ax.grid()

time_stamp = [0]
L1_norm_upwind = [0]
L1_norm_lax_wendroff = [0]
L1_norm_muscl_mc = [0]

L2_norm_upwind = [0]
L2_norm_lax_wendroff = [0]
L2_norm_muscl_mc = [0]

line_L1_upwind, = ax_norm.plot(0, 0, label="L1 Upwind")
line_L1_lax_wendroff, = ax_norm.plot(0, 0, label="L1 Lax-Wendroff")
line_L1_muscl_mc, = ax_norm.plot(0, 0, label="L1 MUSCL w/ MC")

line_L2_upwind, = ax_norm.plot(0, 0,label="L2 Upwind",linestyle = '--')
line_L2_lax_wendroff, = ax_norm.plot(0, 0, label="L2 Lax-Wendroff",linestyle = '--')
line_L2_muscl_mc, = ax_norm.plot(0, 0, label="L2 MUSCL w/ MC",linestyle = '--')

ax_norm.legend(loc="upper right",ncol =2)
ax_norm.grid()

fig.tight_layout()

v_fac = 1
plot_norm = 1

# Initialization function
def init():
    global q_upwind, q_lax_wendroff, q_muscl_mc, time_stamp, L1_norm_upwind, L1_norm_lax_wendroff, L1_norm_muscl_mc
    global L2_norm_upwind, L2_norm_lax_wendroff, L2_norm_muscl_mc

    q_upwind = np.copy(q0)
    q_lax_wendroff = np.copy(q0)
    q_muscl_mc = np.copy(q0)

    time_stamp = [0]
    L1_norm_upwind = [0]
    L1_norm_lax_wendroff = [0]
    L1_norm_muscl_mc = [0]

    L2_norm_upwind = [0]
    L2_norm_lax_wendroff = [0]
    L2_norm_muscl_mc = [0]

    line0.set_ydata(q0)
    line1.set_ydata(q0)
    line2.set_ydata(q0)
    line3.set_ydata(q0)
    line4.set_ydata(q0)

    line_L1_upwind.set_data(time_stamp, L1_norm_upwind)
    line_L1_lax_wendroff.set_data(time_stamp, L1_norm_lax_wendroff)
    line_L1_muscl_mc.set_data(time_stamp, L1_norm_muscl_mc)

    line_L2_upwind.set_data(time_stamp, L2_norm_upwind)
    line_L2_lax_wendroff.set_data(time_stamp, L2_norm_lax_wendroff)
    line_L2_muscl_mc.set_data(time_stamp, L2_norm_muscl_mc)

    return line0, line1, line2, line3, line4, line_L1_upwind, line_L1_lax_wendroff, line_L1_muscl_mc, line_L2_upwind, line_L2_lax_wendroff, line_L2_muscl_mc

def animate(n):
    global q_upwind, q_lax_wendroff, q_muscl_mc, time_stamp, L1_norm_upwind, L1_norm_lax_wendroff, L1_norm_muscl_mc
    global L2_norm_upwind, L2_norm_lax_wendroff, L2_norm_muscl_mc
    global plot_norm

    q_upwind = first_order_upwind(q_upwind, u, dt, dx, v_fac)
    q_lax_wendroff = lax_wendroff(q_lax_wendroff, u, dt, dx, v_fac)
    q_muscl_mc = muscl_mc(q_muscl_mc, u, dt, dx, v_fac)

    t = (n + 1) * dt * v_fac
    q_exact = f((x - t * u) % L)
    time_stamp.append(t)
    L1_norm_upwind.append(get_L1(q_upwind, q_exact))
    L1_norm_lax_wendroff.append(get_L1(q_lax_wendroff, q_exact))
    L1_norm_muscl_mc.append(get_L1(q_muscl_mc, q_exact))

    L2_norm_upwind.append(get_L2(q_upwind, q_exact))
    L2_norm_lax_wendroff.append(get_L2(q_lax_wendroff, q_exact))
    L2_norm_muscl_mc.append(get_L2(q_muscl_mc, q_exact))

    line0.set_ydata(q0)
    line1.set_ydata(q_upwind)
    line2.set_ydata(q_lax_wendroff)
    line3.set_ydata(q_muscl_mc)
    line4.set_ydata(q_exact)

    line_L1_upwind.set_data(time_stamp, L1_norm_upwind)
    line_L1_lax_wendroff.set_data(time_stamp, L1_norm_lax_wendroff)
    line_L1_muscl_mc.set_data(time_stamp, L1_norm_muscl_mc)

    line_L2_upwind.set_data(time_stamp, L2_norm_upwind)
    line_L2_lax_wendroff.set_data(time_stamp, L2_norm_lax_wendroff)
    line_L2_muscl_mc.set_data(time_stamp, L2_norm_muscl_mc)

    title.set_text(f'Time = {t:.2f}')
    # print(n+1)
    if n+1 == Nt//v_fac and plot_norm ==10:
        
        fig_1 = plt.figure(figsize=(8, 5))

        ax_norm_1 = fig_1.gca()

        ax_norm_1.plot(time_stamp, L1_norm_upwind, 'b-' ,label="L1 Upwind")
        ax_norm_1.plot(time_stamp, L1_norm_lax_wendroff, 'r-',label="L1 Lax-Wendroff")
        ax_norm_1.plot(time_stamp, L1_norm_muscl_mc, label="L1 MUSCL w/ MC")

        ax_norm_1.plot(time_stamp, L2_norm_upwind,label="L2 Upwind",linestyle = '--')
        ax_norm_1.plot(time_stamp, L2_norm_lax_wendroff, label="L2 Lax-Wendroff",linestyle = '--')
        ax_norm_1.plot(time_stamp, L2_norm_muscl_mc, label="L2 MUSCL w/ MC",linestyle = '--')

        ax_norm_1.legend(loc="upper right",ncol =2)
        ax_norm_1.grid()
        ax_norm_1.set_xlabel("Timestamp")
        ax_norm_1.set_ylabel("Norm value")
        ax_norm_1.set_ylim(-0.05,0.1)


        fig_1.tight_layout()

        fig_1.savefig(f"advec_norms_CFL_{CFL}_Nx_{Nx}.pdf")

        plot_norm+=1

    return line0, line1, line2, line3, line4, line_L1_upwind, line_L1_lax_wendroff, line_L1_muscl_mc, line_L2_upwind, line_L2_lax_wendroff, line_L2_muscl_mc

anim = FuncAnimation(fig, animate, init_func=init, frames=Nt//v_fac, interval=5, blit=True)

plt.show()

# writer_ffmpeg = animation.FFMpegWriter(fps=30)
# anim.save(f'advec_1d_CFL_{CFL:.2f}.mp4', writer=writer_ffmpeg)

# try:
#     writer_ffmpeg = animation.FFMpegWriter(fps=30)
#     anim.save(f'advec_1d_CFL_{CFL:.2f}.mp4', writer=writer_ffmpeg)

# except:
#     writer_gif = animation.PillowWriter(fps=30)
#     anim.save(f'advec_1d_CFL_{CFL:.2f}.gif', writer=writer_gif)



# plt.close()
