import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit
from matplotlib import animation


def run_stress(T,CFL,Nx):

    # Parameters
    L = 4 * np.pi
    T = T  # 5 periods
    T_p = T *2*np.pi

    Nx = Nx  # Number of spatial points
    # u = -1  
    CFL = CFL
    mu = 1
    rho = 0.25
    c = np.sqrt(mu / rho)

    # Create a non-uniform grid
    x = np.linspace(0, L, Nx)
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])  # Extend the last dx for boundary conditions

    # Ensure CFL condition is met
    dt = CFL * np.min(np.abs(dx))/c

    Nt = int(np.round(T_p / dt))
    print(f"CFL condition: {CFL:.1e}")
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
    w0 = 0.5 * sigma0 + 0.25 * v0
    z0 = -0.5*sigma0 + 0.25 * v0

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
            v = u * dt / dx[1:-1]

            if u > 0:
                dQ_inh = q - np.roll(q, 1)
                dQ_iph = np.roll(q, -1) - q
                dQ_inf = np.roll(q, 1) - np.roll(q, 2)

                theta_in = np.where(np.abs(dQ_inh) > 1e-6, dQ_inf / dQ_inh, 0)
                theta_ip = np.where(np.abs(dQ_iph) > 1e-6, dQ_inh / dQ_iph, 0)

                psi_in = np.zeros_like(q)
                psi_ip = np.zeros_like(q)
                # print(np.array([(1 + theta_in) / 2, 2*np.ones_like(theta_in), 2 * theta_in]).min(axis=0))
                # for j in range(len(q)):
                psi_in = np.max([psi_in, np.array([(1 + theta_in) / 2, 2*np.ones_like(theta_in), 2 * theta_in]).min(axis=0)],axis=0)
                psi_ip = np.max([psi_ip, np.array([(1 + theta_ip) / 2, 2*np.ones_like(theta_in), 2 * theta_ip]).min(axis=0)],axis=0)

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



    # Exact solutions for characteristic variables
    w_exact = 0.5*sigma0 + 0.25 * f((x + T_p * 2) % L)
    z_exact = -0.5*sigma0 + 0.25 * f((x + T_p * -2) % L)




    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title(f'Characteristic Variable w, CFL = {CFL:.2f}')

    w_upwind = first_order_upwind(np.copy(w0), -c, dt, dx, Nt)
    w_lax_wendroff = lax_wendroff(np.copy(w0), -c, dt, dx, Nt)
    w_muscl_mc = muscl_mc(np.copy(w0), -c, dt, dx, Nt)

    line0, = ax.plot(x, w0, '-.', c='blue', label='Initial Condition (w)', linewidth=1.5, alpha=0.2)
    line1, = ax.plot(x, w_upwind, '-', c='purple', label='Upwind (w)')
    line2, = ax.plot(x, w_lax_wendroff, '-', c='orange', label='Lax-Wendroff (w)')
    line3, = ax.plot(x, w_muscl_mc, '-', c='green', label='MUSCL w/ MC (w)')
    line4, = ax.plot(x, w_exact, '-.', c='blue', label='Exact Solution (w)', linewidth=1.5)

    ax.legend(loc="upper right")
    ax.grid()
    fig.tight_layout()

    fig.savefig(f"figures_stress/stress_CharW_T_{T}_CFL_{CFL:.2e}_nx_{Nx:.1e}.pdf")

    # plt.show()


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(-0.75, 0)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(f'Characteristic Variable z, CFL = {CFL:.2f}')

    z_upwind = first_order_upwind(np.copy(z0), c, dt, dx, Nt)
    z_lax_wendroff = lax_wendroff(np.copy(z0), c, dt, dx, Nt)
    z_muscl_mc = muscl_mc(np.copy(z0), c, dt, dx, Nt)

    line0, = ax.plot(x, z0, '-.', c='blue', label='Initial Condition (z)', linewidth=1.5, alpha=0.2)
    line1, = ax.plot(x, z_upwind, '-', c='purple', label='Upwind (z)')
    line2, = ax.plot(x, z_lax_wendroff, '-', c='orange', label='Lax-Wendroff (z)')
    line3, = ax.plot(x, z_muscl_mc, '-', c='green', label='MUSCL w/ MC (z)')
    line4, = ax.plot(x, z_exact, '-.', c='blue', label='Exact Solution (z)', linewidth=1.5)

    ax.legend(loc="upper right")
    ax.grid()
    fig.tight_layout()

    fig.savefig(f"figures_stress/stress_CharZ_T_{T}_CFL_{CFL:.2e}_nx_{Nx:.1e}.pdf")


    # w0 = 0.5 * sigma0 + 0.25 * v0
    # z0 = -0.5*sigma0 + 0.25 * v0

    # Transform 
    sigma_upwind = (w_upwind - z_upwind)
    v_upwind = 2*(w_upwind + z_upwind)

    sigma_lax_wendroff = (w_lax_wendroff - z_lax_wendroff)
    v_lax_wendroff = 2*(w_lax_wendroff + z_lax_wendroff)

    sigma_muscl_mc = (w_muscl_mc - z_muscl_mc)
    v_muscl_mc = 2 * (w_muscl_mc + z_muscl_mc)

    # Exact solutions for sigma and v
    sigma_exact = (w_exact - z_exact)
    v_exact = 2*(w_exact + z_exact)


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('sigma')
    ax.set_title(f'Stress Equations, sigma, CFL = {CFL:.2f}')

    line0, = ax.plot(x, sigma0, '-.', c='blue', label='Initial Condition (sigma)', linewidth=1.5, alpha=0.2)
    line1, = ax.plot(x, sigma_upwind, '-', c='purple', label='Upwind (sigma)')
    line2, = ax.plot(x, sigma_lax_wendroff, '-', c='orange', label='Lax-Wendroff (sigma)')
    line3, = ax.plot(x, sigma_muscl_mc, '-', c='green', label='MUSCL w/ MC (sigma)')
    line4, = ax.plot(x, sigma_exact, '-.', c='blue', label='Exact Solution (sigma)', linewidth=1.5)

    ax.legend(loc="upper right")
    ax.grid()
    fig.tight_layout()

    fig.savefig(f"figures_stress/stress_SIGMA_T_{T}_CFL_{CFL:.2e}_nx_{Nx:.1e}.pdf")

    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title(f'Stress Equations, v , CFL = {CFL:.2f}')

    line0, = ax.plot(x, v_upwind, '-.', c='blue', label='Initial Condition (v)', linewidth=1.5, alpha=0.2)
    line1, = ax.plot(x, v_upwind, '-', c='purple', label='Upwind (v)')
    line2, = ax.plot(x, v_lax_wendroff, '-', c='orange', label='Lax-Wendroff (v)')
    line3, = ax.plot(x, v_muscl_mc, '-', c='green', label='MUSCL w/ MC (v)')
    line4, = ax.plot(x, v_exact, '-.', c='blue', label='Exact Solution (v)', linewidth=1.5)

    ax.legend(loc="upper right")
    ax.grid()
    fig.tight_layout()

    fig.savefig(f"figures_stress/stress_V_T_{T}_CFL_{CFL:.2e}_nx_{Nx:.1e}.pdf")


def main():
    
    import itertools as it

    CFL_list = [0.01,0.1,0.5,0.9,1.1]
    T_list = [1, 2, 5]
    Nx_list = [500]

    configs = list(it.product(CFL_list,T_list,Nx_list))
    
    for CFL, T, Nx in configs:
        print(f"Now Running stress for CFL = {CFL}, T = {T}, Nx = {Nx}")
        
        run_stress(T,CFL,Nx)
        print("Finished!")
        print()
    
if __name__ == '__main__':
    main()