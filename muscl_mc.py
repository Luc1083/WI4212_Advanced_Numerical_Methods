import numpy as np
import matplotlib.pyplot as plt

def flux_limiter(r, type="MC"):
    if type == "minmod":
        return max(0, min(1, r))
    elif type == "vanleer":
        return (r + abs(r)) / (1 + abs(r))
    elif type == "MC":
        return max(0, min(2*r, (1 + r)/2, 2))
    else:
        raise ValueError("Unknown limiter type")

def muscl_step(u, dt, dx, limiter_type="MC", alpha=1.0):
    N = len(u)
    u_new = np.zeros_like(u)
    
    # Compute slopes and apply flux limiter
    delta_u = np.zeros_like(u)
    r = np.zeros_like(u)
    
    for i in range(1, N-1):
        if u[i+1] != u[i]:
            r[i] = (u[i] - u[i-1]) / (u[i+1] - u[i])
        else:
            r[i] = 0
        phi = flux_limiter(r[i], limiter_type)
        delta_u[i] = 0.5 * phi * (u[i+1] - u[i-1])
    
    # Reconstruct left and right states
    u_L = np.zeros(N)
    u_R = np.zeros(N)
    

    u_L[2:] = u[1:-1] + 0.5 * delta_u[1:-1]
    u_R[1:-1] = u[1:-1] - 0.5 * delta_u[1:-1]
    
    # Compute numerical fluxes using Lax-Friedrichs
    F = np.zeros(N+1)

    F[1:] = 0.5 * (u_L[:] + u_R[:]) - 0.5 * alpha * (u_R[:] - u_L[:])
    
    # Update solution

    u_new[1:] = u[1:] - dt/dx[1:] * (F[2:] - F[1:-1])
    
    return u_new

# Example usage with a specific flux function and initial conditions
Nx = 500
dt = 0.0001
timesteps = 10000

x = np.linspace(0, 3, Nx)
dx = np.diff(x)
dx = np.append(dx, dx[-1])  # Extend the last dx for boundary conditions

print(1*dt/dx[0])

# Initial conditions and parameters
u_initial = np.zeros_like(x)
u_initial[np.where(x <= 0.5)] = 1.0

# Time integration loop
u = u_initial.copy()
for t in range(timesteps):
    u = muscl_step(u, dt, dx, limiter_type="MC", alpha=1.0)

plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u')
plt.title('MUSCL Scheme with Lax-Friedrichs Flux')
plt.show()
