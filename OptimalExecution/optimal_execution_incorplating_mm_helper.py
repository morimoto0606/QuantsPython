import numpy as np
import numpy as np
from scipy.optimize import  bisect


def hjb_solver(t, dt, kappa, xi, phi, q, qAC, lamb, alpha):
    Ndt = t.shape[0]
    Nq = q.shape[0]

    omega = np.full((Nq, Ndt), np.nan)
    # Terminal conditions for all q
    omega[:, omega.shape[1]-1] = np.exp(-kappa * (xi + alpha * q) * q)

    # Boundary conditions along q = 0
    omega[0, :] = np.exp(-kappa * phi * (np.sum(np.power(qAC, 2) * dt) - np.cumsum(np.power(qAC, 2) * dt)))
    exe = np.zeros((Nq, Ndt))
    exe[:, exe.shape[1]-1] = 1
    for k in range(Ndt - 2, -1, -1):
        
        # Solve the HJB in the continuation region from t+dt to t
        omega[1:omega.shape[0], k] \
            = omega[1:omega.shape[0], k+1] \
                + dt * (-kappa * phi * np.multiply(np.power((q[1:q.shape[0]] - qAC[k+1]), 2), omega[1:omega.shape[0], k+1]) +
                                           np.exp(-1) * lamb * omega[0:(omega.shape[0]-1), k+1])

        idx_exe = np.insert(np.exp(-kappa * xi) * omega[0:(omega.shape[0]-1), k] > omega[1:omega.shape[0], k], 0, 0)
        exe[:, k] = idx_exe
        omega[1:omega.shape[0], k] = np.maximum(np.exp(-kappa * xi) * omega[0:(omega.shape[0]-1), k], omega[1:omega.shape[0], k])
    return omega, exe


def bisect_check(f, a, b, xtol, dx):
    if f(a) * f(b) < 0:
        return bisect(f, a=a, b=b, xtol=xtol)
    elif abs(f(a)) > abs(f(b)):
        while abs(f(b)) > xtol:
            b = b + dx
        return b
    else:
        while abs(f(a)) > xtol:
            a = a - dx
        return a
    

def derivative(f, x, dx=1e-6):
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def hjb_solver_mm(t, dt, kappa, xi, phi, q, qAC, lambL, lambI, alpha, alphaL, alphaM, betaM):
    Ndt = t.shape[0]
 
    Ndt = t.shape[0]
    Nq = q.shape[0]

    h = np.full((Nq, Ndt), np.nan)
    delta_i = np.full((Nq, Ndt), np.nan)
    delta_l = np.full((Nq, Ndt), np.nan)
    zeta = np.full((Nq, Ndt), 1)
    h[:, h.shape[1]-1] = -q * (xi + alpha * q )

    # Boundary conditions along q = 0
    h[0, :] = -phi * (np.sum(np.power(qAC, 2) * dt) - np.cumsum(np.power(qAC, 2) * dt))
    exe = np.zeros((Nq, Ndt))
    exe[:, exe.shape[1]-1] = 1

    for k in range(Ndt - 2, -1, -1):
        y_l = 0
        y_i = 0
        for i in range(1, h.shape[0]):
            y_i = lambI / kappa * np.exp(-1 - kappa * (h[i, k+1] - h[i-1, k+1]))
            limitorder = lambda x: (x -alphaL * lambL * np.exp(-kappa*x) + h[i-1, k+1] - h[i, k+1]) * lambL * np.exp(-kappa* x)
            objective = lambda x: derivative(limitorder, x, dx=1e-6)
            
            x_l = bisect_check(f = objective, a=-0.5, b= 0.5, xtol=1e-4, dx=0.001)
            y_l = limitorder(x_l)
            h[i, k] = h[i, k+1] + dt * (-phi * (q[i] - qAC[k+1])**2 + y_l + y_i)
            delta_l[i, k] = x_l
            delta_i[i, k] = 1/kappa + h[i, k+1] - h[i-1, k+1] 
            morder = [h[i-z, k] - h[i, k] - (xi * z + alphaM * np.power(z, betaM)) for z in range(0, i+1)]
            zeta[i, k] = np.argmax(morder)
 
        idx_exe = np.zeros(h.shape[0], dtype=int)
        for i in range(1, h.shape[0]):
            if h[i-zeta[i, k], k] - h[i, k] - (xi * zeta[i, k] + alphaM * np.power(zeta[i, k], betaM)) > 0:
                idx_exe[i] = 1

        exe[:, k] = idx_exe
        for i in range(1,h.shape[0]):
            h[i, k] = np.maximum(h[i-zeta[i, k], k] - (xi * zeta[i, k] + alphaM * np.power(zeta[i, k], betaM)), h[i, k])

    return h, exe, delta_l, delta_i, zeta



def find_opt_t(exe, t):
    Nq = exe.shape[0]
    t_opt = np.full(Nq, np.nan)
    for k in range(0, Nq, 1):
        idx = np.where(exe[k, :] == 1)[0]
        if idx.shape[0] > 0:
            t_opt[k] = t[idx[0]]
    return t_opt