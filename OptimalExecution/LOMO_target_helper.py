import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 10,
          'figure.figsize': (8, 4),
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'xtick.labelsize': 15,
         'ytick.labelsize': 15}
pylab.rcParams.update(params)
font = {'family': 'serif',
        'style': 'italic',
        'weight': 1,
        'size': 16,
        }

import numpy as np

def nan_to_num(x, nan):
    """Change the NaNs in a numpy array to the desired values.
    :param x: a numpy array
    :param nan: desired value
    :return: a deep copy of x with changed values
    """
    y = np.copy(x)
    y[np.isnan(y)] = nan
    return y

def AC_solver(phiAC, aAC, tau, T, Nq):
    gamma = np.sqrt(phiAC / aAC)
    qAC = Nq * np.divide((np.exp(gamma * tau) - np.exp(-gamma * tau)), 
                         np.exp(gamma * T) - np.exp(-gamma * T))
    return qAC

def plot_curve(x, y, xlab=None, ylab=None , title=None):
    plt.plot(x, y)
    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if title is not None:
        plt.title(title)
    plt.show()

def hjb_solver(t, dt, kappa, xi, phi, q, qAC, lamb):
    Ndt = t.shape[0]
    Nq = q.shape[0]

    omega = np.full((Nq, Ndt), np.NaN)
    # Terminal conditions for all q
    omega[:, omega.shape[1]-1] = np.exp(-kappa * xi * q)

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

def hjb_solver_loop(t, dt, kappa, xi, phi, q, qAC, lamb):
    Ndt = t.shape[0]
    Nq = q.shape[0]

    h = np.full((Nq, Ndt), np.NaN)
    delta = np.full((Nq, Ndt), np.NaN)
    h[:, h.shape[1]-1] = -xi * q

    # Boundary conditions along q = 0
    h[0, :] = -phi * (np.sum(np.power(qAC, 2) * dt) - np.cumsum(np.power(qAC, 2) * dt))
    exe = np.zeros((Nq, Ndt))
    exe[:, exe.shape[1]-1] = 1


    h = np.full((Nq, Ndt), np.NaN)
    # Terminal conditions for all q
    h[:, h.shape[1]-1] = -xi * q

    # Boundary conditions along q = 0
    h[0, :] = -phi * (np.sum(np.power(qAC, 2) * dt) - np.cumsum(np.power(qAC, 2) * dt))
    exe = np.zeros((Nq, Ndt))
    exe[:, exe.shape[1]-1] = 1
    for k in range(Ndt - 2, -1, -1):
        for i in range(1, h.shape[0]):
            y = np.exp(-1) * lamb / kappa * np.exp(-kappa * (h[i, k+1] - h[i-1, k+1]))
            h[i, k] = h[i, k+1] + dt * (-phi * np.power((q[i] - qAC[k+1]), 2) + y)

        idx_exe = np.insert(h[0:(h.shape[0]-1), k] - h[1:(h.shape[0]), k] - xi > 0, 0, 0)
        exe[:, k] = idx_exe
        h[1:h.shape[0], k] = np.maximum(h[0:(h.shape[0]-1), k] - xi, h[1:h.shape[0], k])

    return h, exe




import numpy as np
from scipy.optimize import minimize, newton, brentq, bisect
from scipy.misc import derivative
import pandas as pd

def my_bisect(f, a, b, xtol, dx):
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
        

 

def hjb_solver_optimize(t, dt, kappa, xi, phi, q, qAC, lamb):
    Ndt = t.shape[0]
    Nq = q.shape[0]

    h = np.full((Nq, Ndt), np.NaN)
    delta = np.full((Nq, Ndt), np.NaN)
    h[:, h.shape[1]-1] = -xi * q

    # Boundary conditions along q = 0
    h[0, :] = -phi * (np.sum(np.power(qAC, 2) * dt) - np.cumsum(np.power(qAC, 2) * dt))
    exe = np.zeros((Nq, Ndt))
    exe[:, exe.shape[1]-1] = 1

    
    df = pd.DataFrame() 
    for k in range(Ndt - 2, -1, -1):
        # Solve the HJB in the continuation region from t+dt to t
        #omega[1:omega.shape[0], k] \
        #    = omega[1:omega.shape[0], k+1] \
        #        + dt * (-kappa * phi * np.multiply(np.power((q[1:q.shape[0]] - qAC[k+1]), 2), omega[1:omega.shape[0], k+1]) + y)
        y = 0
        for i in range(1, h.shape[0]):
            
            limitorder = lambda delta: (delta + h[i-1, k+1] - h[i, k+1]) * lamb * np.exp(-kappa* delta)
            objective = lambda delta: derivative(limitorder, delta, 1e-6)
            #if k == 5999:
            #    if i == 1:
            #        func = [limitorder(0.001 * d) for d in range(21, 100)]
            #        dfunc = [objective(0.001 * d) for d in range(21, 100)]
            #        df = pd.DataFrame({'delta': list(range(21,100)), 'func': func, 'dfunc': dfunc})

            x = bisect(f = objective, a=-0.5, b=0.5, xtol=1e-4)
            #x = my_bisect(f = objective, a=-0.1, b= 0.2, xtol=1e-4, dx=0.001)
            #print(k, i, x)
            y = limitorder(x)
            h[i, k] = h[i, k+1] + dt * (-phi * (q[i] - qAC[k+1])**2 + y)
            delta[i, k] = x
        idx_exe = np.insert(h[0:(h.shape[0]-1), k] - h[1:(h.shape[0]), k] - xi > 0, 0, 0)
        exe[:, k] = idx_exe
        h[1:h.shape[0], k] = np.maximum(h[0:(h.shape[0]-1), k] - xi, h[1:h.shape[0], k])

    return h, exe, df, delta


def hjb_solver_internal_flow(t, dt, kappa, xi, phi, q, qAC, lambL, lambI, alphaL, alphaM):
    """
    additional parameters
    - lambL: intensity for passive order
    - lambI: intensity for internal order
    - alphaL: coefficient for market impact against intensity for Passive order
    - alphaM: coefficient for market impact against intensity for Agressive order
    """
    Ndt = t.shape[0]
 
    Ndt = t.shape[0]
    Nq = q.shape[0]

    h = np.full((Nq, Ndt), np.NaN)
    delta_i = np.full((Nq, Ndt), np.NaN)
    delta_l = np.full((Nq, Ndt), np.NaN)
    zeta = np.full((Nq, Ndt), 1)
    h[:, h.shape[1]-1] = -xi * q

    # Boundary conditions along q = 0
    h[0, :] = -phi * (np.sum(np.power(qAC, 2) * dt) - np.cumsum(np.power(qAC, 2) * dt))
    exe = np.zeros((Nq, Ndt))
    exe[:, exe.shape[1]-1] = 1

    
    df = pd.DataFrame() 
    for k in range(Ndt - 2, -1, -1):
        y_l = 0
        y_i = 0
        for i in range(1, h.shape[0]):
            y_i = np.exp(-1) * lambI / kappa * np.exp(-kappa * (h[i, k+1] - h[i-1, k+1]))
            limitorder = lambda x: (x -alphaL * lambL * np.exp(-kappa*x) + h[i-1, k+1] - h[i, k+1]) * lambL * np.exp(-kappa* x)
            objective = lambda x: derivative(limitorder, x, 1e-6)
            
            x_l = my_bisect(f = objective, a=-0.5, b= 0.5, xtol=1e-4, dx=0.001)
            #print(k, i, x)
            y_l = limitorder(x_l)

            h[i, k] = h[i, k+1] + dt * (-phi * (q[i] - qAC[k+1])**2 + y_l + y_i)
            delta_l[i, k] = x_l
            delta_i[i, k] = 1/kappa + h[i, k+1] - h[i-1, k+1] 
            morder = [h[i-z, k] - h[i, k] - (xi + alphaM) * z for z in (1, i+1)]
            zeta[i, k] = np.argmax(morder) 
 
        idx_exe = np.insert(h[0:(h.shape[0]-1), k] - h[1:(h.shape[0]), k] - (xi + alphaM) * zeta[1:h.shape[0], k] > 0, 0, 0)
        exe[:, k] = idx_exe
        for i in range(1, h.shape[0]):
            h[i, k] = np.maximum(h[i-zeta[i, k], k] - (xi + alphaM) * zeta[i, k], h[i, k])
    return h, exe, df, delta_l, delta_i, zeta

def hjb_solver_internal_flow_power(t, dt, kappa, xi, phi, q, qAC, lambL, lambI, alphaL, alphaM, betaM, gammaM):
    """
    additional parameters
    - lambL: intensity for passive order
    - lambI: intensity for internal order
    - alphaL: coefficient for market impact against intensity for Passive order
    - alphaM: coefficient for market impact against intensity for Agressive order
    """
    Ndt = t.shape[0]
 
    Ndt = t.shape[0]
    Nq = q.shape[0]

    h = np.full((Nq, Ndt), np.NaN)
    delta_i = np.full((Nq, Ndt), np.NaN)
    delta_l = np.full((Nq, Ndt), np.NaN)
    zeta = np.full((Nq, Ndt), 1)
    h[:, h.shape[1]-1] = -xi * q

    # Boundary conditions along q = 0
    h[0, :] = -phi * (np.sum(np.power(qAC, 2) * dt) - np.cumsum(np.power(qAC, 2) * dt))
    exe = np.zeros((Nq, Ndt))
    exe[:, exe.shape[1]-1] = 1

    
    df = pd.DataFrame() 
    for k in range(Ndt - 2, -1, -1):
        y_l = 0
        y_i = 0
        for i in range(1, h.shape[0]):
            y_i = np.exp(-1) * lambI / kappa * np.exp(-kappa * (h[i, k+1] - h[i-1, k+1]))
            limitorder = lambda x: (x -alphaL * lambL * np.exp(-kappa*x) + h[i-1, k+1] - h[i, k+1]) * lambL * np.exp(-kappa* x)
            objective = lambda x: derivative(limitorder, x, 1e-6)
            
            x_l = my_bisect(f = objective, a=-0.5, b= 0.5, xtol=1e-4, dx=0.001)
            #print(k, i, x)
            y_l = limitorder(x_l)
            h[i, k] = h[i, k+1] + dt * (-phi * (q[i] - qAC[k+1])**2 + y_l + y_i)
            delta_l[i, k] = x_l
            delta_i[i, k] = 1/kappa + h[i, k+1] - h[i-1, k+1] 
            morder = [h[i-z, k] - h[i, k] - (xi * z + alphaM * (betaM + np.power(z, gammaM))) for z in range(0, i+1)]
            if (k==Ndt-2 and i==9):
                df = pd.DataFrame({'mordr': morder})
            zeta[i, k] = np.argmax(morder)
            #print(k, i, len(morder), morder, zeta[i, k])
 
        idx_exe = np.insert(h[0:(h.shape[0]-1), k] - h[1:(h.shape[0]), k] - (xi * zeta[1:h.shape[0], k] + [alphaM * (betaM + np.power(z, gammaM)) for z in zeta[1:h.shape[0], k]])  > 0, 0, 0)
        #print('idex_exe', idx_exe)
        exe[:, k] = idx_exe
        for i in range(1,h.shape[0]):
            h[i, k] = np.maximum(h[i-zeta[i, k], k] - (xi * zeta[i, k] + alphaM * (betaM + np.power(zeta[i, k], gammaM))), h[i, k])
    return h, exe, df, delta_l, delta_i, zeta


def hjb_solver_maket_impact(t, dt, kappa, xi, phi, q, qAC, lamb_P, lamb_I, alpha_P):
    """
    additional parameters
    - lambP: intensity for passive order
    - lambI: intensity for internal order
    - alpha_P: coefficient for market impact against intensity for Passive order
    - alpha_A: coefficient for market impact against intensity for Agressive order
    """
    Ndt = t.shape[0]
    Nq = q.shape[0]

    omega = np.full((Nq, Ndt), np.NaN)
    # Terminal conditions for all q
    omega[:, omega.shape[1]-1] = np.exp(-kappa * xi * q)

    # Boundary conditions along q = 0
    omega[0, :] = np.exp(-kappa * phi * (np.sum(np.power(qAC, 2) * dt) - np.cumsum(np.power(qAC, 2) * dt)))
    exe = np.zeros((Nq, Ndt))
    exe[:, exe.shape[1]-1] = 1
    for k in range(Ndt - 2, -1, -1):
        tau = (Ndt - k) * dt
        # Solve the HJB in the continuation region from t+dt to t
        omega[1:omega.shape[0], k] \
            = omega[1:omega.shape[0], k+1] \
                + dt * (-kappa * phi * np.multiply(np.power((q[1:q.shape[0]] - qAC[k+1]), 2), omega[1:omega.shape[0], k+1]) +
                        np.exp(-(1 + kappa * alpha_P * lamb_P)) * lamb_P + np.exp(-1) * lamb_I) * omega[0:(omega.shape[0]-1), k+1]
        idx_exe = np.insert(np.exp(-kappa * xi) * omega[0:(omega.shape[0]-1), k] > omega[1:omega.shape[0], k], 0, 0)
        exe[:, k] = idx_exe
        omega[1:omega.shape[0], k] = np.maximum(np.exp(-kappa * xi) * omega[0:(omega.shape[0]-1), k], omega[1:omega.shape[0], k])

    return omega, exe


def find_opt_t(exe, t):
    Nq = exe.shape[0]
    t_opt = np.full(Nq, np.NaN)
    for k in range(0, Nq, 1):
        idx = np.where(exe[k, :] == 1)[0]
        if idx.shape[0] > 0:
            t_opt[k] = t[idx[0]]
    return t_opt

def find_opt_t_q(exe, t, zeta):
    Nq = exe.shape[0]
    t_opt = np.full(Nq, np.NaN)
    q_opt = np.full(Nq, 0)
    for k in range(0, Nq, 1):
        idx = np.where(exe[k, :] == 1)[0]
        if idx.shape[0] > 0:
            t_opt[k] = t[idx[0]]
            q_opt[k] = k - zeta[k, idx[0]]
    return t_opt, q_opt



def plot_topt(t_opt, q, qAC, t):
    # Plot q vs t_opt
    line1, = plt.plot(t_opt, q, 'bo')
    # Plot qAC vs t
    line2, = plt.plot(t, qAC, 'k-', linewidth=2)
    plt.legend([line1, line2], ["Execute MO", "Target Schedule"])
    plt.ylabel(r'Inventory ($q_t$)')
    plt.xlabel(r'Time ($sec$)')
    plt.title(r'Target and Behind Schedule Times')
    plt.show()

def find_delta(kappa, omega, Nq, Ndt):
    delta = np.full((Nq + 1, Ndt + 1), np.NaN)
    delta[1:delta.shape[0], :] = 1 / kappa + 1 / kappa * np.log(np.divide(omega[1:omega.shape[0], :], omega[0:(omega.shape[0]-1), :]))
    return delta

def plot_multi_lines(t, y, xlab=None, ylab=None, title=None):
    color_idx = np.linspace(0, 1, y.shape[0])
    for i, line in zip(color_idx, range(0, y.shape[0], 1)):
        plt.plot(t, y[line, :], color=plt.cm.rainbow(i))
    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if title is not None:
        plt.title(title)
    plt.show()


def generate_simulations(Nsims, s0, Ndt, Nq, dt, delta, lamb, kappa, sigma, xi, t_opt, t):
    
    Qpath = np.full((Nsims, Ndt + 1), np.NaN)
    # Starting inventory
    Qpath[:, 0] = Nq
    Xpath = np.full((Nsims, Ndt + 1), np.NaN)
    # Starting cash
    Xpath[:, 0] = 0
    Spath = np.full((Nsims, Ndt + 1), np.NaN)
    # Starting Mid-price
    Spath[:, 0] = s0
    deltaPath = np.full((Nsims, Ndt + 1), np.NaN)
    isFilled = np.full((Nsims, Ndt + 1), np.NaN)
    isMO = np.zeros((Nsims, Ndt + 1))
    pricePerShare = np.full((Nsims, Ndt + 1), np.NaN)

    mu = 0
    for k in range(0, Ndt, 1):
        idx = Qpath[:, k] > 0

        deltaPath[idx, k] = delta[Qpath[idx, k].astype(int) - 1, k]
        isFilled[idx, k] = np.random.rand(np.sum(idx)) < nan_to_num(lamb * np.exp(-kappa * deltaPath[idx, k]) * dt, np.Inf)

        deltaPath[(1 - idx).astype(bool), k] = np.NaN
        isFilled[(1 - idx).astype(bool), k] = 0

        idx = np.logical_and((k + 1) * dt > t_opt[Qpath[:, k].astype(int)], Qpath[:, k] > 0)
        isMO[idx, k] = 1
        isFilled[:, k] = np.logical_and(1 - isMO[:, k], isFilled[:, k])

        idx = Qpath[:, k] > 0
        Xpath[idx, k + 1] = Xpath[idx, k] + np.multiply(isFilled[idx, k], Spath[idx, k] + deltaPath[idx, k]) + \
                            np.multiply(isMO[idx, k], Spath[idx, k] - xi)
        Xpath[(1 - idx).astype(bool), k + 1] = Xpath[(1 - idx).astype(bool), k]

        Qpath[:, k + 1] = Qpath[:, k] - isMO[:, k] - isFilled[:, k]
        Spath[:, k + 1] = Spath[:, k] + mu * dt + sigma * np.sqrt(dt) * np.random.randn(Nsims)

        idx = np.logical_and(Qpath[:, k + 1] < Nq, Qpath[:, k + 1] > 0)

        pricePerShare[idx, k + 1] = np.divide(Xpath[idx, k + 1], Nq - Qpath[idx, k + 1])

        idx = Qpath[:, k + 1] == 0
        pricePerShare[idx, k + 1] = pricePerShare[idx, k]

    Xpath[:, Xpath.shape[1] - 1] += np.multiply(Qpath[:, Qpath.shape[1] - 1], Spath[:, Spath.shape[1] - 1] - xi)
    Qpath[:, Qpath.shape[1] - 1] = 0

    idx = (Nq - Qpath[:, Qpath.shape[1] - 1]) > 0
    pricePerShare[idx, pricePerShare.shape[1] - 1] = np.divide(Xpath[idx, Xpath.shape[1] - 1],
                                                               Nq - Qpath[idx, Qpath.shape[1] - 1])

    twap = np.divide(np.cumsum(Spath[:, 0:(Spath.shape[1] - 1)] * dt, axis=1), t[1:t.shape[0]]) - xi
    twap = np.concatenate((Spath[:, 0][:, np.newaxis], twap), axis=1)

    return deltaPath, Qpath, isMO, Xpath, Spath, pricePerShare, twap


def plot_inventory(is_MO, Qpath, t):
    color_idx = np.linspace(0, 1, Qpath.shape[0])
    for i, line in zip(color_idx, range(0, Qpath.shape[0], 1)):
        plt.step(t, Qpath[line, :], color=plt.cm.rainbow(i))
        specific_qpath = Qpath[line, :]
        thisisMO = is_MO[line, :].astype(bool)
        plt.plot(t[thisisMO], specific_qpath[thisisMO], 'bo')
    plt.xlabel(r'Time ($sec$)')
    plt.ylabel(r'Inventory ($Q^*_t$)')
    plt.title(r'Inventory Sample Paths')
    plt.show()


def plot_multi_steps(t, y, xlab=None, ylab=None, title=None):
    
    color_idx = np.linspace(0, 1, y.shape[0])
    for i, line in zip(color_idx, range(0, y.shape[0], 1)):
        plt.step(t, y[line, :], color=plt.cm.rainbow(i))
    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if title is not None:    
        plt.title(title)
    plt.show()


def plot_price_per_share(t, pricePerShare, twap):
    color_idx = np.linspace(0, 1, pricePerShare.shape[0])
    for i, line in zip(color_idx, range(0, pricePerShare.shape[0], 1)):
        plt.plot(t, pricePerShare[line, :], color=plt.cm.rainbow(i), linestyle='-')
        plt.plot(t, twap[line, :], color=plt.cm.rainbow(i), linestyle='--')
    plt.ylabel(r'Price / Share ($\frac{X_t}{Q_t}$)')
    plt.xlabel(r'Time ($sec$)')
    plt.show()


def plot_histogram(x, xlab=None, prob=None, bins= None):    
    
    if bins is None:
        counts = plt.hist(x[~np.isnan(x)])
    else:
        counts = plt.hist(x[~np.isnan(x)], bins=bins)
    
    if prob is not None:    
        color_idx = np.linspace(0, 1, prob.shape[0])        
        q = np.quantile(x[~np.isnan(x)], prob)        
        for i, vline in zip(color_idx, range(0, prob.shape[0], 1)):          
            plt.axvline(x=q[vline], ymin=0, ymax=np.max(counts[0]), linestyle='--', color=plt.cm.rainbow(i), label='quantile ' + str(prob[vline]))

    if xlab is not None:
        plt.xlabel(xlab)
    plt.ylabel(r'Frequency')
    plt.legend()    
    plt.show()


class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)


def plot_heat_map(t, q, myn_per_sim, meanq, medq, qAC, xlab=None, ylab=None, title=None):
    
    plt.tick_params(direction='in', bottom=True, top=True, left=True, right=True)

    x_cord, y_cord = np.meshgrid(t, q)
    cmap = plt.get_cmap('YlOrRd')
    plot = plt.contourf(x_cord, y_cord, myn_per_sim, 100, cmap=cmap, levels=np.linspace(myn_per_sim.min(), myn_per_sim.max(), 1000))
    fmt = FormatScalarFormatter("%.2f")
    plt.colorbar(plot, format=fmt)

    plt.plot(t, meanq, linestyle='-', color='black', label=r'mean $Q_t^*$')
    plt.plot(t, medq, linestyle='--', color='black', label=r'median $Q_t^*$')
    plt.plot(t, qAC, linestyle='-', color='blue', label=r'target $q_t$')
    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()
