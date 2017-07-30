import math
import numpy as np
from scipy.stats import norm


def stochastic_mesh(t_start, t_end, path_start, path_end, payoff):
    """
    stochastic mesh continuous function
    :param t_start: t
    :param t_end: T
    :param path_start: X_t
    :param path_end: X_T
    :param payoff: f(X_T)
    :return: (Q_t,T f)(x)
    """
    def c(x):
        sigma = math.sqrt(t_end - t_start)
        z1 = (path_end - path_start) / sigma
        z2 = (path_end - x) / sigma
        p1 = norm.pdf(z1)
        p2 = norm.pdf(z2) * payoff
        return np.sum(p2 / p1) / p1.size
    return c

def lsm(t_start, t_end, path_start, payoff, degree):
    """
    LSM continuous function
    :param t_start: t
    :param t_end: T
    :param path_start: X_t
    :param payoff: F(X_T)
    :param degree: degree of polynomial for regression
    :return: (Q_t,T f)(x)
    """
    def c(x):
        coefficient = np.polyfit(path_start, payoff, degree)
        return np.polyval(coefficient, x)
    return c


def generate_path(s0, dt, path_num, grid, r, sigma):
    """
    generate path matrix(grid_num + 1, path_num)
    :param s0: initial value
    :param dt: time grid size
    :param path_num: number of path
    :param grid_num: number of grid t1 < ... < tn
    :param r: interest rate
    :param sigma: volatility
    :return: path matrix on t0 = 0 < t1 < ... < tn
    """
    dt = np.diff(grid)[:, np.newaxis]
    print('grid', grid)
    print('dt', dt)
    print('size', dt.size)
    dw = np.sqrt(dt) * np.random.standard_normal((dt.size, path_num))
    print('dw', dw)
    d_log_s = (r - 0.5 * sigma * sigma) * dt + sigma * dw
    # add zero vector
    d_log_s = np.vstack((np.zeros((1, path_num)), d_log_s))
    print('dlogs', d_log_s)
    s = s0 * np.exp(np.cumsum(d_log_s, axis=0))
    print('s', s)
    return s


 
def f(x):
    return 2 * x


def g(x):
    return 3 * x


def integrate(phi):
    z = phi(1)
    return z

from functools import partial
if __name__ == '__main__':
    r = 0.1
    sigma = 0.3
    s0 = 100
    payment_num = 1
    grid_num = 5
    path_num = 3
    
    payment_grid = np.linspace(0, 10, payment_num + 1)
    calculation_grid = np.linspace(0, 10, grid_num + 1)
    print(payment_grid)
    print(calculation_grid)
    
    s = generate_path(s0=s0, dt=0.1, path_num=path_num, grid=calculation_grid, r=r, sigma=sigma)
   
    z = integrate(f)
    print(z)
    z = integrate(g)
    print(z)

    hoge = partial(f, x=2)
    print(hoge())
    t0 = 0
    t1 = 1
    x0 = np.array([0, 0])
    x1 = np.array([1, 2])
    f_x = np.array([2, 4])

    s = stochastic_mesh(t0, t1, x0, x1, f_x)

    x = 0.2
    y = s(x)
    print(y)
 
    print('hello world')

