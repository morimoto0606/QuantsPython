from tkinter import W
from typing import List
import numpy as np
from scipy.optimize import minimize
import pandas as pd

class OptimalExecution:
    def __init__(self, f, df, G, X, N, T, sigma, lamda, smooth_penalty, alpha) -> None:
        self.f = f
        self.df = df
        self.X = X
        self.N = N
        self.T = T
        self.sigma = sigma
        self.lamda = lamda
        self.Delta = self.T / self.N
        self.t = np.arange(N) * self.Delta
        self.G = np.array([[G(np.abs(self.t[i] -self.t[j])) for j in range(N)] for i in range(N)])
        self.G = np.identity(N)
        self.diagG = np.diag(np.diag(self.G))
        self.G1 = np.tril(self.G) - self.diagG
        self.G2 = np.triu(self.G)
        self.G3 = np.array([[max(min(i-1, j), 0.0) for j in range(N)] for i in range(N)])
        self.mu = smooth_penalty
        alpha0 = alpha(self.t)
        self.alpha = np.cumsum(alpha0)
        self.basis_func = lambda t: np.array([np.power(t, i) for i in range(2)])
        self.grid = np.linspace(0, T, N)

    '''
    solve by optimizer
    '''
    def target(self, param: List[float]) -> List[float]:
        v = param[:-1]
        v[-1] = self.X / self.Delta - np.sum(v[:-1])
        c = param[-1]
        dfv = self.df(v)
        fv = self.f(v)
        G2 = np.array([dfv[i] * self.G2[i] for i in range(self.N)])
        phi = self.alpha + self.G1.dot(fv) + G2.dot(v) + 2.0 * self.lamda * self.sigma * self.G3.dot(v) * self.Delta - c
        diffv = np.diff(v)
        return phi.dot(phi) + self.mu * diffv.dot(diffv)


    def solve_by_opt(self, x0, method, constraints) -> List[float]:
        res = minimize(fun = self.target,
            x0 = x0,
            method = method,
            options={"disp":True},
            constraints=constraints)
        return res

    '''
    solve by lsm
    '''
    def get_v(self, coeffs: List[float])->List[float]:
        return coeffs.dot(self.basis_func(self.grid))
 
    def target_lsm(self, param: List[float]) -> List[float]:
        v = self.get_v(coeffs=param)
        dfv = self.df(v)
        fv = self.f(v)
        G2 = np.array([dfv[i] * self.G2[i] for i in range(self.N)])
        phi = self.alpha + self.G1.dot(fv) + G2.dot(v) + 2.0 * self.lamda * self.sigma * self.G3.dot(v) * self.Delta
        return max(phi) - min(phi)


    def solve_by_lsm(self, a0, method) -> List[float]:
        def cons_positive(x):
            v = self.get_v(x)
            return min(v)

        def cons_x(x):
            v = self.get_v(x)
            X1 = np.trapz(v, x=self.grid)
            return self.X - X1

        cons = (
            {'type': 'ineq', 'fun': cons_positive },
            {'type': 'eq', 'fun': cons_x},
            )
 
        res = minimize(fun = self.target_lsm,
            x0 = a0,
            method = method,
            options={"disp":True},
            constraints=cons)
        v = self.get_v(res.x)
        return v



def get_optimal_execution_curve(
    X: int,
    N: int,
    T: int,
    beta: float,
    gamma: float,
    sigma: float,
    lamda: float,
    smooth_penalty: float,
    numerical='opt',
    alpha=lambda t: 0):
    """
    f(x) = x^{beta}
    G(t) = t^{-gamma}
    """
    def f(x):
        return np.power(np.where(x > 0, x, 0), beta)
    def G(t):
        if t > 0:
            return np.power(t, -gamma)
        else:
            return 0

    def df(x):
        return np.power(np.where(x > 0, x, 0), beta-1) * beta

    opt = OptimalExecution(
        f,
        df,
        G,
        X,
        N,
        T,
        sigma,
        lamda,
        smooth_penalty,
        numerical='opt',
        alpha = lambda t: 0)

    if numerical == 'opt':
        cons = {'type': 'ineq', 'fun': lambda x: min(x[:-1])}
        method='SLSQP'
        x0=np.full(N+1, X/T)
        res=opt.solve_by_opt(x0=x0, method=method, constraints=cons)
        v=list(res.x)[:-1]
    elif numerical=='lsm':
        method='SLSQP'
        a0 = np.zeros(5)
        a0[0] = X / T
        v = opt.solve_by_lsm(a0=a0, method=method)


    else:
        assert('invalid method')
    
    cum_v = np.cumsum(v) * T / N
    v_twap = X / T
    v_ratio = np.array(v) / v_twap
    df = pd.DataFrame({'v': v_ratio, 'cum_v': cum_v})
    return df
