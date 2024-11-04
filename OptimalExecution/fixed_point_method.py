from typing import List
import numpy as np
from scipy.optimize import minimize


class FixedPointMethod:
    def __init__(self, f, df, G, X, N, T, sigma, lamda) -> None:
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
        self.diagG = np.diag(np.diag(self.G))
        self.G1 = np.tril(self.G) - self.diagG
        self.G2 = np.triu(self.G)
        self.G3 = np.array([[max(i-j-1.0, 0.0) for j in range(N)] for i in range(N)])



    def solve_lineq(self, v: List[float])->List[float]:
        dfv = self.df(v)
        G1 = np.array([dfv[j] * self.G1.T[j] for j in range(self.N)]).T
        G2 = np.array([dfv[i] * self.G2[i] for i in range(self.N)])
        A = G1 + G2 - 2.0 * self.lamda * (self.sigma ** 2) * self.G3
        b = 1.0 + G1.dot(v)
        x = np.linalg.solve(A, b)
        return x

    def solve(self, ini: List[float], num_iter: int) -> List[float]:
        v = ini
        for i in range(num_iter):
            v = self.solve_lineq(v)
        return v