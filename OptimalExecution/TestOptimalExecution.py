import unittest
import numpy as np
from OptimalExecution import OptimalExecution

class TestOptimalExecution(unittest.TestCase):
    def setUp(self) -> None:
        def G(t):
            if t > 0:
                return np.power(t, -2) 
            else:
                return 0.0
        self.G = G
        def f(x):
            return x
        self.f = f

        def df(x):
            epsilon = 0.0001
            return (f(x + epsilon) - f(x - epsilon)) /(2.0 * epsilon)
            #return -0.5 / f(x)

        self.df = df

        self.X = 100 * 1e6
        self.N = 10
        self.T = 1.0
        self.sigma = 0.1
        self.lamda = 0.0


    def test_optimize(self):
        opt = OptimalExecution(
            self.f, 
            self.df, 
            self.G, 
            self.X, 
            self.N, 
            self.T, 
            self.sigma, 
            self.lamda)

        method = 'Nelder-Mead'
        x0 = np.full(self.N + 1, self.X / self.T)
        x0[-1] = 1.0
        res = opt.optimize(x0=x0, method=method)
        print(res)

if __name__ == '__main__':
    unittest.main()
