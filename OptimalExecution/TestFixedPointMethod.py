import unittest
import numpy as np
from fixed_point_method import FixedPointMethod 

class TestFixedPointMethod(unittest.TestCase):
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
         obj = FixedPointMethod(
            self.f, 
            self.df, 
            self.G, 
            self.X, 
            self.N, 
            self.T, 
            self.sigma, 
            self.lamda)

        x0 = np.full(self.N, self.X / self.T)
        res = obj.solve(x0, 2)
        print(res)

if __name__ == '__main__':
    unittest.main()
