from cmath import exp
import unittest
import numpy as np
from sympy import expint
import homotopy_method
import copy

from homotopy_method import HomotopyMethod
class TestHomotopyMethod(unittest.TestCase):
    def setUp(self):
        print("set Up")
        self.amount = 1
        self.vol = 0.01
        self.num_discretization = 100
        self.expiry = 10
        self.risk_aversion= 0.0
        self.beta=1
        self.gamma = 0.9
        self.alpha= lambda t: 0.0001 * (1-np.exp(-10*t))
        self.order = 2
        self.method = HomotopyMethod(
            amount=self.amount,
            vol=self.vol,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            risk_aversion=self.risk_aversion,
            expiry=self.expiry,
            num_discretization=self.num_discretization,
            order=self.order
        )
        print('success construct')

    def test_increment_v(self):
        v = np.zeros((1, self.num_discretization))
        self.method.get_increment_v(v, 1)


    def test_matrix(self):
        a = np.array([[1,2], [3,4]])
        b = np.array([5,6])
        print('np.dot', np.dot(a, b))
        print('@', a @ b)
    
    def test_df(self):
        beta = 0.5
        prod = np.prod([beta-i for i in range(3)]) 
        expected = beta * (beta-1) * (beta-2)
        print(prod)
        print(expected)
        self.assertEqual(prod, expected)
        self.assertEqual(1+1, 2)
        m = 4
        def power_prod(beta, k):
            x = [beta-i for i in range(k)]
            c = np.prod(x)
            return c

        funcs = [lambda x: np.power(x, beta) / np.prod([beta-i for i in range(k)]) for k in range(m)]
        print("power_prod", power_prod(beta, 3))
        print()
        self.assertEqual(funcs[3](1), 1/power_prod(beta, 3))


    def test_pow(self):
        def pow(x):
            return np.power(x, 2) 

        print(pow([2, 3.1]))



if __name__ == '__main__':
    unittest.main()