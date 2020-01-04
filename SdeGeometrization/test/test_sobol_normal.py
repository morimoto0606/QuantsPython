import sys
sys.path.append('../')

import unittest
import numpy as np
import sobol_normal
import numpy.random

class TestStochasticLift(unittest.TestCase):
    def setUp(self):
        self.mat = np.array([[1,2,3,4],[5,6,7,8]])

    def test_reshape(self):
        self.assertTupleEqual((2,4), self.mat.shape)
        actual = np.reshape(self.mat, (2,2,2))
        expected = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
        self.assertTrue((expected == actual).all())
        print(np.transpose(self.mat))

    def test_sobol_normal(self):
        no_path = 4
        no_step = 3
        no_state = 4
        actual = sobol_normal.generate_path(no_path, no_step, no_state)
        self.assertTupleEqual((no_path,no_step,no_state), actual.shape)

    def test_expectation(self):
        no_path = 1000
        no_step = 2
        no_state = 2
        sobol_path = sobol_normal.generate_path(no_path, no_step, no_state)
        pseud_path = np.reshape(np.random.normal(0,1,no_path*no_step*no_state), (no_path, no_step, no_state))
        def payoff(x):
            return np.array([np.sum(p) for p in x])
        pv_sobol = np.mean(payoff(sobol_path))
        pv_pseud = np.mean(payoff(pseud_path))

        print("pv_sobol, pv_pseud", pv_sobol, pv_pseud)





if __name__ == "__main__":
    unittest.main()