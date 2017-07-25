import unittest
import lsm

class TestFiniteDifferencde(unittest.TestCase):
    def setUp(self):
        self._grid = []
        self._seed = 1
        for i in range(0, 3):
            self._grid.append(i)
        print(self._grid)
        self._spot = 1
        self._r = 0.1
        self._sigma = 0.3
        self._dt = [1.0, 1.0]
        self._dw = [ 1.62434536, -0.61175641]

    def test_generate_bm(self):
        actual = lsm.generate_bm(self._grid, self._seed)
        print(actual)

    def tearDown(self):
        print('finite_difference_test_finish')

    def assertTest(self):
        self.assertAlmostEquals(0.1, 0.1)