import unittest
import numpy
from unittest.mock import Mock


class TestLsm(unittest.TestCase):
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

    def test_create_bs_generator(self):
        actual = lsm.create_bs_path_generator(self._spot, self._r, self._sigma)
        actual(self._dt, self._dw)

    #def testEasyClacFail(self):
     #   self.assertEqual((0, 4 + 2, "wrong calclation"))

    def tearDown(self):
        print('tearDown')
