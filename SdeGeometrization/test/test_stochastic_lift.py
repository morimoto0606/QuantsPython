import sys
sys.path.append('../')

import unittest
import numpy as np
from runge_kutta import RungeKutta5
import tensorflow as tf
from stochastic_lift import StochasticLift
from vector_field import *
from runge_kutta import *
from numpy.linalg import inv

class TestStochasticLift(unittest.TestCase):
    def setUp(self):
        self.vec_field = Sabr(1.0, .4, .9, -0.7)
        self.stepsize = .25
        self.step = int(1.0 / self.stepsize)
        print(self.step)
        self.normal = np.array([0.3, 0.4]) 
        self.bm = np.sqrt(self.stepsize) * self.normal
        self.rk = RungeKutta5()
        self.rk_diff = RungeKutta5Differentiable()
        self.ini = np.array([1.0, 0.3])
        self.lift = StochasticLift(self.stepsize, self.rk, self.rk_diff, self.vec_field, self.ini)

    def test_evolveJacobiInv(self):
        d = self.vec_field.get_bm_size()
        lifted_ini = list(self.ini) + list(np.identity(d).flatten())
        idmatrix = np.identity(len(lifted_ini))
        x = tf.convert_to_tensor(lifted_ini, np.float32)
        self.assertEqual(list(self.ini) + [1., 0., 0., 1.], lifted_ini)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            flow = self.rk_diff.solve_iterative(1.0, self.vec_field.lifted_v(self.bm, True), x)
            flows = [tf.tensordot(tf.convert_to_tensor(v, np.float32), flow, 1) 
                for v in idmatrix[:self.vec_field.get_state_size()]]
            y0 = tf.tensordot(flows, tf.convert_to_tensor([1,0], np.float32), 1)
            y1 = tf.tensordot(flows, tf.convert_to_tensor([0,1], np.float32), 1)
        dy0 = g.gradient(y0, x).numpy()
        dy1 = g.gradient(y1, x).numpy()
        jacobi = [list(dy0)[:self.vec_field.get_state_size()], list(dy1)[:self.vec_field.get_state_size()]]

        del g
        h = 1e-6
        lifted_ini_d0_plus = lifted_ini + np.array([h,0,0,0,0,0])
        lifted_ini_d0_minus = lifted_ini + np.array([-h,0,0,0,0,0])
        flow_d0_plus = self.rk.solve_iterative(1.0, self.vec_field.lifted_v(self.bm), lifted_ini_d0_plus)
        flow_d0_minus = self.rk.solve_iterative(1.0, self.vec_field.lifted_v(self.bm), lifted_ini_d0_minus)
 
        lifted_ini_d1_plus = lifted_ini + np.array([0,h,0,0,0,0])
        lifted_ini_d1_minus= lifted_ini + np.array([0,-h,0,0,0,0])
  
        flow_d1_plus = self.rk.solve_iterative(1.0, self.vec_field.lifted_v(self.bm), lifted_ini_d1_plus)
        flow_d1_minus = self.rk.solve_iterative(1.0, self.vec_field.lifted_v(self.bm), lifted_ini_d1_minus)
        dy0_expected = list((flow_d0_plus - flow_d0_minus)/(2.*h))[:self.vec_field.get_state_size()]
        dy1_expected = list((flow_d1_plus - flow_d1_minus)/(2.*h))[:self.vec_field.get_state_size()]
        expected = np.transpose(np.array([dy0_expected, dy1_expected])).tolist()

        print("jacobi", jacobi)
        print("expected", expected)
        for (e, a) in zip(expected, jacobi):
            for (x, y) in zip(e, a):
                self.assertAlmostEqual(x, y, 3)
        
        actualJacobiInv = self.lift.evolveJacobiInv(self.ini, self.bm)
        actualId = np.dot(actualJacobiInv, np.array(jacobi))
        print("actualId", actualId)
        expectedId = np.identity(self.vec_field.get_state_size())
        for (e, a) in zip(expectedId, actualId):
            for (x, y) in zip(e, a):
                self.assertAlmostEqual(x, y, 4)

    def test_price(self):
        strike = 1.05
        payoff = lambda x: np.maximum(x- strike, 0)
        drift_func = lambda x: self.vec_field.get_v0()(x)

        path = self.lift.generate_path(10000, self.step)
        drift1 = np.array([drift_func(p) for p in path])[:,0]
        drift1 = drift1[~np.isnan(drift1)]
        print(drift1)
        drift0 = drift_func(self.ini)[0]
        print(drift0)
        drift= np.mean(0.5 * (drift0+drift1))
        path = path[:,0]
        path = path[~np.isnan(path)]
        pay = payoff(path)
        pv = np.mean(pay)
        stdv = np.std(pay)
        print("drift", drift)
        print("step", self.step)
        print("pv10", pv)
        print("std10", stdv)
        maxmum = np.max(pay)
        print("max pay", maxmum)
        print("e^0.5*sigma**2", np.exp(0.5 * 0.09))


if __name__ == "__main__":
    unittest.main()