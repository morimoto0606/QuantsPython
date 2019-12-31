import sys
sys.path.append('../')

import unittest
import numpy as np
from runge_kutta import RungeKutta5
import tensorflow as tf
from stochastic_lift import StochasticLift
from vector_field import *
from runge_kutta import RungeKutta5
from numpy.linalg import inv

class TestStochasticLift(unittest.TestCase):
    def setUp(self):
        self.vec_field = Sabr(1.0, 1.0, 0.9, -0.7)
        self.stepsize = 0.5
        self.normal = np.array([0.3, 0.5]) 
        self.bm = np.sqrt(self.stepsize) * self.normal
        self.rk = RungeKutta5()
        self.ini = np.array([1.0, 0.3])
        self.lift = StochasticLift(self.stepsize, self.rk, self.vec_field, self.ini)

    def test_evolveJacobiInv(self):
        d = self.vec_field.get_bm_size()
        lifted_ini = list(self.ini) + list(np.identity(d).flatten())
        idmatrix = np.identity(len(lifted_ini))
        x = tf.convert_to_tensor(lifted_ini, np.float32)
        self.assertEqual(list(self.ini) + [1., 0., 0., 1.], lifted_ini)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            flow = self.rk.solve_iterative(1.0, self.vec_field.lifted_v(self.bm), x)
            flows = [tf.tensordot(tf.convert_to_tensor(v, np.float32), flow, 1) 
                for v in idmatrix[:self.vec_field.get_state_size()]]
            y0 = tf.tensordot(flows, tf.convert_to_tensor([1,0], np.float32), 1)
            y1 = tf.tensordot(flows, tf.convert_to_tensor([0,1], np.float32), 1)
        dy0 = g.gradient(y0, x).numpy()
        dy1 = g.gradient(y1, x).numpy()
        jacobi = [list(dy0)[:self.vec_field.get_state_size()], list(dy1)[:self.vec_field.get_state_size()]]

        del g
        h = 1e-2
        lifted_ini_d0 = lifted_ini + np.array([h,0,0,0,0,0])
        flow_d0 = self.rk.solve_iterative(1.0, self.vec_field.lifted_v(self.bm), lifted_ini_d0)
        lifted_ini_d1 = lifted_ini + np.array([0,h,0,0,0,0])
        flow_d1 = self.rk.solve_iterative(1.0, self.vec_field.lifted_v(self.bm), lifted_ini_d1)
        dy0_expected = list((flow_d0 - flow).numpy()/h)[:self.vec_field.get_state_size()]
        dy1_expected = list((flow_d1 - flow).numpy()/h)[:self.vec_field.get_state_size()]
        expected = np.transpose(np.array([dy0_expected, dy1_expected])).tolist()

        print("jacobi", jacobi)
        print("expected", expected)
        for (e, a) in zip(expected, jacobi):
            for (x, y) in zip(e, a):
                self.assertAlmostEqual(x, y, 4)
        
        actualJacobiInv = self.lift.evolveJacobiInv(self.ini, self.bm)
        actualId = np.dot(actualJacobiInv, np.array(jacobi))
        print(actualId)
        expectedId = np.identity(self.vec_field.get_state_size())
        for (e, a) in zip(expectedId, actualId):
            for (x, y) in zip(e, a):
                self.assertAlmostEqual(x, y, 4)

    def test_price(self):
        strike = 1.05
        payoff = lambda x: np.maximum(x- strike, 0)
        path = self.lift.generate_path(10, 2)
        #path = np.array([[1.,2.],[3.,4.]])
        #print(payoff(path))
        pay = payoff(path)
        print(pay)
        pv = np.mean(pay)
        stdv = np.std(pay)
        print("pv10", pv)
        print("std10", stdv)
        maxmum = np.max(pay)
        print("max pay", maxmum)
 


        path = self.lift.generate_path(100, 2)
        #path = np.array([[1.,2.],[3.,4.]])
        #print(payoff(path))
        pay = payoff(path)
        pv = np.mean(pay)
        stdv = np.std(pay)
        print("pv100", pv)
        print("std100", stdv)
        maxmum = np.max(pay)
        print("max pay", maxmum)
 

        path = self.lift.generate_path(1000, 2)
        #path = np.array([[1.,2.],[3.,4.]])
        #print(payoff(path))
        pay = payoff(path)
        pv = np.mean(pay)
        stdv = np.std(pay)
        print("pv1000", pv)
        print("std1000", stdv)
        maxmum = np.max(pay)
        print("max pay", maxmum)
 

        path = self.lift.generate_path(10000, 2)
        #path = np.array([[1.,2.],[3.,4.]])
        #print(payoff(path))
        pay = payoff(path)
        pv = np.mean(pay)
        stdv = np.std(pay)
        print("max pay", maxmum)
        print("pv10000", pv)
        print("std10000", stdv)
        maxmum = np.max(pay)
        print("max pay", maxmum)
 




    