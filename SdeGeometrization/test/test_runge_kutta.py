import sys
sys.path.append('../')

import unittest
import numpy as np
from runge_kutta import RungeKutta5
import tensorflow as tf

class TestRungeKutta(unittest.TestCase):
    def setUp(self):
        self.a = 1.0
        self.vector_field = lambda x: self.a * x
        self.rk = RungeKutta5()
        self.ini = 3.0
        self.h = .1
 
    def test_expfunc(self):
        actual = self.rk.solve(self.h, self.vector_field, self.ini)
        print("actual", actual)
        expected = self.ini * np.exp(self.a * self.h)
        print("expected", expected)
        self.assertAlmostEqual(expected, actual, 4)
   
    def test_aad_expfunc(self):
        def xi(x):
           return self.rk.solve(self.h, self.vector_field,x)

        print("gradient")
        z = tf.convert_to_tensor(self.ini, np.float32)
        with tf.GradientTape() as tape:
            tape.watch(z)
            y = xi(z)
        actual = tape.gradient(y, z)
        del tape
        expected = np.exp(self.a * self.h)
        print('expected', expected)
        print('actual', actual)
        #for (x, y) in zip(expected, actual):
        #    self.assertAlmostEqual(x,y,4)

class TestRungeKuttaMult(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((3,3))
        self.a[1,0] = 1.
        self.a[2,0] = 2.
        self.a[2,1] = 3.

        self.vector_field = lambda x: np.dot(self.a, x)
        def vec_field_aad(x):
            tf_a = tf.convert_to_tensor(self.a, np.float32)
            return tf.tensordot(tf_a, x, 1)
        self.vector_field_aad = vec_field_aad

        self.rk = RungeKutta5()
        self.ini = np.array([3.,2.,1.])
        self.h = 0.1

    def test_matrix(self):
        b = np.dot(self.a,self.a)
        print("print b")
        print(b)
        #c = np.dot(b,self.a)
        #print("print c")
        #print(c)

    def test_expfunc(self):
        actual = self.rk.solve(self.h, self.vector_field, self.ini)
        expected = np.dot(np.identity(3) + self.h * self.a + 1./2. * self.h ** 2 * np.dot(self.a,self.a), self.ini)
        print("actual")
        print(actual)
        print("expected")
        print(expected)
        for (x,y) in zip(expected, actual):
            self.assertAlmostEqual(x, y, 4)
    
    def test_aad_expfunc(self):
        x = tf.convert_to_tensor(self.ini, np.float32)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            ode = self.rk.solve(self.h, self.vector_field_aad, x)
            print("ode", ode)
            y0 = tf.tensordot(tf.convert_to_tensor([1,0,0], np.float32), ode, 1)
            y1 = tf.tensordot(tf.convert_to_tensor([0,1,0], np.float32), ode, 1)
            y2 = tf.tensordot(tf.convert_to_tensor([0,0,1], np.float32), ode, 1)
        dy0 = g.gradient(y0, x).numpy()
        dy1 = g.gradient(y1, x).numpy()
        dy2 = g.gradient(y2, x).numpy()
        print("dy0", dy0)
        print("dy1", dy1)
        print("dy2", dy2)
        del g
 
        #self.assertAlmostEqual(expected, actual, 4)



if __name__ == "__main__":
    unittest.main()