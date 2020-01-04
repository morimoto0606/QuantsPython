import numpy as np
from typing import List
from typing import Callable
import tensorflow as tf

class RungeKutta5:
    def __init__(self):
        self._a = np.zeros((6, 6))
        self._a[1,0] = 2./5
        self._a[2,0] = 11./64
        self._a[2,1] = 5./64
        self._a[3,2] = 0.5
        self._a[4,0] = 3./64
        self._a[4,1] = -15./64
        self._a[4,2] = 3./8
        self._a[4,3] = 9./16
        self._a[5,1] = 5./7
        self._a[5,2] = 6./7
        self._a[5,3] = -12./7
        self._a[5,4] = 8./7

        self._b=np.array([7.,0.,32.,12.,32.,7.])/90

    def solve(self,
        h: float,
        vecfield: Callable[[np.ndarray], np.ndarray],
        ini_val: np.ndarray):
        k0 = vecfield(ini_val)
        k1 = vecfield(ini_val + h * (self._a[1,0] * k0))
        k2 = vecfield(ini_val + h * (self._a[2,0] * k0 + self._a[2,1] * k1))
        k3 = vecfield(ini_val + h * (self._a[3,0] * k0 + self._a[3,1] * k1 + self._a[3,2] * k2))
        k4 = vecfield(ini_val + h * (self._a[4,0] * k0 + self._a[4,1] * k1 + self._a[4,2] * k2 + self._a[4,3] * k3))
        k5 = vecfield(ini_val + h * (self._a[5,0] * k0 + self._a[5,1] * k1 + self._a[5,2] * k2 + self._a[5,3] * k3 + self._a[5,4] * k4))
        k = np.array([k0, k1, k2, k3, k4, k5])
        ret = ini_val + h * np.dot(self._b, k)
        return ret

    def solve_iterative(self,
        h: float,
        vecfields: List[Callable[[np.ndarray], np.ndarray]],
        ini_val: np.ndarray):
        x = ini_val
        for v in vecfields:
            x = self.solve(h, v, x)
        return x



class RungeKutta5Differentiable:
    def __init__(self):
        self._a = np.zeros((6, 6))
        self._a[1,0] = 2./5
        self._a[2,0] = 11./64
        self._a[2,1] = 5./64
        self._a[3,2] = 0.5
        self._a[4,0] = 3./64
        self._a[4,1] = -15./64
        self._a[4,2] = 3./8
        self._a[4,3] = 9./16
        self._a[5,1] = 5./7
        self._a[5,2] = 6./7
        self._a[5,3] = -12./7
        self._a[5,4] = 8./7

        self._b=tf.convert_to_tensor(np.array([7.,0.,32.,12.,32.,7.])/90, np.float32)

    def solve(self,
        h: float,
        vecfield: Callable[[np.ndarray], np.ndarray],
        ini_val: np.ndarray):
        k0 = vecfield(ini_val)
        k1 = vecfield(ini_val + h * (self._a[1,0] * k0))
        k2 = vecfield(ini_val + h * (self._a[2,0] * k0 + self._a[2,1] * k1))
        k3 = vecfield(ini_val + h * (self._a[3,0] * k0 + self._a[3,1] * k1 + self._a[3,2] * k2))
        k4 = vecfield(ini_val + h * (self._a[4,0] * k0 + self._a[4,1] * k1 + self._a[4,2] * k2 + self._a[4,3] * k3))
        k5 = vecfield(ini_val + h * (self._a[5,0] * k0 + self._a[5,1] * k1 + self._a[5,2] * k2 + self._a[5,3] * k3 + self._a[5,4] * k4))
        k = tf.convert_to_tensor([k0, k1, k2, k3, k4, k5], np.float32)
        ret = ini_val + h * tf.tensordot(self._b, k, 1)
        return ret

    def solve_iterative(self,
        h: float,
        vecfields: List[Callable[[np.ndarray], np.ndarray]],
        ini_val: np.ndarray):
        h = tf.convert_to_tensor(h, np.float32)
        x = ini_val
        for v in vecfields:
            x = self.solve(h, v, x)
        return x

class RungeKutta4:
    def __init__(self):
        self._a = np.zeros((4, 4))
        self._a[1,0] = 0.5
        self._a[2,1] = 0.5
        self._a[3,2] = 1.0
        self._b=np.array([1.,2.,2.,1.])/6

    def solve(self,
        h: float,
        vecfield: Callable[[np.ndarray], np.ndarray],
        ini_val: np.ndarray):
        k0 = vecfield(ini_val)
        k1 = vecfield(ini_val + h * (self._a[1,0] * k0))
        k2 = vecfield(ini_val + h * (self._a[2,0] * k0 + self._a[2,1] * k1))
        k3 = vecfield(ini_val + h * (self._a[3,0] * k0 + self._a[3,1] * k1 + self._a[3,2] * k2))
        k = np.array([k0, k1, k2, k3])
        ret = ini_val + h * np.dot(self._b, k)
        return ret

    def solve_iterative(self,
        h: float,
        vecfields: List[Callable[[np.ndarray], np.ndarray]],
        ini_val: np.ndarray):
        x = ini_val
        for v in vecfields:
            x = self.solve(h, v, x)
        return x

class RungeKutta4Differentiable:
    def __init__(self):
        self._a = np.zeros((4, 4))
        self._a[1,0] = 0.5
        self._a[2,1] = 0.5
        self._a[3,2] = 1.0
        self._b=tf.convert_to_tensor(np.array([1.,2.,2.,1.])/6, np.float32)

    def solve(self,
        h: float,
        vecfield: Callable[[np.ndarray], np.ndarray],
        ini_val: np.ndarray):
        k0 = vecfield(ini_val)
        k1 = vecfield(ini_val + h * (self._a[1,0] * k0))
        k2 = vecfield(ini_val + h * (self._a[2,0] * k0 + self._a[2,1] * k1))
        k3 = vecfield(ini_val + h * (self._a[3,0] * k0 + self._a[3,1] * k1 + self._a[3,2] * k2))
        k = tf.convert_to_tensor([k0, k1, k2, k3], np.float32)
        ret = ini_val + h * tf.tensordot(self._b, k, 1)
        return ret

    def solve_iterative(self,
        h: float,
        vecfields: List[Callable[[np.ndarray], np.ndarray]],
        ini_val: np.ndarray):
        h = tf.convert_to_tensor(h, np.float32)
        x = ini_val
        for v in vecfields:
            x = self.solve(h, v, x)
        return x
