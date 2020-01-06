import tensorflow as tf
import numpy as np
from runge_kutta import *
from vector_field import *
from numpy.linalg import inv
from numpy.linalg import det
import sobol_normal
import time

class StochasticLift:
    def __init__(self,
        stepsize: float,
        rk,
        rk_diff,
        vec_field,
        ini: np.ndarray):
        self.stepsize = stepsize
        self.rk = rk
        self.rk_diff = rk_diff
        self.vec_field = vec_field
        self.ini = ini
        self.idmatrix = np.identity(len(ini) + vec_field.get_bm_size() ** 2)

    def get_lifted_ini(self,
        ini: np.ndarray):
        d = self.vec_field.get_bm_size()
        lifted_ini = list(ini) + list(np.identity(d).flatten())
        return lifted_ini

    def evolveJacobiInv(self,
        prev: np.ndarray,
        bm: np.ndarray):
        """
        dX(t) = sum_{i=1}^{d} V_i(X(t)) \circ dB^i(t)
        J(t) = dX/dx
        calculate J^{-1}(t) by AAD for one sample rand_nominals = [z1,,,zd]
        """
        #start = time.time()
        lifted_ini = self.get_lifted_ini(prev)
        x = tf.convert_to_tensor(lifted_ini, np.float32)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            flow = self.rk_diff.solve_iterative(1.0, self.vec_field.lifted_v(bm,True), x) 
            flows = [tf.tensordot(tf.convert_to_tensor(v, np.float32), flow, 1) 
                for v in self.idmatrix[:self.vec_field.get_state_size()]]
        dy = np.array([g.gradient(f, x).numpy() for f in flows])
        dymat = np.dot(dy, np.identity(len(lifted_ini))[:,:self.vec_field.get_state_size()])
        del g
        #print("dymat", dymat)
        #print("time evolveJacobiInv", time.time() - start)
        #if (dymat[0].size==2):
        #    d = 1.0 / (dymat[0,0] * dymat[1,1] - dymat[0,1] * dymat[1,0])
        #    mat = d * np.array([[dymat[1,1],-dymat[0,1]],[-dymat[1,0], dymat[0,0]]]) 
        ##    print("mat", mat)
        #    return mat
        #print("inv(dymat)", inv(dymat))
        return inv(dymat)

    def evolveZeta(self,
        prev: np.ndarray,
        bm: np.ndarray):
        #start = time.time()
        jacobi = self.evolveJacobiInv(prev, bm)
        v0 = self.vec_field.get_v0()

        def vec_field_zeta(x):
            lifted_ini = self.get_lifted_ini(x)
            flow = self.rk.solve_iterative(1.0, self.vec_field.lifted_v(bm), lifted_ini)[:self.vec_field.get_state_size()]
            m = np.dot(jacobi, v0(flow))
            return m

        zeta = self.rk.solve(self.stepsize, vec_field_zeta, prev)
        #print("jacobi", jacobi)
        #print("zeta", zeta)
        #print("time evolveZeta", time.time() - start)
        return zeta

    def evolveX(self,
        prev: np.ndarray,
        bm: np.ndarray
        ):
        #start = time.time() 
        zeta = self.evolveZeta(prev, bm)
        #zeta = prev
        lifted_ini = self.get_lifted_ini(zeta)
        x =  self.rk.solve_iterative(1.0, self.vec_field.lifted_v(bm), lifted_ini)[:self.vec_field.get_state_size()]
        #print("x", x)
        #print("time evolveX", time.time() - start)
        return x

    def generate_path(self,
        no_path: int,
        no_step: int):
        start = time.time()
        #rnd_normal = tf.random.normal(shape=(no_path, no_step, self.vec_field.get_bm_size()), seed=1)
        #rnd_normal = sobol_normal.generate_path(no_path, no_step, self.vec_field.get_bm_size())
        np.random.seed(10)
        rnd_normal = np.random.normal(0., 1., (no_path, no_step, self.vec_field.get_bm_size()))
        mean = np.mean(rnd_normal)
        print("mean", mean)
        rnd_normal = rnd_normal - mean
        bm = np.sqrt(self.stepsize) * rnd_normal
        def get_onpath(bm_path): 
            #print("bm_path", bm_path)
            start = time.time() 
            x = self.ini
            for b in bm_path:
                x = self.evolveX(x, b)
            print("one path", x)
            print("time generatePath", time.time() - start)
            return x
        path = np.array([get_onpath(bm_path) for bm_path in bm])
        path = np.where(abs(path) < 1000 * self.ini[0], path, np.nan)
        print("time", time.time() - start)
        return path