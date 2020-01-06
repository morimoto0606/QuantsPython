import numpy as np
import tensorflow as tf

class Sabr:
    """
    gamma[i,j,k] = ChristrfelSymbol Gamma_{i,j}^k
    """
    def __init__(self,
        a: float,
        b: float,
        beta: float,
        rho: float):
        self.state_size = 2
        self.bm_size = 2
        self.a = a
        self.b = b
        self.beta = beta
        self.rho = rho
        self.gamma = np.zeros((2,2,2))
        self.gamma[0,0,1] = -b * (1.-(rho ** 2.)) ** 0.5
        self.gamma[0,1,0] = self.gamma[0,0,1]
        self.gamma[1,1,0] = -b * rho 
        self.gamma[1,0,1] = self.gamma[1,1,0]
        

    def get_state_size(self):
        return self.state_size

    def get_bm_size(self):
        return self.bm_size

    def lifted_v(self, 
        bm: np.ndarray,
        is_differentiable=False):
        """
        return random vector field:  [B^i(t)V_i(x), i=1,..,d]
        """
        def vec1(x: np.ndarray):
            """
            x[0]=x1, x[1]=x2, x[2]=e11, x[3]=e12,x[4]=e21,x[5]=e22
            """
            v0 = x[2] * self.a * x[0] ** self.beta * x[1]
            v1 = x[2] * self.b * self.rho * x[1] + x[4] * self.b * (1. - self.rho ** 2.) ** 0.5 * x[1]
            n0 = x[2] * x[2] * self.gamma[0,0,0] + x[2] * x[4] * self.gamma[0,1,0] + x[4] * x[2] * self.gamma[1,0,0] + x[4] * x[4] * self.gamma[1,1,0]
            n1 = x[2] * x[2] * self.gamma[0,0,1] + x[2] * x[4] * self.gamma[0,1,1] + x[4] * x[2] * self.gamma[1,0,1] + x[4] * x[4] * self.gamma[1,1,1]
            n2 = x[2] * x[3] * self.gamma[0,0,0] + x[2] * x[5] * self.gamma[0,1,0] + x[4] * x[3] * self.gamma[1,0,0] + x[4] * x[5] * self.gamma[1,1,0]
            n3 = x[2] * x[3] * self.gamma[0,0,1] + x[2] * x[5] * self.gamma[0,1,1] + x[4] * x[3] * self.gamma[1,0,1] + x[4] * x[5] * self.gamma[1,1,1]
            if is_differentiable:
                return tf.convert_to_tensor([v0, v1, n0, n1, n2, n3], np.float32) * bm[0]
            else:
                return np.array([v0, v1, n0, n1, n2, n3]) * bm[0]
 
        def vec2(x: np.ndarray):
            """
            x[0]=x1, x[1]=x2, x[2]=e11, x[3]=e12,x[4]=e21,x[5]=e22
            """
            v0 = x[3] * self.a * x[0] ** self.beta * x[1]
            v1 = x[3] * self.b * self.rho * x[1] + x[5] * self.b * (1. - self.rho ** 2.) ** 0.5 * x[1]
            n0 = x[3] * x[2] * self.gamma[0,0,0] + x[3] * x[4] * self.gamma[0,1,0] + x[5] * x[2] * self.gamma[1,0,0] + x[5] * x[4] * self.gamma[1,1,0]
            n1 = x[3] * x[2] * self.gamma[0,0,1] + x[3] * x[4] * self.gamma[0,1,1] + x[5] * x[2] * self.gamma[1,0,1] + x[5] * x[4] * self.gamma[1,1,1]
            n2 = x[3] * x[3] * self.gamma[0,0,0] + x[3] * x[5] * self.gamma[0,1,0] + x[5] * x[3] * self.gamma[1,0,0] + x[5] * x[5] * self.gamma[1,1,0]
            n3 = x[3] * x[3] * self.gamma[0,0,1] + x[3] * x[5] * self.gamma[0,1,1] + x[5] * x[3] * self.gamma[1,0,1] + x[5] * x[5] * self.gamma[1,1,1]
            if is_differentiable:
                return tf.convert_to_tensor([v0, v1, n0, n1, n2, n3], np.float32) * bm[1]
            else:
                return np.array([v0, v1, n0, n1, n2, n3]) * bm[1]
        return [vec1, vec2]

    def get_v0(self):
        def vec0(x: np.ndarray):
            v0 = -0.5 * (self.a ** 2. * self.beta * x[1]**2 * x[0] **  (2. * self.beta -1.) + self.a * self.b * self.rho * x[1] * x[0] ** self.beta)
            v1 = -0.5 * self.b ** 2. * x[1]
            return np.array([v0, v1])
        return vec0
        
