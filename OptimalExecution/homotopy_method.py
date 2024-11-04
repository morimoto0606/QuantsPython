import numpy as np
import copy
from regex import R
from scipy.optimize import minimize
from soupsieve import select

class HomotopyMethod:
    """
    f(x) = x^{beta}
    G(t) = t^{-gamma}
    v^0(t): arbitrary given
    v^m(t) = v^{m-1}(t) + h R^m(t)
    """
    def __init__(self, 
        amount, 
        vol, 
        alpha, 
        beta, 
        gamma, 
        risk_aversion, 
        expiry, 
        num_discretization, 
        order) -> None:
        self.amount = amount
        self.T = expiry
        self.N = num_discretization
        self.timegrid = np.linspace(0, expiry, num_discretization)
        self.delta = expiry / num_discretization
        self.grid = np.linspace(0, expiry, num_discretization)
        self.order = order
        self.beta = beta
        self.gamma = gamma
        self.vol = vol
        self.risk_aversion = risk_aversion 
        self.v = np.zeros(num_discretization)
        alpha_t = 0.5 * self.delta * (np.roll(alpha(self.timegrid), 1) + alpha(self.timegrid))
        alpha_t[0] = 0
        self.int_alpha = np.cumsum(alpha_t)
        index = 2-gamma
        self.G = np.array([[
            (self.T/self.N)**index * ((i-j+1)**index - 2.0 * (i - j)**index + (i-j-1)**index) / ((1-gamma)*index) 
            if i != j else 2.0 * (self.T/self.N)**index / ((1-gamma)*index) for j in range(self.N) ] for i in range(self.N)])
        self.f = lambda x: np.np.power(x, self.beta)
        self.df = [lambda x: np.prod([self.beta-i for i in range(k)]) * np.power(x, self.beta - k) for k in range(order)]
        #self.df = [lambda x: np.power(x, 2)]

    def get_increment_v(self, v, order):
        """
        R^m(t) = \int_0^t G(t-s) D^{m-1} f(phi_(s)) ds 
            + \int_t^T G(s-t) D^{m-1}(f'(phi(t))f(phi(s))) ds
            -2 nu * \int_0^t \sigma^2_s (\int_0^s D^{m-1}phi(u) du) ds
        v = [[v^i(t_0), ..., v^i(t_{N-1})] for i in range(m)]
        """
        m = order
        print(self.df[0](1.0))
        print(self.df[0](0.0))
        def f(x):
            return np.power(x, 2)
        f(np.array([0.0, .0]))
        print(self.df[0](np.array([0.0, 1.0])))
        mat1 = [self.diff1(self.df, v, m-1)] * self.N
        mat2 = [self.diff2(self.df, v[:, n], v, m-1) for n in range(self.N)]
        mat3 = [np.cumsum(v[m-1]) * self.delta] * self.N
        mat1a = np.array([[mat1[i][j] for j in range(i)] for i in range(self.N)])
        mat2a = np.array([[mat2[i][j] for j in range(i)] for i in range(self.N)])
        mat3a = np.array([[mat3[i][j] for j in range(i)] for i in range(self.N)])

        g = np.array([self.int_g(j, j+1, self.gamma) for j in range(self.N)])
        s = (np.array([self.vol] * self.N) if len(self.vol) == 1 else np.array(self.vol)) * self.delta 
        v1 = np.dot(mat1a, g)
        v2 = np.dot(mat2a, g)
        v3 = np.dot(mat3a, s)
        return v1 + v2 - 2.0 * self.nu * v3


    def calc_v(self, h):
        v = np.zeros((self.order, self.N))
        v[0] = self.v0(self.timegrid)
        for m in range(1, self.order):
            v[m] = v[m-1] + h * self.get_increment_v(v, m)
        return np.sum(v, axis=0)



    def target(self, param):
        h = param[0]
        c = param[1]
        f_v = [self.f(self.v) for _ in range(self.N)]
        df_v = [self.df[0](self.v) for _ in range(self.N)]
        term0 = np.zeros(self.N)
        term1 = np.zeros(self.N)
        F = np.zeros(self.N)
        for i in range(self.N):
            term0[i] = self.int_alpha[i]
            for j in range(self.N):
                F[j] = self.v[j] * df_v[i] if j > i else f_v[j]
            term1[i] = self.G.dot(F)

        v = self.calc_v(h)
        sumv = np.cumsum(v)
        sum_sigma_v = np.multiply(self.vol, sumv)
        term2 = -2.0 * np.cumsum(sum_sigma_v) * np.power(self.delta, 2)
        res = term0 + term1 + term2 + c
        return res.dot(res)

    def get_optimal_v(self, ini):
        '''
        solve by optimizer
        '''
        res = minimize(fun = self.target,
            x0 = ini,
            method='SLSQP',
            options={"disp":True})
        h = res[0]
        return self.calc_v(h)


    def v0(self, t):
        """
        initial function of the approximation of v(t)
        """
        c = self.amount / (np.power(self.T/2, self.gamma))
        return c * np.power(t(self.T - t), 0.5 * (self.gamma - 1))

    def diff1(self, df, b, order):
        """
        input
            df: vector of function s.t. df=[f, df, d2f, ....]
            b: vector for coeff of b[p][i] s.t. b(t_i) = b[0][i] + b[1][i]p + 1/2! b[2][i]p**2 + ..., i.e. b[k][i] = d^k b(t_i, p)/dp^k|p=0
            b[k] = [b[k][0], ..., b[k][n-1]]
            order: int for order of expansion
        return
            d**m {f(b(p))}|p=0
            where d**m f = 1/m! d**mf/dp**m
        """
        term0 = 0 if order < 0 else df[0](b[0]) 
        term1 = 0 if order < 1 else b[1] * df[1](b[0]) 
        term2 = 0 if order < 2 else 1./2 * (b[2] * df[1](b[0]) + b[1]**2 * df[2](b[0])) 
        term3 = 0 if order < 3 else 1./6 * (b[3] * df[1](b[0]) + df[3](b[0]) * b[1]**3 + 3 * b[1] * b[2] * df[2](b[0])) 
        term4 = 0 if order < 4 else 1./24 * (b[4] * df[1](b[0]) + df[4](b[0]) * b[1]**4 + 6 * df[3](b[0]) * b[1]**2 * b[2] * + df[2](b[0]) * (3 * b[2]**2 + 4 * b[3] * b[1])) 
        term5 = 0 if order < 5 else 1./120 * (b[5] * df[1](b[0]) + df[5](b[0]) * b[1]**5 + 5 * b[4] * b[1] * df[2](b[0]) + 10 * b[3] * b[2] * df[2](b[0]) + 10 * b[3] * df[3](b[0]) * b[1]**2 + 10 * df[4](b[0]) * b[1]**3 * b[2] + 15 * df[3](b[0]) * b[1] * b[2]**2)
        term6 = 0 if order < 6 else 1./720 * (b[6] * df[1](b[0]) + df[6](b[0]) * b[1]**6 + 20 * b[3] * df[4](b[0]) * b[1]**3 + 15 * df[5](b[0]) * b[1]**4 * b[2]  + 45 * df[4](b[0]) * b[1]**2 * b[2]**2 + 15 * df[3](b[0]) * (b[2]**3 + b[4] * b[1]**2 + 4 * b[3] * b[1] * b[2]) + df[2](b[0]) * (10 * b[3]**2 + 6 * b[5] * b[1] + 15 * b[4] * b[2]))
        return term0 + term1 + term2 + term3 + term4 + term5 + term6


    def diff2(self, df, a, b, order):   
        """
        input
            df: vector of function s.t. df=[f, df, d2f, ....]
            a: vector for coeff of a[p] s.t. a = a[0] + a[1]p + 1/2! a[2]p**2 + ..., i.e. a[i] = d^i a(p)/dp|p=0
            b: vector for coeff of b[p] s.t. b = b[0] + b[1]p + 1/2! b[2]p**2 + ..., i.e. b[i] = d^i b(p)/dp|p=0
            order: int for order of expansion
        return
            d**m {f(a(p)) * f'(b(p))}|p=0
            where d**m f = 1/m! d**mf/dp**m
        """
        term0 = 0 if order < 0 else df[0](b[0]) * df[1](a[0]) 
        term1 = 0 if order < 1 else (a[1] * df[0](b[0]) * df[2](a[0]) 
            + b[1] * df[1](a[0]) * df[1](b[0])) 
        term2 = 0 if order < 2 else 1./2 * (a[2] * df[0](b[0]) * df[2](a[0]) 
            + 2. * a[1] * b[1] * df[2](a[0]) * df[1](b[0]) 
            + df[3](a[0]) * a[1]**2 * df[0](b[0]) 
            + b[2] * df[1](a[0]) * df[1](b[0]) 
            + b[1]**2 * df[1](a[0]) * df[2](b[0])) 
        term3 = 0 if order < 3 else 1./6 * (a[3] * df[0](b[0]) * df[2](a[0]) 
            + 3 * a[2] * b[1] * df[2](a[0]) * df[1](b[0]) 
            + 3 * a[1] * b[2] * df[2](a[0]) * df[1](b[0]) 
            + 3 * a[1] * b[1]**2 * df[2](a[0]) * df[2](b[0]) 
            + 3 * df[3](a[0]) * a[1]**2 * b[1] * df[1](b[0]) 
            + df[4](a[0]) * a[1]**3 * df[0](b[0]) 
            + 3 * df[3](a[0]) * a[1] * a[2] * df[0](b[0]) 
            + b[3] * df[1](a[0]) * df[1](b[0]) 
            + df[3](b[0]) * b[1]**3 * df[1](a[0]) 
            + 3. * b[1] * b[2] * df[1](a[0]) * df[2](b[0])) 
        term4 = 0 if order < 4 else 1./24 * (a[4] * df[0](b[0]) * df[2](a[0]) 
            + 4. * a[3] * b[1] * df[2](a[0]) * df[1](b[0]) 
            + 6. * a[2] * b[2] * df[2](a[0]) * df[1](b[0]) 
            + 6. * a[2] * b[1]**2 * df[2](a[0]) * df[2](b[0])
            + 3. * df[3](a[0]) * a[2]**2 * df[0](b[0]) 
            + 4. * b[3] * a[1] * df[2](a[0]) * df[1](b[0]) 
            + 6. * df[3](a[0]) * a[1]**2 * b[2] * df[1](b[0])
            + 4. * df[4](a[0]) * a[1]**3 * b[1] * df[1](b[0]) 
            + 6. * df[3](a[0]) * a[1]**2 * b[1]**2 * df[2](b[0]) 
            + 4. * a[1] * df[3](b[0]) * b[1]**3 * df[2](a[0]) 
            + 12. * a[1] * b[1] * b[2] * df[2](a[0]) * df[2](b[0]) 
            + df[5](a[0]) * a[1]**4 * df[0](b[0])
            + 4. * a[3] * df[3](a[0]) * a[1] * df[0](b[0]) 
            + 12. * df[3](a[0]) * a[1] * a[2] * b[1] * df[1](b[0]) 
            + 6. * df[4](a[0]) * a[1]**2 * a[2] * df[0](b[0]) 
            + b[4] * df[1](a[0]) * df[1](b[0]) 
            + 3. * b[2]**2 * df[1](a[0]) * df[2](b[0]) 
            + df[4](b[0]) * b[1]**4 * df[1](a[0]) 
            + 4. * b[3] * b[1] * df[1](a[0]) * df[2](b[0]) 
            + 6. * df[3](b[0]) * b[1]**2 * b[2] * df[1](a[0])) 
        term5 = 0 if order < 5 else 1./120 * (a[5] * df[0](b[0]) * df[2](a[0])
            + 5. * a[4] * b[1] * df[2](a[0]) * df[1](b[0]) 
            + 10. * a[3] * b[2] * df[2](a[0]) * df[1](b[0]) 
            + 10. * a[3] * b[1]**2 * df[2](a[0]) * df[2](b[0]) 
            + 10. * b[3] * a[2] * df[2](a[0]) * df[1](b[0]) 
            + 10. * a[2] * df[3](b[0]) * b[1]**3 * df[2](a[0]) 
            + 15. * df[3](a[0]) * a[2]**2 * b[1] * df[1](b[0]) 
            + 30. * a[2] * b[1] * b[2] * df[2](a[0]) * df[2](b[0]) 
            + 5. * b[4] * a[1] * df[2](a[0]) * df[1](b[0]) 
            + 10. * b[3] * df[3](a[0]) * a[1]**2 * df[1](b[0]) 
            + 15. * a[1] * b[2]**2 * df[2](a[0]) * df[2](b[0]) 
            + 10. * df[4](a[0]) * a[1]**3 * b[2] * df[1](b[0]) 
            + 10. * df[3](a[0]) * a[1]**2 * df[3](b[0]) * b[1]**3
            + 5. * df[5](a[0]) * a[1]**4 * b[1] * df[1](b[0]) 
            + 10. * df[4](a[0]) * a[1]**3 * b[1]**2 * df[2](b[0]) 
            + 5. * a[1] * df[4](b[0]) * b[1]**4 * df[2](a[0]) 
            + 20. * b[3] * a[1] * b[1] * df[2](a[0]) * df[2](b[0]) 
            + 30. * df[3](a[0]) * a[1]**2 * b[1] * b[2] * df[2](b[0]) 
            + 30. * a[1] * df[3](b[0]) * b[1]**2 * b[2] * df[2](a[0])
            + df[6](a[0]) * a[1]**5 * df[0](b[0]) 
            + 5. * a[4] * df[3](a[0]) * a[1] * df[0](b[0]) 
            + 10. * a[3] * df[3](a[0]) * a[2] * df[0](b[0]) 
            + 20. * a[3] * df[3](a[0]) * a[1] * b[1] * df[1](b[0]) 
            + 10. * a[3] * df[4](a[0]) * a[1]**2 * df[0](b[0]) 
            + 30. * df[3](a[0]) * a[1] * a[2] * b[2] * df[1](b[0]) 
            + 30. * df[4](a[0]) * a[1]**2 * a[2] * b[1] * df[1](b[0]) 
            + 30. * df[3](a[0]) * a[1] * a[2] * b[1]**2 * df[2](b[0]) 
            + 10. * df[5](a[0]) * a[1]**3 * a[2] * df[0](b[0]) 
            + 15. * df[4](a[0]) * a[1] * a[2]**2 * df[0](b[0]) 
            + b[5] * df[1](a[0]) * df[1](b[0])
            + df[5](b[0]) * b[1]**5 * df[1](a[0]) 
            + 5. * b[4] * b[1] * df[1](a[0]) * df[2](b[0]) 
            + 10. * b[3] * b[2] * df[1](a[0]) * df[2](b[0]) 
            + 10. * b[3] * df[3](b[0]) * b[1]**2 * df[1](a[0]) 
            + 10. * df[4](b[0]) * b[1]**3 * b[2] * df[1](a[0]) 
            + 15. * df[3](b[0]) * b[1] * b[2]**2 * df[1](a[0])) 
        term6 = 0 if order < 6 else 1./720 * (df[1](b[0]) * df[6](a[0]) * a[1]**6
            + 6. * b[1] * df[2](b[0]) * df[5](a[0]) * a[1]**5
            + 15. * b[2] * df[2](b[0]) * df[4](a[0]) * a[1]**4 
            + 15. * b[1]**2 * df[3](b[0]) * df[4](a[0]) * a[1]**4 
            + 15. * df[1](b[0]) * a[2] * df[5](a[0]) * a[1]**4 
            + 20. * df[2](b[0]) * b[3] * df[3](a[0]) * a[1]**3 
            + 60. * b[1] * b[2] * df[3](a[0]) * df[3](b[0]) * a[1]**3 
            + 60. * b[1] * a[2] * df[2](b[0]) * df[4](a[0]) * a[1]**3 
            + 20. * df[1](b[0]) * a[3] * df[4](a[0]) * a[1]**3 
            + 20. * b[1]**3 * df[3](a[0]) * df[4](b[0]) * a[1]**3
            + 90. * a[2] * b[2] * df[2](b[0]) * df[3](a[0]) * a[1]**2 
            + 60. * b[1] * df[2](b[0]) * a[3] * df[3](a[0]) * a[1]**2 
            + 45. * b[2]**2 * df[2](a[0]) * df[3](b[0]) * a[1]**2 
            + 60. * b[1] * df[2](a[0]) * b[3] * df[3](b[0]) * a[1]**2 
            + 90. * b[1]**2 * a[2] * df[3](a[0]) * df[3](b[0]) * a[1]**2 
            + 15. * df[1](b[0]) * df[3](a[0]) * a[4] * a[1]**2 
            + 15. * df[2](a[0]) * df[2](b[0]) * b[4] * a[1]**2 
            + 45. * df[1](b[0]) * a[2]**2 * df[4](a[0]) * a[1]**2 
            + 90. * b[1]**2 * b[2] * df[2](a[0]) * df[4](b[0]) * a[1]**2 
            + 15. * b[1]**4 * df[2](a[0]) * df[5](b[0]) * a[1]**2 
            + 60. * b[2] * df[2](a[0]) * df[2](b[0]) * a[3] * a[1] 
            + 60. * a[2] * df[2](a[0]) * df[2](b[0]) * b[3] * a[1] 
            + 90. * b[1] * a[2]**2 * df[2](b[0]) * df[3](a[0]) * a[1] 
            + 60. * df[1](b[0]) * a[2] * a[3] * df[3](a[0]) * a[1] 
            + 180. * b[1] * a[2] * b[2] * df[2](a[0]) * df[3](b[0]) * a[1] 
            + 60. * b[1]**2 * df[2](a[0]) * a[3] * df[3](b[0]) * a[1] 
            + 60. * df[1](a[0]) * b[2] * b[3] * df[3](b[0]) * a[1] 
            + 30. * b[1] * df[2](a[0]) * df[2](b[0]) * a[4] * a[1] 
            + 30. * b[1] * df[1](a[0]) * df[3](b[0]) * b[4] * a[1] 
            + 90. * b[1] * df[1](a[0]) * b[2]**2 * df[4](b[0]) * a[1] 
            + 60. * b[1]**3 * a[2] * df[2](a[0]) * df[4](b[0]) * a[1] 
            + 60. * b[1]**2 * df[1](a[0]) * b[3] * df[4](b[0]) * a[1] 
            + 6. * df[1](b[0]) * df[2](a[0]) * a[5] * a[1] 
            + 6. * df[1](a[0]) * df[2](b[0]) * b[5] * a[1] 
            + 60. * b[1]**3 * df[1](a[0]) * b[2] * df[5](b[0]) * a[1] 
            + 6. * b[1]**5 * df[1](a[0]) * df[6](b[0]) * a[1] 
            + 10. * df[1](b[0]) * df[2](a[0]) * a[3]**2 
            + 45. * a[2]**2 * b[2] * df[2](a[0]) * df[2](b[0]) 
            + 60. * b[1] * a[2] * df[2](a[0]) * df[2](b[0]) * a[3] 
            + 20. * df[1](a[0]) * df[2](b[0]) * a[3] * b[3] 
            + 15. * df[1](b[0]) * a[2]**3 * df[3](a[0]) 
            + 45. * df[1](a[0]) * a[2] * b[2]**2 * df[3](b[0]) 
            + 10. * df[0](a[0]) * b[3]**2 * df[3](b[0])
            + 45. * b[1]**2 * a[2]**2 * df[2](a[0]) * df[3](b[0]) 
            + 60. * b[1] * df[1](a[0]) * b[2] * a[3] * df[3](b[0]) 
            + 60. * b[1] * df[1](a[0]) * a[2] * b[3] * df[3](b[0]) 
            + 15. * df[1](b[0]) * a[2] * df[2](a[0]) * a[4] 
            + 15. * df[1](a[0]) * b[2] * df[2](b[0]) * a[4] 
            + 15. * b[1]**2 * df[1](a[0]) * df[3](b[0]) * a[4]
            + 15. * df[1](a[0]) * a[2] * df[2](b[0]) * b[4] 
            + 15. * df[0](a[0]) * b[2] * df[3](b[0]) * b[4] 
            + 15. * df[0](a[0]) * b[2]**3 * df[4](b[0]) 
            + 90. * b[1]**2 * df[1](a[0]) * a[2] * b[2] * df[4](b[0]) 
            + 20. * b[1]**3 * df[1](a[0]) * a[3] * df[4](b[0]) 
            + 60. * df[0](a[0]) * b[1] * b[2] * b[3] * df[4](b[0])
            + 15. * df[0](a[0]) * b[1]**2 * b[4] * df[4](b[0])
            + 6. * b[1] * df[1](a[0]) * df[2](b[0]) * a[5] 
            + 6. * df[0](a[0]) * b[1] * df[3](b[0]) * b[5] 
            + 45. * df[0](a[0]) * b[1]**2 * b[2]**2 * df[5](b[0]) 
            + 15. * b[1]**4 * df[1](a[0]) * a[2] * df[5](b[0])
            + 20. * df[0](a[0]) * b[1]**3 * b[3] * df[5](b[0]) 
            + df[1](a[0]) * df[1](b[0]) * a[6] 
            + df[0](a[0]) * df[2](b[0]) * b[6] 
            + 15. * df[0](a[0]) * b[1]**4 * b[2] * df[6](b[0])
            + df[0](a[0]) * b[1]**6 * df[7](b[0]))

        return term0 + term1 + term2 + term3 + term4 + term5 + term6


    
def get_homotopy_curve(
    amount,
    vol,
    alpha,
    beta,
    gamma,
    risk_aversion,
    expiry,
    num_discretization,
    order,
    ini):
    """
    f(x) = x^{beta}
    G(t) = t^{-gamma}
    """
    hom = HomotopyMethod(
        amount=amount,
        vol=vol,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        risk_aversion=risk_aversion,
        expiry=expiry,
        num_discretization=num_discretization,
        order=order)
    v = hom.get_optimal_v(ini)

    cum_v = np.cumsum(v) * expiry / num_discretization
    v_twap = amount / expiry
    v_ratio = np.array(v) / v_twap
    df = pd.DataFrame({'v': v_ratio, 'cum_v': cum_v})
    return df

