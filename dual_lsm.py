#
# Valuation of American Options
# with Least Square Monte Carlo
# dual Algorithm
# American Put Option

import math
import numpy as np
np.random.seed(150000)

# model parameters

s0 = 36.0
K = 40.0
T = 1.0
r = 0.06
sigma = 0.2

# simulation parameters
I = 4096
M = 50
dt = T / M
df = math.exp(-r * dt)
dim = 3

# stock price paths
# log s(t) - log s(t-1) = (r - 0.5 * sigma ** 2) * dt + sigma * dw
dw = math.sqrt(dt) * np.random.standard_normal((M, I))
d_log_s = (r - 0.5 * sigma ** 2) * dt + sigma * dw
print("dlogS", d_log_s)
log_s = np.cumsum(d_log_s, axis=0)
log_s = np.vstack((np.zeros((1, I)), log_s))
print("log_s", log_s)
s = s0 * np.exp(log_s)
print("s", s)
# payoff
h = np.maximum(K - s, 0)
v = h[-1]
print("h", h)
print("v", v)

# Lsm Valuation by backward induction
reg = np.zeros((M + 1, dim + 1))
for t in range(M-1, 0, -1):
    reg[t] = np.polyfit(s[t], v, dim)
    c = np.polyval(reg[t], s[t])
    v = np.where(h[t] > c, h[t], v * df)
    print("reg[t]", reg[t])


# generator
print("Generator")


def reg_coeff():
    for t in range(M-1, 0, -1):
        yield np.polyfit(s[t], v, dim)


def  generate_next_state(st, J):
    '''
    function to generate next state s_{t+1} from s_t 
    :param st: initial state  start from here
    :param J: number of paths to simulate
    :return: s_{t+1}
    '''
    ran = np.random.standard_normal(J)
    states = st * np.exp((r - 0.5 * sigma ** 2) * dt
                         * sigma * ran * math.sqrt(dt))
    return states

J = 50


# dual_simulation
Q = np.zeros((M + 1, I))
U = np.zeros((M + 1, I))
for t in range(1, M + 1):
    for i in range(I):
        vt = max(h[t, i], np.polyval(reg[t], s[t, i]))
        # estimated value v(t, i)
        st = generate_next_state(s[t-1, i], J)
        ct = np.polyval(reg[t], st)
        ht = np.maximum(40. - st, 0)
        vtj = np.sum(np.where(ht > ct, ht, ct)) / J
        Q[t, i] = Q[t-1, i] / df + (vt - vtj) # optimal martingale
        U[t, i] = max(U[t-1, i] / df, h[t, i] - Q[t, i])
        if t == M:
            U[t, i] = np.maximum(U[t - 1, i] / df,
                                 np.mean(ht) - Q[t, i])
U0 = np.sum(U[M]) / I * df ** M
print("U0", U0)
print("4.39054815428")







