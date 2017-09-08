#
# Valuation of American Options
# with Least Square Monte Carlo
# Primal Algorithm
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
I = 100000
M = 50
dt = T / M
df = math.exp(-r * dt)
dim = 5

# stock price paths
# log s(t) - log s(t-1) = (r - 0.5 * sigma ** 2) * dt + sigma * dw
dw = math.sqrt(dt) * np.random.standard_normal((M, I))
d_log_s = (r - 0.5 * sigma ** 2) * dt + sigma * dw
d_log_s = np.vstack((np.zeros((1, I)), d_log_s))
s = s0 * np.exp(np.cumsum(d_log_s, axis=0))

# payoff
h = np.maximum(K - s, 0)

# final value
v = h[-1]

# American Option Value by backward induction
for t in range(M-1, 0, -1):
    reg = np.polyfit(s[t], v, dim)
    conti = np.polyval(reg, s[t])
    v = np.where(h[t] > conti, h[t], df * v)

v0 = np.sum(v) / I
print(v0)






