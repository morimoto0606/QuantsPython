#
# Valuation of American Option
# with Least-Square Monte Carlo
# Primal Algorithm
# American Put Option

# This is Path wise version
# we have data by [X(t_1), ..., X(t_n)]
import numpy as np
import math
np.random.seed(150000)

# Model Parameters
s0 = 36.
K = 40.
T = 1.0
r = 0.06
sigma = 0.2

# simulation parameters
I = 25000
M = 50
dt = T / M
df = np.exp(-r * dt)
print("df")
print(df)
# stock price paths
# 配列ごとの足し算
# log s(t) - log s(t-1) = (r - 0.5 * sigma **2) * dt + sigma * sqrt(dt) * z, t = 1, .., M
d_log_s = (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * np.random.standard_normal((I, M))
print("d_log_s")
print(d_log_s)
initial = np.zeros((I, 1))
print(initial)
d_log_s = np.hstack((initial, d_log_s))
print("d_log_s")
print(d_log_s)

cum_sum = np.cumsum(d_log_s, axis=1)
print("cumsum")
print(cum_sum)
s = s0 * np.exp(cum_sum)
print("s")
print(s)

h = np.maximum(K - s, 0)
print("h", h)
v = h[-1]
print("v", v)

v = [x[-1] for x in h]
print("v", v)

for t in range(M - 1, 0, -1):
    state = [x[t] for x in s]
    regression = np.polyfit(state, v, 3)
    c = np.polyval(regression, state)

    execute = [x[t] for x in h]
    z = [a * df for a in v]
    v = np.where(execute > c, execute, z)

v0 = np.sum(v) / I
print(v0)



