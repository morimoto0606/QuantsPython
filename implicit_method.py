import numpy as np
import math

s0 = 36.0
K = 36.0
T = 1.0
r = 0.06
sigma = 0.2

# simulation parameters
I = 10
M = 5
dt = T / M
df = math.exp(-r * dt)
dim = 5

dw = math.sqrt(dt) * np.random.standard_normal((M, I))
print('dw', dw)
d_log_s = (r - 0.5 * sigma * sigma) * dt + sigma * dw
print('d_log_s', d_log_s)
# add zero vector
d_log_s = np.vstack((np.zeros((1, I)), d_log_s))
print('d_log_s')
print(d_log_s)
s = s0 * np.exp(np.cumsum(d_log_s, axis=0))
print("s", s)
#payoff
F = s - K
print("F", F)
fut_pay = np.zeros((M + 1, I))
reg = np.zeros((M + 1, dim + 1))
fut_pay[M] = F[M]
for t in range(M - 1, 0, -1):
   fut_pay[t] = F[t] + fut_pay[t+1]
   reg[t] = np.polyfit(s[t], F[t], dim)
print("fut_pay", fut_pay)
print("reg", reg)
print(fut_pay.size)
print(fut_pay[0].size)

exposure = np.zeros((M + 1, I))
print('exposure size', exposure.shape[0], exposure.shape[1])
print('fut_pay size', fut_pay.shape[0], fut_pay.shape[1])

for t in range(1, M + 1):
   v = np.polyval(reg[t], s[t])
   exposure[t - 1] = np.maximum(v, 0)
print("exposure", exposure)
epe = np.sum(exposure, axis=1)
print("epe", epe)

print("exposure")
exp_and_payoff = zip(exposure, fut_pay)

for t in range(1, M + 1) :
   v = np.polyval(reg[t], s[t])
   v_c = zip(v, fut_pay[t])
   exposure[t - 1] = np.where(v > 0, fut_pay[t - 1], 0)
   print('e, v, f', exposure[t-1], v, fut_pay[t-1])

grid = np.ones(M + 1)
grid = grid * T / M

print("grid", grid)

cva = np.dot(epe, grid)
print("cva", cva)
