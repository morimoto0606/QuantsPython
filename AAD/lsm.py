import math
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
tfe=tf.contrib.eager

np.random.seed(15000)
T = 10
I = 100
M = 10
r = 0.01
dt = T/M
df = math.exp(-r * dt)
K = 100
def stock_price(s0, sigma):
    return s0 * np.exp(np.cumsum((r - 0.5*sigma ** 2)*dt
            + sigma * math.sqrt(dt)* np.random.standard_normal((M+1, I)), axis=0))

def payoff(S, K):
    return tf.maximum(K - S, 0)

def bermuda_option(s0, sigma):
    S = stock_price(s0, sigma)
    h = payoff(S, K)
    V = h[-1]
    for t in range(M-1, 1, -1):
        rg = tf.polyfit(S[t], V * df, 5)
        C = tf.polyval(rg, S[t])
        V = tf.where(h[t] > C, h[t], V * df)
    V0 = df* tf.sum(V) / I
    return V0

def test(S, sigma):
    S = stock_price(S, sigma)
    Sum = np.sum(S) / I
    return Sum

def test2(S):
    return tf.exp(S)
def f(x):
    return tf.square(x)

#print(test(100, 0.3))
print(test(100, 0.3))
grad = tfe.gradients_function(test)
gradf = tfe.gradients_function(f)
print(gradf(100.0)[0].numpy())
#print(grad(100, 0.3)[0])
print(bermuda_option(100, 0.3))
grad_bermda_option = tfe.gradients_function(bermuda_option)
print(grad_bermda_option(100.0, 0.3)[0])
#print(bermuda_option(100,0.1))
