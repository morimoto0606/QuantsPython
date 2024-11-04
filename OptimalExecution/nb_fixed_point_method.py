#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from fixed_point_method import FixedPointMethod 

def G(t):
     if t > 0:
         return np.power(t, -2) 
     else:
         return 0.0
def f(x):
    return x

def df(x):
    epsilon = 0.0001
    return (f(x + epsilon) - f(x - epsilon)) /(2.0 * epsilon)


X = 1
N = 10
T = 1.0
sigma = 0.1
lamda = 0.0

obj = FixedPointMethod(
    f, 
    df, 
    G, 
    X, 
    N, 
    T, 
    sigma, 
    lamda)

x0 = np.full(N, X / T)
res = obj.solve(x0, 10)

v = res
cum_v = np.cumsum(v)
df = pd.DataFrame({'v': v, 'cum_v': cum_v})
df.plot(secondary_y='cum_v')

#%%
import tangent
def f(x):
    return x 

df = tangent.grad(f, verbose=1)