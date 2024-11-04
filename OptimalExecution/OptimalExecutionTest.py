#%% 
import numpy as np
from OptimalExecution import OptimalExecution

def G(t): 
    return np.power(t, -1.0) 
def f(x):
    return np.sqrt(x)

def df(x):
    return -0.5 / f(x)

X = 100 * 1e6
N = 1
T = 1
sigma = 0.1
lamda = 1.0

print(df(1))

#%%
opt = OptimalExecution(f, df, G, X, N, T, sigma, lamda)
method = 'Nelder-Mead'
res = opt.optimize(method)
print(res)

#%%
import numpy as np
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
np.diag(np.diag(a))
a
b = np.array([a[i] * i for i in range(3)])
b
#%%
def f(x):
    return x + 1

v = np.array([1, 2, 3])
f(v[:-1])

