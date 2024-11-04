#%%
import matplotlib.pyplot as plt


#%%
import numpy as np
from numpy.core.fromnumeric import argmin
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

N = 100
T = 100
t = np.linspace(0,1,N)

alpha_up = lambda t: 0.0001 * (1-np.exp(-10*t))
alpha_down = lambda t: -0.0001 * (1-np.exp(-10*t))
alpha_updown = lambda t: 0.0001 * (np.cos(2*t))

alpha_tup = alpha_up(t)
alpha_tdown = alpha_down(t)
alpha_tupdown = alpha_updown(t)
df_alpha = pd.DataFrame({'t':t, 'alpha_up': alpha_tup, 'alpha_down': alpha_tdown, 'alpha_updown': alpha_tupdown})
df_alpha.plot(x='t')

#%%
from OptimalExecution import OptimalExecution, get_optimal_execution_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

X = 1
N = 100
T = 100
sigma = 0.01
lamda = 0.0
smooth_penalty = 1e1
beta=1
t=np.linspace(0, 1, N)

v1 = get_optimal_execution_curve(X, N, T, beta, 0.9, sigma, lamda, smooth_penalty, 'opt', alpha_up)
v2 = get_optimal_execution_curve(X, N, T, beta, 0.9, sigma, lamda, smooth_penalty, 'opt', alpha_down)
v3 = get_optimal_execution_curve(X, N, T, beta, 0.9, sigma, lamda, smooth_penalty, 'opt', alpha_updown)


#df_v = pd.DataFrame({'t': t, 'v(alpha_up)': v1['v']})
df_v = pd.DataFrame({'t': t, 'v(alpha_up)': v1['v'], 'v(alpha_down': v2['v'], 'v(alpha_updown)': v3['v']})
df_cum = pd.DataFrame({'t': t, 'v(alpha_up)': v1['cum_v'], 'v(alpha_down': v2['cum_v'], 'v(alpha_updown)': v3['cum_v']})



df_v.plot(x='t', figsize=(10, 5))
df_cum.plot(x='t', figsize=(10, 5))

#%%

def C(T):
    X = 1
    sigma = 0.01
    lamda = 0.1
    C = X**2 / T + 0.5 * lamda * T * X - lamda ** 2 / (sigma ** 2) * (np.exp(sigma**2 * T) - 1 - sigma ** 2 * T - 0.5 * sigma ** 4 * T ** 2)
    return C

t = np.linspace(0,30,101)
df = pd.DataFrame({'t': t, 'C': C(t)})
df.plot(x='t')
print(t)
print(C(t))

#%%
def get_optimal_T(X, a, b, Tmax):
    def A(a, b, t):
        return b * (t + (np.exp(-a * t) - 1) / a)

    def intA(a, b, t):
        return b * (0.5 * (t ** 2) - t / a - (np.exp(-a * t) - 1)/(a**2))

    def v(a, b, t, T):
        c = (X + intA(a,b,T)) / T
        return c - A(a, b, t)

    def C(T, a, b, X):
        return (X**2 + X * intA(a, b, T)) / T

    t = np.linspace(0,Tmax,100)
    c_vec = C(t, a, b, X)
    minimize_t = t[np.argmin(c_vec)]

    df_c = pd.DataFrame({'T': t, 'C(T)': c_vec})

    t = np.linspace(0,minimize_t,100)
    df_v = pd.DataFrame({'t': t, 'v': v(a, b, t, minimize_t)})
    return {'df_v': df_v, 'df_c': df_c, 'T': minimize_t}

X=1
a=10
b=-0.01
Tmax=100
res = get_optimal_T(X, a, b, Tmax)
res['df_v'].plot(x='t')
res['df_c'].plot(x='T')

print(res['T'])


#%%
X=1
a=10
b=-0.001

def A(a, b, t):
    return b * (t + (np.exp(-a * t) - 1) / a)

def intA(a, b, t):
    return b * (0.5 * (t ** 2) - t / a - (np.exp(-a * t) - 1)/(a**2))

def v(a, b, t, T):
    c = (X + intA(a,b,T)) / T
    return c - A(a, b, t)

def C(T, a, b, X):
    return (X**2 + X * intA(a, b, T)) / T

print(A(a, b, t))
print(intA(a, b, t))


T = 10
t = np.linspace(0,T,100)
df = pd.DataFrame({'t': t, 'v': v(a, b, t, T)})
df = pd.DataFrame({'t': t, 'C(T)': C(t, a, b, X)})
#
#df = pd.DataFrame({'t': t, 'A': A(a, b, t)})
df.plot(x='t')