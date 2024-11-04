#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import LOMO_target_helper as lth
plt.style.use('seaborn')


# initial various parameters
T = 60
Ndt = 6000
dt = T / Ndt
t = np.arange(0, T + dt, dt)
tau = T - t
gamma = 0.05
gamma2 = 0.1

t = np.arange(0, T + dt, dt)
q0 = 10
df_alm = pd.DataFrame({'t': t, 'q': q0 * np.sinh(gamma * tau) / np.sinh(gamma * T)})
df_alm2 = pd.DataFrame({'t': t, 'q': q0 * np.sinh(gamma2 * tau) / np.sinh(gamma2 * T)})

df_twap = pd.DataFrame({'t': t, 'q': q0 * tau / T})
ax = df_twap.plot(kind='line', x='t', y='q', color='r', label='Almgren-Chriss kappa=0 (TWAP)')
ax.set_ylabel(r'$Q_t$')  
# dfのラインプロット
df_alm.plot(kind='line', x='t', y='q', ax=ax, color='b', label='Almgren-Chriss kappa=0.05')
df_alm2.plot(kind='line', x='t', y='q', ax=ax, color='g', label='Almgren-Chriss kappa=0.1')

#%% Transient
gamma = 1
v = np.power((tau+1) * (T+1-tau), -1/gamma)
qv = np.cumsum(v)
print(qv)
adj = 10 / qv[-1]
qv2 =  10 - adj * (qv)
print(qv2)

#%%
df_tran = pd.DataFrame({'t': t, 'Transient market impact':qv2})
ax_tran = df_tran.plot(kind='line', x='t', label='transient')
df_twap.plot(kind='line', x='t', y='q', linestyle='--', ax=ax_tran, color='r', label='Almgren-Chriss kappa=0 (TWAP)')
df_alm.plot(kind='line', x='t', y='q', linestyle='--', ax=ax_tran, color='g', label='Almgren-Chriss kappa=0.1')
plt.show()




#%%

# price jump sizes
sigma = 0.01
kappa = 100
xi = 0.01
alpha = 0.001
lamb = 50 / 60
Nq = 10
phi = 0.001
sigma = 0.01

## AC Solution
phiAC = 10 ** (-5)
aAC = 0.001
#
#
## %%
# Almbren-Chriss 
qAC = lth.AC_solver(phiAC, aAC, tau, T, Nq)

# HJB
q = np.arange(0, Nq + 1, 1)
q = np.linspace(0, Nq, 11)
q


#%% Normal case
omega, exe = lth.hjb_solver(t, dt, kappa, xi, phi, q, qAC, lamb)
t_opt = lth.find_opt_t(exe, t)
optimal_exe = pd.DataFrame({'tau': t_opt, 'q':q})

# プロット用の共通のaxを作成
ax = optimal_exe.plot(kind='scatter', x='tau', y='q', color='r', label='market order')
# dfのラインプロット
df = pd.DataFrame({'t': t, 'q': q0 * np.sinh(gamma2 * tau) / np.sinh(gamma2 * T)})
df.plot(kind='line', x='t', y='q', ax=ax, color='b', label='Almgren-Chriss benchmark')
# グラフを表示
plt.show()


# Solve for delta
delta = lth.find_delta(kappa, omega, Nq, Ndt)
print(delta)

delta.shape
df_delta = pd.DataFrame({'t': t})
for i, d in enumerate(delta):
    title = f'delta_Q={i}'
    df_delta[title] = d
df_delta.plot(x='t', y=['delta_Q=1', 'delta_Q=3', 'delta_Q=5', 'delta_Q=7', 'delta_Q=9'])
plt.title('Delta for limit order')


# %% solve the QVI (large market impact for limit order, enable to see internal and limit order)
lambL = 50/60
lambI = 60/60
alphaL = 1e-6
alphaM = 1e-5
betaM = 1
gammaM =2 
print(f'lambL={lambL}, lambI={lambI}, alphaL={alphaL}, alphaM={alphaM}, betaM={betaM}, gammaM={gammaM}')

omega, exe = lth.hjb_solver(t, dt, kappa, xi, phi, q, qAC, lamb)
h2, exe2, df2, delta_l, delta_i, zeta \
    = lth.hjb_solver_internal_flow_power(t, dt, kappa, xi, phi, q, qAC, lambL, lambI, alphaL, alphaM, betaM, gammaM)

# obtain the optimal time at which to execute market orders
t_opt = lth.find_opt_t(exe, t)
#%%
t_opt2, q_opt2 = lth.find_opt_t_q(exe2, t, zeta)
print(t_opt)
print(t_opt2)
print(q_opt2)
print(delta_l, delta_i, zeta)
optimal_exe = pd.DataFrame({'tau': t_opt, 'q':q})
optimal_exe2 = pd.DataFrame({'tau': t_opt2, 'q':q})
optimal_exe2 = pd.DataFrame({'tau': t_opt2, 'q':q, 'q_opt': q_opt2})

display(optimal_exe)
display(optimal_exe2)

#%%
df2.plot()
#%%


# プロット用の共通のaxを作成
ax = optimal_exe.plot(kind='scatter', x='tau', y='q', color='r', label='market order')
optimal_exe2.plot(kind='scatter', x='tau', y='q', color='g', ax=ax, label='market order with internal')  # 追加するscatter
# dfのラインプロット
df.plot(kind='line', x='t', y='q', ax=ax, color='b', label='Almgren-Chriss benchmark')
plt.title('market order')
# グラフを表示
plt.show()

# 1つ目の df_delta グラフ (実線)
df_delta_i = pd.DataFrame({'t': t})
for i, d in enumerate(delta_i):
    title = f'delta_I Q={i}'
    df_delta_i[title] = d
ax = df_delta_i.plot(x='t', y=['delta_I Q=1', 'delta_I Q=3', 'delta_I Q=5', 'delta_I Q=7', 'delta_I Q=9'], linestyle='-', ax=None)

# 2つ目の df_delta グラフ (点線)
df_delta_l = pd.DataFrame({'t': t})
for i, d in enumerate(delta_l):
    title = f'delta_L Q={i}'
    df_delta_l[title] = d
df_delta_l.plot(x='t', y=['delta_L Q=1', 'delta_L Q=3', 'delta_L Q=5', 'delta_L Q=7', 'delta_L Q=9'], linestyle='--', ax=ax)
plt.title('Delta for limit and internal order')
# グラフを表示
plt.show()



# %% solve the QVI for omega (small market impact for limit order)
lambL = 50/60
lambI = 60/60
alphaL = 0.005
alphaM = 0.05
betaM = 0.0
gammaM = 0.5
print(f'lambL={lambL}, lambI={lambI}, alphaL={alphaL}, alphaM={alphaM}, betaM={betaM}, gammaM={gammaM}')

omega, exe = lth.hjb_solver(t, dt, kappa, xi, phi, q, qAC, lamb)
h2, exe2, df2, delta_l, delta_i, zeta \
    = lth.hjb_solver_internal_flow_power(t, dt, kappa, xi, phi, q, qAC, lambL, lambI, alphaL, alphaM, betaM, gammaM)


# obtain the optimal time at which to execute market orders
t_opt = lth.find_opt_t(exe, t)
t_opt2 = lth.find_opt_t(exe2, t)
print(t_opt)
print(t_opt2)
print(delta_l, delta_i, zeta)
optimal_exe = pd.DataFrame({'tau': t_opt, 'q':q})
optimal_exe2 = pd.DataFrame({'tau': t_opt2, 'q':q})
display(optimal_exe)
display(optimal_exe2)



# プロット用の共通のaxを作成
ax = optimal_exe.plot(kind='scatter', x='tau', y='q', color='r', label='market order')
optimal_exe2.plot(kind='scatter', x='tau', y='q', color='g', ax=ax, label='market order with internal')  # 追加するscatter
# dfのラインプロット
df.plot(kind='line', x='t', y='q', ax=ax, color='b', label='Almgren-Chriss benchmark')
plt.title('market order')
# グラフを表示
plt.show()

# 1つ目の df_delta グラフ (実線)
df_delta_i = pd.DataFrame({'t': t})
for i, d in enumerate(delta_i):
    title = f'delta_I Q={i}'
    df_delta_i[title] = d
ax = df_delta_i.plot(x='t', y=['delta_I Q=1', 'delta_I Q=3', 'delta_I Q=5', 'delta_I Q=7', 'delta_I Q=9'], linestyle='-', ax=None)

# 2つ目の df_delta グラフ (点線)
df_delta_l = pd.DataFrame({'t': t})
for i, d in enumerate(delta_l):
    title = f'delta_L Q={i}'
    df_delta_l[title] = d
df_delta_l.plot(x='t', y=['delta_L Q=1', 'delta_L Q=3', 'delta_L Q=5', 'delta_L Q=7', 'delta_L Q=9'], linestyle='--', ax=ax)
plt.title('Delta for limit and internal order')
# グラフを表示
plt.show()



# %%
