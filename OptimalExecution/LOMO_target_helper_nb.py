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

# v(t) = sinh(gamma(T-t))
gamma = 0.1

a = np.sinh(gamma * T)
t = np.arange(0, T + dt, dt)
df = pd.DataFrame({'t': t, 'v': 10 * np.sinh(gamma * tau) / a})
df.plot(x='t')


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
#df_qac = pd.DataFrame({'t':t, 'qAC':qAC})
#df_qac.plot(x='t')

# HJB
q = np.arange(0, Nq + 1, 1)
q = np.linspace(0, Nq, 11)
q

# %%
# solve the QVI for omega
omega, exe = lth.hjb_solver(t, dt, kappa, xi, phi, q, qAC, lamb)
omega2, exe2, df2, delta2 = lth.hjb_solver_optimize(t, dt, kappa, xi, phi, q, qAC, lamb)
omega3, exe3 = lth.hjb_solver_loop(t, dt, kappa, xi, phi, q, qAC, lamb)


#%%
# obtain the optimal time at which to execute market orders
t_opt = lth.find_opt_t(exe, t)
t_opt2 = lth.find_opt_t(exe2, t)
t_opt3 = lth.find_opt_t(exe3, t)
print(t_opt)
print(t_opt2)
print(t_opt3)
#%%
##%%
#df2
##%%
#ax = df2.plot(x='delta', y='func', color='blue', label='func')
## dfuncを右のy軸に追加
#df.plot(x='delta', y='dfunc', color='red', label='dfunc', secondary_y=True, ax=ax)
## グラフの表示
#plt.show()
##
##%%
#t_opt2 = lth.find_opt_t(exe2, t)
#print(t_opt)
#print(t_opt2)
#
#
#
##%%
#qAC_opt = lth.AC_solver(phiAC, aAC, T-t_opt, T, Nq)
#qAC_opt2 = lth.AC_solver(phiAC, aAC, T-t_opt2, T, Nq)
#


# %%
optimal_exe = pd.DataFrame({'tau': t_opt, 'q':q})
optimal_exe2 = pd.DataFrame({'tau': t_opt2, 'q':q})
optimal_exe3 = pd.DataFrame({'tau': t_opt3, 'q':q})
display(optimal_exe)
display(optimal_exe2)
display(optimal_exe3)



# %% plot q-opt and almgren
ax = optimal_exe.plot(kind='scatter', x='tau', y='q', color='r')
df.plot(kind='line', x='t', y='v', ax=ax, color='b')
plt.show()

# %% plot q-opt and almgren
ax = optimal_exe2.plot(kind='scatter', x='tau', y='q', color='r')
df.plot(kind='line', x='t', y='v', ax=ax, color='b')
plt.show()



#%%
optimal_exe

# %%
optimal_exe['lambda'] = optimal_exe['q'] / (61-optimal_exe['tau'])
optimal_exe

# %%
# Solve for delta
delta = lth.find_delta(kappa, omega, Nq, Ndt)
print(delta)
#%%
delta.shape
df_delta = pd.DataFrame({'t': t})
for i, d in enumerate(delta):
    title = f'delta_Q={10-i}'
    df_delta[title] = d
df_delta.plot(x='t', y=['delta_Q=1', 'delta_Q=3', 'delta_Q=5', 'delta_Q=7', 'delta_Q=9'])
#%%
# Solve for delta
print(delta2)
#%%
delta.shape
df_delta = pd.DataFrame({'t': t})
for i, d in enumerate(delta2):
    title = f'delta_Q={10-i}'
    df_delta[title] = d
df_delta.plot(x='t', y=['delta_Q=1', 'delta_Q=3', 'delta_Q=5', 'delta_Q=7', 'delta_Q=9'])



# %%
# plot the results
lth.plot_topt(t_opt, q, qAC, t)

# %%
lth.plot_multi_lines(t[0:(t.shape[0]-1)], delta[:, 0:(delta.shape[1]-1)],  xlab=r"Time ($sec$)", ylab=r"$\delta^*(t,q)$", title=r"Optimal Limit Order Depth")

# %%
df_delta = optimal_exe.plot(kind='scatter', x='tau', y='q', color='r')
df.plot(kind='line', x='t', y='v', ax=ax, color='b')
plt.show()

#%%
#################
# HJB enhanced
q = np.linspace(0, Nq, 11)
lamb_P = 50/60
lamb_I = 40/60
alpha_P = 1e-4

omega_imp, exe_imp = lth.hjb_solver_maket_impact(t, dt, kappa, xi, phi, q, qAC, lamb_P, lamb_I, alpha_P)
omega_imp

# %%
t_opt_imp = lth.find_opt_t(exe_imp, t)
print(t_opt)
print(t_opt_imp)


# %%
optimal_exe_imp = pd.DataFrame({'tau': t_opt_imp, 'q':q})


# %% plot q-opt and almgren
ax = optimal_exe_imp.plot(kind='scatter', x='tau', y='q', color='r')
df.plot(kind='line', x='t', y='v', ax=ax, color='b')
plt.show()


# %%
optimal_exe['lambda'] = optimal_exe['q'] / (61-optimal_exe['tau'])
optimal_exe

# %%
# Solve for delta
delta = lth.find_delta(kappa, omega, Nq, Ndt)

#%%
delta.shape
df_delta = pd.DataFrame({'t': t})
for i, d in enumerate(delta):
    title = f'delta_Q={10-i}'
    df_delta[title] = d
df_delta.plot(x='t', y=['delta_Q=1', 'delta_Q=3', 'delta_Q=5', 'delta_Q=7', 'delta_Q=9'])


# %%
# plot the results
lth.plot_topt(t_opt, q, qAC, t)

# %%
lth.plot_multi_lines(t[0:(t.shape[0]-1)], delta[:, 0:(delta.shape[1]-1)],  xlab=r"Time ($sec$)", ylab=r"$\delta^*(t,q)$", title=r"Optimal Limit Order Depth")

# %%
df_delta = optimal_exe.plot(kind='scatter', x='tau', y='q', color='r')
df.plot(kind='line', x='t', y='v', ax=ax, color='b')
plt.show()

