#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from homotopy_method import *

amount = 1
vol = 0.01
num_discretization = 100
expiry = 10
risk_aversion= 0.0
beta=1
gamma = 0.9
alpha_up = lambda t: 0.0001 * (1-np.exp(-10*t))
order = 2
v1 = get_homotopy_curve(
    amount=amount, 
    vol=vol, 
    alpha=alpha_up,
    beta=beta, 
    gamma=gamma,
    risk_aversion=risk_aversion,
    expiry=expiry,
    num_discretization=num_discretization,
    order=order,
    ini = [0.1, 0])

df_v = pd.DataFrame({'t': t, 'v(alpha_up)': v1['v'], 'v(alpha_down': v2['v'], 'v(alpha_updown)': v3['v']})
df_cum = pd.DataFrame({'t': t, 'v(alpha_up)': v1['cum_v'], 'v(alpha_down': v2['cum_v'], 'v(alpha_updown)': v3['cum_v']})


df_v.plot(x='t', figsize=(10, 5))
df_cum.plot(x='t', figsize=(10, 5))

# %%
