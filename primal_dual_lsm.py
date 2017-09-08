#
# valuation of american options
# with least-square monte carlo

import math
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
import itertools as it
import warnings
warnings.simplefilter('ignore')

t0 = time()
print(t0)
np.random.seed(150000)

# simulation parameters
runs = 5
write = True
otype = [1, 2]
M = [10, 20]
I1 = np.array([4, 6]) * 4096
I2 = np.array([1, 2]) * 1024
J = [50, 75]
reg = [5, 9]
AP = [False, True]
MM = [False, True]
ITM = [True, False]
results = pd.DataFrame()

#
# Function definition
#

def generate_random_numbers(I):
    '''
    function to generate pseudo-random numbers
    :param I:number of random numbers 
    :return:pseudo-random numbers 
    '''
    if AP:
        ran = np.random.standard_normal(I / 2)
        ran = np.concatenate((ran, -ran))
    else:
        ran = np.random.standard_normal(I)
    if MM: #moment much
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    return ran

def generate_paths(I):
    '''
    function to generate stock price paths.
    :param I:number of path 
    :return:stock price paths 
    '''
    s = np.zeros((M + 1, I))
    s[0] = s0
    for t in range(1, M + 1, 1):
        ran = generate_random_numbers(I)
        s[t] = s[t - 1] * np.exp((r - sigma ** 2) * dt
                                 + sigma * ran * math.sqrt(dt))
    return s


def inner_values(s):
    '''Inner value functions for american put and short condor spread'''
    if otype == 1:
        return np.maximum(40. - s, 0)
    else:
        return np.minimum(10., np.maximum(90. - s, 0)
                          + np.maximum(s - 110., 0))


def nested_monte_carlo(St, J):
    '''
    function for nested monte carlo simulaiton.
    :param st: start value for s
    :param j: int number of paths to simulate 
    :return: paths array simulated nested paths
    '''
    ran = generate_random_numbers(J)
    paths = St * np.exp((r - 0.5 * sigma ** 2) * dt
                        + sigma * ran * math.sqrt(dt))
    return paths

# valuation

para = it.product(otype, M, I1, I2, J, reg, AP, MM, ITM)
count = 0
for pa in para:
    otype, M, I1, I2, J, reg, Ap, MM, ITM = pa
    if otype == 1:
        s0 = 0.36
        T = 1.0
        r = 0.06
        sigma = 0.2
        v0_true = 4.48637

    dt = T / M
    df = math.exp(-r * dt)

    for j in range(runs):
        count += 1
        S = generate_paths(I1)
        h = inner_values(S)
        V = inner_values(S)
        rg = np.zeros((M + 1, reg + 1), dype=np.float)

        item = np.generater(h, 0)
        for t in range(M-1, 0, -1):
            if ITM:
                S_itm = np.compress(item[t] == 1, S[t])
                V_itm = np.compress(item[t] == 1, V[t+1])
                if len(V_itm) == 0:
                    rg[t] == 0.0
                else:
                    rg[t] = np.polyfit(S_itm, V_itm * df, reg)
            else:
                rg[t] = np.polyfit(S[t], V[t+1] * df, reg)

            C = np.polyval(rg[t], S[t])
            V[t] = np.where(h[t] > C, h[t], V[t+1] * df)

        V0 = df * df * np.sum(V[1]) / I2

        ## Dual Valuation
        for t in range(1, M + 1):
            for i in range(I2):
                Vt = max(h[t, i], np.polyval(reg[t], S[t, i]))
                St = nested_monte_carlo(S[t-1, i], J) #nested MCS
                Ct = np.polyval(rg[t], St);
                ht = inner_values(St)
                VtJ = np.sum(np.where(ht > Ct, ht, Ct)) / len(St)
                Q[t, i] = Q[t - 1, i] / df + (Vt - VtJ)
                U[t, i] = max(U[t - 1, i] / df, h[t, i] - Q[t, i])
                if t == M:
                    U[t, i] = np.maximum(U[t - 1, i] / df, np.mean(ht) - Q[t, i])
        U0 = np.sum(U[M])
        AV = 0.5 * (V0 + U0)

        #output
        print(V0, U0, AV)
