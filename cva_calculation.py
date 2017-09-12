import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import csv


def generate_normal_path_dict(s0, path_num, grid, r, sigma, seed):
    """
    generate path matrix(grid_num + 1, path_num)
    :param s0: initial value
    :param path_num: number of path
    :param grid: number of grid t1 < ... < tn
    :param r: interest rate
    :param sigma: volatility
    :param seed : seed
    :return: path matrix on t0 = 0 < t1 < ... < tn
    """
    dt = np.diff(grid)[:, np.newaxis]
    print('grid', grid)
    print('dt', dt)
    print('size', dt.size)
    np.random.seed(seed)
    dw = np.random.standard_normal((path_num, dt.size))
    #dw = np.random.randn(path_num, dt.size)
    print('E[dw]', np.sum(dw, axis=0) / path_num)
    dw = np.sqrt(dt) * np.transpose(dw)
    d_s = r * dt + sigma * dw
    # add zero vector
    d_s = np.vstack((np.zeros((1, path_num)), d_s))
    path = s0 + np.cumsum(d_s, axis=0)
    return dict(zip(grid, path))


def generate_lognormal_path_dict(s0, path_num, grid, r, sigma, seed):
    """
    generate path matrix(grid_num + 1, path_num)
    :param s0: initial value
    :param path_num: number of path
    :param grid: number of grid t1 < ... < tn
    :param r: interest rate
    :param sigma: volatility
    :param seed : seed
    :return: path matrix on t0 = 0 < t1 < ... < tn
    """
    dt = np.diff(grid)[:, np.newaxis]
    print('grid', grid)
    print('dt', dt)
    print('size', dt.size)
    np.random.seed(seed)
    dw = np.random.standard_normal((path_num, dt.size))
    print('E[dw]', np.sum(dw, axis=0) / path_num)
    dw = np.sqrt(dt) * np.transpose(dw)
    d_log_s = (r - 0.5 * sigma * sigma) * dt + sigma * dw
    # add zero vector
    d_log_s = np.vstack((np.zeros((1, path_num)), d_log_s))
    path = s0 * np.exp(np.cumsum(d_log_s, axis=0))
    return dict(zip(grid, path))


def normal_pdf(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * math.pi)


def pdf_log_normal(x, y, mu, sigma, tau):
    lognorm_pdf = lambda z: normal_pdf(
        ((np.log(y) - np.log(z)) - (mu - 0.5 * sigma ** 2) * tau)
        / (sigma * np.sqrt(tau)))
    return lognorm_pdf(x)

#    if isinstance(x, np.ndarray):
#        return np.array([lognorm_pdf(z) for z in x])
#    else:
#        return lognorm_pdf(x)


def pdf_normal(x, y, mu, sigma, tau):
    normal_pdf = lambda z: norm.pdf((y - x) / np.sqrt(tau))
    if isinstance(x, np.ndarray):
        return np.array([normal_pdf(z) for z in x])
    else:
        return normal_pdf

def generate_stochastic_mesh(t_start, t_end, path_dict, payoff, mu, sigma, pdf):
    """
    stochastic mesh continuous function
    :param t_start: t
    :param t_end: T
    :param path_dict:
    :param payoff:
    :param pdf:
    :return:
    """
    
    tau = t_end - t_start
    denominator = lambda y: np.sum(pdf(path_dict[t_start], y, mu, sigma, tau))
    
    def c(x):
        mesh = 0
        for y in path_dict[t_end]:
            mesh += pdf(x, y, mu, sigma, tau) * payoff[t_end](y) / denominator(y)
            
        return mesh
    return c


def generate_lsm(t_start, t_end, path, payoff, degree):
    """
    LSM continuous function
    :param t_start: t
    :param t_end: T
    :param path_start: X_t
    :param payoff: F(X_T)
    :param degree: degree of polynomial for regression
    :return: (Q_t,T f)(x)
    """
    print('t_start', t_start)
    print('t_end', t_end)
    print('path[t_start]', path[t_start])
    print('payoff', payoff(path[t_end]))
    coefficient = np.polyfit(path[t_start], payoff(path[t_end]), degree)
    
    def c(x):
        return np.polyval(coefficient, x)
    return c



def generate_exposure_stochastic_mesh(t, path_dic, payoff_dict, mu, sigma, pdf):
    """
    generate exposure function at t by stochastic mesh
    :param t: future time
    :param path_dic: (key, value) = (t, time-wise path)
    :param payoff_dict: map of (T_i, F_i)i = 1,..., n
    :return: \sigma_{T>t}(Q_t,T f)(x)
    """
    
    payment_grid = np.array(list(payoff_dict.keys()))
    future_payment_grid = payment_grid[payment_grid > t]
 
    exposure_func = {}
    for t_pay in future_payment_grid:
        exposure_func[t_pay] = generate_stochastic_mesh(
            t, t_pay, path_dic, payoff_dict, mu, sigma, pdf)
        
    def c(x):
        exposure = 0
        for t_pay in future_payment_grid:
           exposure += exposure_func[t_pay](x)
        return exposure
    return c


def generate_exposure_lsm(t, path_dict, payoff_dict, deg):
    """
    generate exposure function at t by stochastic mesh
    :param t: future time
    :param path_dict: (key, value) = (t, time-wise path)
    :param payoff_dict: map of (T_i, F_i)i = 1,..., n
    :return: \sigma_{T>t}(Q_t,T f)(x)
    """
    
    payment_grid = np.array(list(payoff_dict.keys()))
    print(type(payment_grid))
    print('t', t)
    print('payment_grid', payment_grid)
    future_payment_grid = payment_grid[payment_grid > t]
    print('future_grid', future_payment_grid)
    
    future_payoff = np.zeros(path_dict[0].size)
    for s in future_payment_grid:
        future_payoff += payoff_dict[s](path_dict[s])
    
    coefficients = np.polyfit(path_dict[t], future_payoff, deg)
    
    def c(x):
        return np.polyval(coefficients, x)
    
    return c


def explicit_exe_calculator(epsilon, exposure, row_exposure, current_state):
    """
    :param epsilon: 1 for epe, -1 for ene
    :param exposure: 1 dim vector of exposure
    :param row_exposure: 1 dim vector of row exposure
    :return: epe or ene
    """
    return np.sum(np.where(epsilon * exposure > 0, exposure, 0)) / current_state.size


def implicit_exe_calculator(epsilon, exposure, row_exposure, current_state):
    """
    :param epsilon: 1 for epe, -1 for ene
    :param exposure: 1 dim vector of exposure
    :param row_exposure: 1 dim vector of row exosure
    :return: epe or ene
    """
    return np.sum(np.where(epsilon * exposure > 0, row_exposure, 0)) / current_state.size


def analytic_exe_calculator(t, epsilon, exposure):
    return np.sum(np.where(epsilon * exposure > 0, exposure, 0)) / exposure.size


def cva_calculation(
        r,
        sigma,
        s0,
        path_dict_for_simulation,
        exe_calculator,
        path_generator,
        path_num_for_sim,
        path_num_for_exposure,
        seed_for_sim,
        seed_for_exposure,
        calculation_grid,
        payment_grid,
        payoff):
  
    # generate path
    path_dict_for_exposure = path_generator(
        s0=s0,
        path_num=path_num_for_exposure,
        grid=calculation_grid,
        r=r,
        sigma=sigma,
        seed=seed_for_exposure)
 
    # evaluate cash flow on each calculation grid
    payoffs = [payoff for t in payment_grid]
    payoff_dict = dict(zip(payment_grid, payoffs))
    print('payoffs', payoffs)
    print('payoff_dict', payoff_dict)

    # make exposure by lsm
    exposure_sm = {}
    for t in calculation_grid[calculation_grid > 0.0]:
        g = exposure_calculator(t, path_dict_for_exposure, payoff_dict)
        exposure_sm[t] = g
    exposure_sm[0] = exposure_sm[1]
    # integration for cva


    epe_explicit = []
    epe_implicit = []
    epe_analytic = []

    for t in calculation_grid:
        current_state = path_dict_for_simulation[t]
        future_payoff = np.zeros(current_state.size)
        analytic_exposure = np.zeros(current_state.size)

        for s in payment_grid[payment_grid > t]:
            future_payoff += payoff_dict[s](path_dict_for_simulation[s])
            analytic_exposure += payoff_dict[s](path_dict_for_simulation[t])
        
        e_sm = exposure_sm[t](current_state)
        epe_explicit.append(explicit_exe_calculator(1, e_sm, future_payoff, current_state))
        epe_implicit.append(implicit_exe_calculator(1, e_sm, future_payoff, current_state))
        epe_analytic.append(analytic_exe_calculator(t, 1, analytic_exposure))


    dts = np.diff(calculation_grid)
    cva_explicit = np.dot(epe_explicit[1:], dts)
    cva_implicit = np.dot(epe_implicit[1:], dts)
    cva_analytic = np.dot(epe_analytic[1:], dts)


    calculation_grid = np.append(calculation_grid, [1.1])
    print('calculation_grid', calculation_grid)
    print('dts', dts)
    print('cva_explicit', cva_explicit)
    print('cva_analytic', cva_analytic)
    print('cva_implicit', cva_implicit)


    epe_explicit.append(0)
    epe_implicit.append(0)
    epe_analytic.append(0)
    np.append(calculation_grid, [1.1])

    f = open('exposure.csv', 'a')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow([''])
    writer.writerow(['path_num', 'seed', 'epe_explicit', path_num_for_exposure, seed_for_exposure, epe_explicit])
    writer.writerow(['path_num', 'seed', 'epe_implicit', path_num_for_exposure, seed_for_exposure, epe_implicit])
    writer.writerow(['path_num', 'seed', 'epe_analytic', path_num_for_exposure, seed_for_exposure, epe_analytic])
    f.close()



    #plt.plot(calculation_grid, epe_explicit, label='epe_sm', linestyle='dashed')
    #plt.plot(calculation_grid, epe_implicit, label='epe_implicit', linestyle='dashdot')
    #plt.plot(calculation_grid, epe_analytic, label='epe_analytic')
    #plt.show()

    return [cva_explicit, cva_implicit, cva_analytic]


if __name__ == '__main__':

    # path setting for Black Scholes Model
    r = 0.0
    sigma = 0.3
    s0 = 100
    # calculation setting
    # payment grid is [2, 4, ..., 8, 10]
    payment_grid = np.linspace(2.0, 10.0, 5)
    print ('payment_grid', payment_grid)
    payoff = lambda x: x - s0
    calculation_grid = np.linspace(0, 10.0, 101)
    path_num_for_simulation = 10000
    path_num_for_exposures = [500, 1000, 2000, 5000, 10000]
    seed_for_simulation = 0

    exposure_calculator = lambda t, path_dict_for_exposure, payoff_dict:\
        generate_exposure_stochastic_mesh(t, path_dict_for_exposure, payoff_dict, r, sigma, pdf_log_normal)

    path_dict_for_simulation = generate_lognormal_path_dict(s0=s0, path_num=path_num_for_simulation,
                                                            grid=calculation_grid, r=r, sigma=sigma, seed=seed_for_simulation)

    # TODO loop for seed
    for path_num_for_exposure in path_num_for_exposures:
        cva_explicit_seed = []
        cva_implicit_seed = []
        cva_analytic_seed = []
        for seed_for_exposure in range(1, 11):
            ret = cva_calculation(r, sigma, s0,
                    path_dict_for_simulation,
                    exposure_calculator,
                    generate_lognormal_path_dict,
                    path_num_for_simulation,
                    path_num_for_exposure,
                    seed_for_simulation,
                    seed_for_exposure,
                    calculation_grid,
                    payment_grid,
                    payoff)
            cva_explicit_seed.append(ret[0])
            cva_implicit_seed.append(ret[1])
            cva_analytic_seed.append(ret[2])

        g = open('cva.csv', 'a')
        writer = csv.writer(g, lineterminator='\n')
        writer.writerow([''])
        writer.writerow(['path_num', 'seed', 'cva_explicit', path_num_for_exposure, cva_explicit_seed])
        writer.writerow(['path_num', 'seed', 'cva_implicit', path_num_for_exposure, cva_implicit_seed])
        writer.writerow(['path_num', 'seed', 'cva_analytic', path_num_for_exposure, cva_analytic_seed])
        g.close()

    print('Finished !')

