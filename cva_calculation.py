import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
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
    #dw = np.random.standard_normal((path_num, dt.size))
    dw = np.random.randn(path_num, dt.size)
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


def pdf_log_normal(x, y, mu, sigma, tau):
    lognorm_pdf = lambda z: norm.pdf(
        ((np.log(y) - np.log(z)) - (mu - 0.5 * sigma ** 2) * tau)
        / (sigma * np.sqrt(tau)))

    if isinstance(x, np.ndarray):
        return np.array([lognorm_pdf(z) for z in x])
    else:
        return lognorm_pdf(x)


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


def cva_calculation(
        r,
        sigma,
        s0,
        exposure_calculator,
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
        #g = generate_exposure_stochastic_mesh(t, path_dict_for_exposure,
        #                                      payoff_dict, r, sigma, pdf)
        g = exposure_calculator(t, path_dict_for_exposure, payoff_dict)
        exposure_sm[t] = g
    
    # integration for cva
    path_dict_for_simulation = path_generator(s0=s0, path_num=path_num_for_sim,
                                                  grid=calculation_grid, r=r, sigma=sigma,
                                                  seed=seed_for_sim)
    
    ee_sm = []
    epe_sm = []
    ene_sm = []
    for t in calculation_grid[calculation_grid > 0]:
        current_state = path_dict_for_simulation[t]
        future_payoff = np.zeros(current_state.size)
        
        for s in payment_grid[payment_grid > t]:
            future_payoff += payoff_dict[s](path_dict_for_simulation[s])
        
        e_sm = exposure_sm[t](current_state)
        ee_sm.append(np.sum(e_sm) / current_state.size)
        epe_sm.append(exe_calculator(1, e_sm, future_payoff, current_state))
        ene_sm.append(exe_calculator(-1, e_sm, future_payoff, current_state))

    f = open('exposure_implicit_sm.csv', 'a')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow([''])
    writer.writerow(['ee_sm', path_num_for_exposure, seed_for_exposure, ee_sm])
    writer.writerow(['epe_sm', path_num_for_exposure, seed_for_exposure, epe_sm])
    writer.writerow(['ene_sm', path_num_for_exposure, seed_for_exposure, ene_sm])
    f.close()
    
    print('ee_sm', ee_sm)
    print('epe_sm', epe_sm)
    print('ene_sm', ene_sm)

    dts = np.diff(calculation_grid)
    cva = np.dot(epe_sm, dts)
    dva = np.dot(ene_sm, dts)

    print('calculation_grid', calculation_grid)
    print('dts', dts)
    print('cva', cva)
    print('dva', dva)
    g = open('cva_implicit_sm.csv', 'a')
    writer = csv.writer(g, lineterminator='\n')
    writer.writerow([path_num_for_exposure, seed_for_exposure, cva, dva])
    g.close()
    
    # ee_sm.append(0)
    # epe_sm.append(0)
    # ene_sm.append(0)
    # plt.plot(calculation_grid, ee_sm, label='ee_sm')
    # plt.plot(calculation_grid, epe_sm, label='epe_sm')
    # plt.plot(calculation_grid, ene_sm, label='ene_sm')
    # plt.legend()
    # plt.show()
    #
    return 0


if __name__ == '__main__':
    # path setting
    r = 0
    sigma = 1
    s0 = 0
    
    # payoff setting
    payment_grid = np.array([1.0])
    payoff = lambda x: x
    calculation_grid = np.linspace(0, 1.0, 11)
    path_num_for_simulation = 100
    seed_for_simulation = 10000
    
    path_num_for_exposures = [100]
    
    exposure_calculator = lambda t, path_dict_for_exposure, payoff_dict:\
        generate_exposure_stochastic_mesh(t, path_dict_for_exposure, payoff_dict, r, sigma, pdf_normal)
    for path_num_for_exposure in path_num_for_exposures:
        for seed in range(0, 1):
            cva_calculation(r, sigma, s0,
                            exposure_calculator,
                            implicit_exe_calculator,
                            generate_normal_path_dict,
                            path_num_for_simulation,
                            path_num_for_exposure,
                            seed_for_simulation,
                            seed_for_simulation + seed,
                            calculation_grid,
                            payment_grid,
                            payoff)
       
    r = 0.03
    sigma = 0.3
    s0 = 100
    # calculation setting
    payment_grid = np.linspace(2.0, 10.0, 5)
    
    payoff = lambda x: x - s0
    calculation_grid = np.linspace(0, 10.0, 11)
    path_num_for_simulation = 100
    path_num_for_exposure = 100
    seed_for_simulation = 10000
    seed_for_exposure = 10000
    degree = 3
   
    exposure_calculator = lambda t, path_dict_for_exposure, payoff_dict:\
        generate_exposure_stochastic_mesh(t, path_dict_for_exposure, payoff_dict, r, sigma, pdf_log_normal)
    cva_calculation(r, sigma, s0,
                       exposure_calculator,
                       implicit_exe_calculator,
                       generate_lognormal_path_dict,
                       path_num_for_simulation,
                       path_num_for_exposure,
                       seed_for_simulation,
                       seed_for_exposure,
                       calculation_grid,
                       payment_grid,
                       payoff)
    
    f = open('exposure_implicit_sm.csv', 'r')
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        print(row)
    f.close()
    print('hello world')

