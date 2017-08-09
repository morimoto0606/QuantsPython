import numpy as np
from scipy import interpolate
from scipy.stats import norm

def df(expiry):
	"""
	dummy yield curve
	:param expiry:
	:return: dummy discount factor
	"""
	return np.exp(-0.01 * expiry)


def vol_matrix(tenor, strike):
	'''
	dummy function for vol matrix defined by calbiration
	:param tenor: tenor year
	:param strike: strike rate
	:return: interpolated volatility
	'''
	v = np.linspace(0.1, 0.9, 9).reshape(3, 3)
	tenors = [1., 2., 3.]
	strikes = [0.01, 0.02, 0.03]
	data = interpolate.interp2d(strikes, tenors, v)
	return data(tenor, strike)


def bs_prem(fwd, sigma, strike, tau):
	d1 = (np.log(fwd / strike) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
	d2 = d1 - sigma * np.sqrt(tau)
	return fwd * norm.cdf(d1) - strike * norm.cdf(d2)

if __name__ == '__main__':
	seed = 100000
	np.random.seed(seed)
	path_num = 100000

	# 1. caplet option price
	# PV = df(T_i) * E[\tau * max(L_i(T_i) - K, 0)]
	notional = 100000000
	expiry = 10
	strike = 0.01
	tau = 0.5

	# get initial value of forward libor L(0, 10, 10 + tau)
	l_0 = 1 / tau * (df(expiry) / df(expiry + tau) - 1)
	print('l_0', l_0)
	# get vol
	vol = vol_matrix(expiry, strike)[0]
	print('vol', vol)
	
	# generate path
	# dL_i(t) = sigma_i * L_i dw, i.e L_i = L0 * exp(sigma* W -0.5 * sigma ** 2 t)
	w = np.random.randn(path_num)
	libor = l_0 * np.exp(vol * np.sqrt(expiry) * w - 0.5 * vol ** 2 * expiry)
	print('E[w]', np.sum(w) / path_num)

	# 1-3 caplet valuation
	caplet = notional * df(expiry) * tau * np.sum(np.where(libor - strike > 0, libor - strike, 0)) / path_num
	print('caplet  monte', caplet)
	expected = notional * df(expiry) * tau * bs_prem(l_0, vol, strike, expiry)
	print('caplet expect', expected)
	
	# 2. swap variation
	# PV = \sum_{i=1}^n (\tau * (L_i(T_i) - K) * df(T_i))
	start_dates = np.linspace(0, 9.5, 20)
	end_dates = np.linspace(0.5, 10, 20)
	print('start_dates', start_dates)
	print('end_dates', end_dates)

	vol = vol_matrix(start_dates, strike)
	print('vol', vol)
	# calculate initial libors
	df_start = df(start_dates)
	df_end = df(end_dates)
	print('df_start', df_start)
	print('df_end', df_end)
	
	initial_libors = 1 / tau * (df_start / df_end - 1)
	print('initial_libors', initial_libors)
	
	# generate paths
	w = np.random.randn(path_num, initial_libors.size)
	libors = initial_libors * np.exp(vol * np.sqrt(start_dates) * w - 0.5 * vol * vol * start_dates)

	print('libors', libors)
	swap = notional * np.dot(tau * (np.sum(libors, axis=0) / path_num - strike), df_end)
	swap_analytic = notional * ((df(start_dates[0]) - df(end_dates[-1])) - strike * np.sum(tau * df_end))
	print('swap_analytic', swap_analytic)
	print('swap', swap)
	
	
	# TRF
	# PV = \sum_{i=1}^n (\tau * max(L_i(T_i) - K, 0) * df(T_i)) 1_{\sum_{j=1}^i (\tau * max(L_j(T_j) - K, 0) < R}
	
	barrier = 0.1
	coupons = tau * np.where(libors - strike > 0, libors - strike, 0)
	cum_coupons = np.cumsum(coupons, axis=1)
	print('coupons', coupons)
	print('cum_coupons', cum_coupons)
	
	trf_coupon_paths = np.where(cum_coupons < barrier, cum_coupons, 0) * df_end
	print(trf_coupon_paths)
	trf_coupons = [np.max(v) for v in trf_coupon_paths]
	print(trf_coupons)
	trf_pv = notional * np.sum(trf_coupons) / path_num
	
	print('trf_pv', trf_pv)
	
