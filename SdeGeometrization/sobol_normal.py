import sobol_seq
import numpy as np
from scipy.special import erfinv
from scipy.stats import norm

def generate_path(
    no_path: int,
    no_step: int,
    no_state: int):
    uniform_rnd = sobol_seq.i4_sobol_generate(
        dim_num = 1, n = no_step * no_path* no_state)
    normal = norm.ppf(uniform_rnd)
    normal = np.transpose(normal)
    normal = np.reshape(normal, (no_path, no_step, no_state))
    return normal