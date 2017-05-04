import numpy
import math
import sys

def generate_bm(grid_data, seed):
    dt = numpy.diff(grid_data)
    numpy.random.seed(seed)
    x = numpy.random.randn(len(dt))
    dw = numpy.multiply(numpy.sqrt(dt), x)
    return (dt, dw)


    print(grid_data)
    print(dt)
    print(x)
    print(dw)

grid_data = [0, 0.5, 1.0]
generate_bm(grid_data, 1)


def create_bs_path_generator(spot, r, sigma):
    '''
    create_bs_path
    :param spot: 
    :param r: 
    :param sigma: 
    :return: 
    '''
    drift = r - 0.5 * sigma * sigma

    def get_path(dt, dw):
        drift_array = drift * numpy.array(dt)
        diffusion_array = sigma * numpy.array(dt)
        print("drift_array")
        print(drift_array)
        print(drift + diffusion_array)
        log_increment = numpy.insert(
            drift_array + diffusion_array, 0, 0)
        print(log_increment)
        log_increment = numpy.cumsum(log_increment)
        print(log_increment)
        path = spot * numpy.exp(log_increment)
        print(path)
#        log_path = numpy.cumsum()

    return get_path

    def create_basis_function(state_variables):
        return
