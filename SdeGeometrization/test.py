import numpy as np
import tensorflow as tf
import sobol_seq
print(tf.__version__)
print("power")
print(np.power(0, 0.8))
path = np.array([1, -3, 5, 2])
path = np.where(abs(path) <= 2, path, 0)
print("path", path)
vec, seed = sobol_seq.i4_sobol(4, 1)
print(vec, seed)
print(sobol_seq.i4_sobol_generate_std_normal(3, 5))
#tfe.enable_eager_execution()

#def f(x, y):
#    return x**2 + y**2 + 3*y
#
#grad_f = tfe.gradients_function(f)
#
#x = 2.0
#y = 3.0
#
#print(grad_f(x, y)[0].numpy())
#print(grad_f(x, y))
#
