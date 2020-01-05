import numpy as np
import tensorflow as tf
import sobol_seq
#print(tf.__version__)
#print("power")
#print(np.power(0, 0.8))
#path = np.array([1, -3, 5, 2])
#path = np.where(abs(path) <= 2, path, 0)
#print("path", path)
#shape = [1,2,3,4]
#print(sobol_seq.i4_sobol_generate(shape[0], shape[1], shape[3]))
#print(sobol_seq.i4)

path = np.array([[1,2],[np.nan,np.nan],[3,4],[5,6]])
path = path[~np.isnan(path)]
print(path.reshape(int(path.size/2),2))

ini = [3,4]
liftedini = list(ini) + list(np.identity(2).flatten())
print("liftedini", liftedini)

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
