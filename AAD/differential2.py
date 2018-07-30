import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

def f(x, y):
    return x**2 + y**2 + 3*y

grad_f = tfe.gradients_function(f)

x = 2.0
y = 3.0
print(grad_f(x, y)[0].numpy())
print(grad_f(x, y)[1])

def g(x):
    return tf.square(tf.sin(x))

#print(test(100, 0.3))
gradf = tfe.gradients_function(g)
print(gradf(100.0)[0].numpy())
#
