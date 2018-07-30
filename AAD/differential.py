print("Hello, Python3")

import numpy as np

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def func_x0(x0):
    return x0**2 + 4**2

def func_x1(x1):
    return 3**2 + x1**2

d0 = numerical_diff(func_x0, 3) 
d1 = numerical_diff(func_x1, 4)
print(d0)
print(d1)

from math import pi
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

def f(x):
    return tf.square(tf.sin(x))

def grad(f):
    return lambda x: tfe.gradients_function(f)(x)[0]

import matplotlib.pyplot as plt

x = tf.lin_space(-2*pi, 2*pi,100)
plt.plot(x, f(x), label="f")
plt.plot(x, grad(f)(x), label="second deriv")
plt.legend()
plt.show()

