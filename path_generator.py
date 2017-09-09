import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
rnd.seed(1)
print(rnd.randn(10))
x = rnd.randn(100000)
print(x)
plt.hist(x, 100)
plt.show()

time_grid = np.array([0.1, 0.2, 0.3])

 
def path_generator(mu, sigma, dim, seed):
    rnd.seed(seed)
    def generator():
        w = rnd.randn(dim)

def evolve(prev, mu, sigma, dt, dw):
    return prev * np.exp((mu - 0.5 * sigma ** 2) * dt
                         + sigma * dw())





