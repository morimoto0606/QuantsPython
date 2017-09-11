import time
import numpy as np


def calac_exponenial(num):
    v = np.array([10. for x in range(1, 1000000)])
    start = time.time()
    w = np.exp(v)
    print time.time() - start


if __name__ == '__main__':
    num = 10000000
    calac_exponenial(num)
