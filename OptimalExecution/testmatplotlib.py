#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#%%
import numpy as np
from numpy.core.fromnumeric import argmin
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


#%%
x = np.linspace(0, 10, 100)
fig = plt.figure()
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()

# %%
