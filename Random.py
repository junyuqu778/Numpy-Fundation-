import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#x = np.arange(20).reshape((5, 4))
#print(x)


np.random.seed(20200614)
x = np.random.uniform(-10, 10, size=5000)
y=x.copy()
y[y==0] =1
y[y!=0] = np.sin(y)/y
new_y = y.copy()
y += np.random.uniform(-0.2,0.2,size=5000)
print(y)
plt.scatter(x,y,label='true')
plt.scatter(x,new_y,label='experted')
plt.legend()
plt.show()
