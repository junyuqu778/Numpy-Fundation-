from numpy.linalg import inv
from numpy import dot, transpose

X = [[1, 6, 2] , [1, 8, 1] , [1, 10, 0] , [1 , 14, 2] , [1, 18, 0]]
y = [[7] , [9] , [13] , [17.5] , [18]]
print(dot(inv(dot(transpose(X) , X)) , dot(transpose(X) , y)))

###################################

import numpy as np

np.random.seed(100)
a = np.random.randint(1, 10, [5, 3])

print(a)
print(np.amax(a))
print(np.amax(a, axis=0))
print(np.amax(a, axis=1))


