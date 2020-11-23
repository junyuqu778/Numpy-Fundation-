import numpy as np

np.set_printoptions(precision=3)
rand_arr = np.random.random([5, 3])

print(rand_arr)


np.set_printoptions(threshold=6)
arr1 = np.arange(20)

print(arr1)


np.set_printoptions(threshold=25)
arr2 = np.arange(20)

print(arr2)