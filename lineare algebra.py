import numpy as np

# 创建一个对角矩阵！
x = np.diag((1, 2, 3))
print(x)
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

print(np.linalg.eigvals(x))
# [1. 2. 3.]

a, b = np.linalg.eig(x)
# 特征值保存在a中，特征向量保存在b中
print(a)
# [1. 2. 3.]
print(b)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 检验特征值与特征向量是否正确
for i in range(3):
    if np.allclose(a[i] * b[:, i], np.dot(x, b[:, i])):
        print('Right')
    else:
        print('Error')
# Right
# Right
# Right


import numpy as np

A = np.array([[4, 11, 14], [8, 7, -2]])
print(A)
# [[ 4 11 14]
#  [ 8  7 -2]]

u, s, vh = np.linalg.svd(A, full_matrices=False)
print(u.shape)  # (2, 2)
print(u)
# [[-0.9486833  -0.31622777]
#  [-0.31622777  0.9486833 ]]

print(s.shape)  # (2,)
print(np.diag(s))
# [[18.97366596  0.        ]
#  [ 0.          9.48683298]]

print(vh.shape)  # (2, 3)
print(vh)
# [[-0.33333333 -0.66666667 -0.66666667]
#  [ 0.66666667  0.33333333 -0.66666667]]

a = np.dot(u, np.diag(s))
a = np.dot(a, vh)
print(a)
# [[ 4. 11. 14.]
#  [ 8.  7. -2.]]

import numpy as np

A = np.array([[1, 1], [1, -2], [2, 1]])
print(A)
# [[ 1  1]
#  [ 1 -2]
#  [ 2  1]]

u, s, vh = np.linalg.svd(A, full_matrices=False)
print(u.shape)  # (3, 2)
print(u)
# [[-5.34522484e-01 -1.11022302e-16]
#  [ 2.67261242e-01 -9.48683298e-01]
#  [-8.01783726e-01 -3.16227766e-01]]

print(s.shape)  # (2,)
print(np.diag(s))
# [[2.64575131 0.        ]
#  [0.         2.23606798]]

print(vh.shape)  # (2, 2)
print(vh)
# [[-0.70710678 -0.70710678]
#  [-0.70710678  0.70710678]]

a = np.dot(u, np.diag(s))
a = np.dot(a, vh)
print(a)
# [[ 1.  1.]
#  [ 1. -2.]
#  [ 2.  1.]]

import numpy as np

A = np.array([[1, 1], [1, -2], [2, 1]])
print(A)
# [[ 1  1]
#  [ 1 -2]
#  [ 2  1]]

q, r = np.linalg.qr(A)
print(q.shape)  # (3, 2)
print(q)
# [[-0.40824829  0.34503278]
#  [-0.40824829 -0.89708523]
#  [-0.81649658  0.27602622]]

print(r.shape)  # (2, 2)
print(r)
# [[-2.44948974 -0.40824829]
#  [ 0.          2.41522946]]

print(np.dot(q, r))
# [[ 1.  1.]
#  [ 1. -2.]
#  [ 2.  1.]]

a = np.allclose(np.dot(q.T, q), np.eye(2))
print(a)  # True   （说明q为正交矩阵）

import numpy as np

A = np.array([[1, 1, 1, 1], [1, 3, 3, 3],
              [1, 3, 5, 5], [1, 3, 5, 7]])
print(A)
# [[1 1 1 1]
#  [1 3 3 3]
#  [1 3 5 5]
#  [1 3 5 7]]

print(np.linalg.eigvals(A))
# [13.13707118  1.6199144   0.51978306  0.72323135]

L = np.linalg.cholesky(A)
print(L)
# [[1.         0.         0.         0.        ]
#  [1.         1.41421356 0.         0.        ]
#  [1.         1.41421356 1.41421356 0.        ]
#  [1.         1.41421356 1.41421356 1.41421356]]

print(np.dot(L, L.T))
# [[1. 1. 1. 1.]
#  [1. 3. 3. 3.]
#  [1. 3. 5. 5.]
#  [1. 3. 5. 7.]]


import numpy as np

A = np.array([[1, -2, 1], [0, 2, -1], [1, 1, -2]])
print(A)
# [[ 1 -2  1]
#  [ 0  2 -1]
#  [ 1  1 -2]]

# 求A的行列式，不为零则存在逆矩阵
A_det = np.linalg.det(A)
print(A_det)
# -2.9999999999999996

A_inverse = np.linalg.inv(A)  # 求A的逆矩阵
print(A_inverse)
# [[ 1.00000000e+00  1.00000000e+00 -1.11022302e-16]
#  [ 3.33333333e-01  1.00000000e+00 -3.33333333e-01]
#  [ 6.66666667e-01  1.00000000e+00 -6.66666667e-01]]

x = np.allclose(np.dot(A, A_inverse), np.eye(3))
print(x)  # True
x = np.allclose(np.dot(A_inverse, A), np.eye(3))
print(x)  # True

A_companion = A_inverse * A_det  # 求A的伴随矩阵
print(A_companion)
# [[-3.00000000e+00 -3.00000000e+00  3.33066907e-16]
#  [-1.00000000e+00 -3.00000000e+00  1.00000000e+00]
#  [-2.00000000e+00 -3.00000000e+00  2.00000000e+00]]


#  x + 2y +  z = 7
# 2x -  y + 3z = 7
# 3x +  y + 2z =18

import numpy as np

A = np.array([[1, 2, 1], [2, -1, 3], [3, 1, 2]])
b = np.array([7, 7, 18])
x = np.linalg.solve(A, b)
print(x)  # [ 7.  1. -2.]

x = np.linalg.inv(A).dot(b)
print(x)  # [ 7.  1. -2.]

y = np.allclose(np.dot(A, x), b)
print(y)  # True