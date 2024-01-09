#####################################################
#             MAD - Exercise Set 11                 #
#             Volume under surface                  #
#                                                   #
#                 Question 1b)                      #
#####################################################

import numpy as np

#function
def fun(x, y):
    return np.sin(x) + np.cos(y) + 1

#parameters
n = 30
x_min = -np.pi/2.
x_max = np.pi/2.
y_min = -np.pi/2.
y_max = np.pi/2.

#cartesian product
w_x = np.ones(shape=(1, n + 1)) * 2.
w_y = np.ones(shape=(n + 1, 1)) * 2.
w_x[0, 0] = w_y[0] = w_x[0, -1] = w_y[-1] = 1.
W = np.matmul(w_y, w_x)

#stepsize
h_x = (x_max - x_min)/n
h_y = (y_max - y_min)/n

#steps
lin_x = np.linspace(x_min, x_max, n + 1)
lin_y = np.linspace(y_min, y_max, n + 1)

#loop over both coordinates
val = 0
iter_y = 0
for y in lin_y:
    iter_x = 0
    for x in lin_x:
        val += W[iter_x, iter_y] * fun(x, y)
        iter_x += 1
    iter_y += 1

#apply weights of trapezoidal rule
V = val * h_x * 1/2. * h_y * 1/2.

#print
print("Volume_:%f" %V)
