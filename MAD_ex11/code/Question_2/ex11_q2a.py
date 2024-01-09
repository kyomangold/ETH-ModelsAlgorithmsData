#####################################################
#             MAD - Exercise Set 11                 #
#            Monte-carlo integration                #
#                                                   #
#                 Question 2a)                      #
#####################################################

import numpy as np

#function
def fun(x, y, z):
    return 1. if np.linalg.norm([x, y, z]) <= 1. else 0.

#parameters
n = 30
x_min = -1.1
x_max = 1.1
y_min = -1.1
y_max = 1.1
z_min = -1.1
z_max = 1.1

# exact value
def exact_val(d):
    def recursion(d):
        if d == 0.5:
            return np.sqrt(np.pi)
        elif d == 1.:
            return 1
        else:
            return (d-1)*recursion(d-1)
    return np.pi**(d/2) / recursion(d/2 + 1)
V_exact = exact_val(3)

#cartesian product, replaced with a function here since we cannot do a simple matrix multiplication in 3d
def cart_eval(i):
   return 1. if (i is 0 or i is n - 1) else 2.

#stepsize
h_x = (x_max - x_min)/n
h_y = (y_max - y_min)/n
h_z = (z_max - z_min)/n

#steps
lin_x = np.linspace(x_min, x_max, n + 1)
lin_y = np.linspace(y_min, y_max, n + 1)
lin_z = np.linspace(z_min, z_max, n + 1)

#loop over all coordinates
nev = 0
val = 0
iter_z = 0
for z in lin_z:
    iter_y = 0
    for y in lin_y:
        iter_x = 0
        for x in lin_x:
            val += cart_eval(iter_x) * cart_eval(iter_y) * cart_eval(iter_z) * fun(x, y, z)
            nev += 1
            iter_x += 1
        iter_y += 1
    iter_z += 1

#apply weights of trapezoidal rule
V = val * h_x * 1/2. * h_y * 1/2. * h_z * 1/2.

#print
print("Volume_exact_:%f Volume_:%f Error_:%f nev_:%f" %(V_exact, V, V_exact - V, nev))
