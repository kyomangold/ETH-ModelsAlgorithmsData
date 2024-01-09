#####################################################
#             MAD - Exercise Set 11                 #
#            Monte-carlo integration                #
#                                                   #
#                 Question 2b)                      #
#####################################################

import numpy as np

# params
n = 100000
dims = 10
low = -1.
high = 1.

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
V_exact = exact_val(dims)

# function
def fun(r_vect):
    return 1. if np.linalg.norm(r_vect) <= 1. else 0.

# domain
dom = np.power((high - low), dims)

# random numbers
rand = np.random.uniform(low=low, high=high, size=(n, dims))

# check function
val = 0
val_sq = 0
for i in range(n):    
    val += fun(rand[i, :])
    val_sq += fun(rand[i, :])**2

# obtain integral & error approximation
exp = val * 1./n
exp_sq = val_sq * 1./n
V = dom * exp
err = dom * np.sqrt((exp_sq - exp**2)/n)
err_rel = err / V_exact
print("Volume_exact_:%f Volume_:%f Error_relative_:%f  Error_:%f" %(V_exact, V, err_rel, err))
