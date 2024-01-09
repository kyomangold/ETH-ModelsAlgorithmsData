#####################################################
#           MAD - Exercise Set 4                    #
#      Lagrange Interpolation and Splines           #
#                                                   #
#                 Question 1                        #
#####################################################

import math
import matplotlib.pyplot as plt
import numpy as np
import sys

def fun(x):
    return 1 / (1 + x**2)**0.5

# compute the function evaluation of the j'th basis poynomial
# using the training data (x0,x1,...,xk) on the evaluation 
# data (whatever length you decide to put in)
def compute_basis_polynomial(x_train, x_eval, j):
    xj = x_train[j] # extract j'th element
    x_train = np.delete(x_train, j) # remove j'th element
    lj = np.ones(np.size(x_eval)) # prepare vector for output
    for i in range(len(x_train)):
        lj *= (x_eval - x_train[i])/(xj - x_train[i])
    return lj

# compute the full lagrange interpolation using 
# the data (x_train, y_train) and evaluated at (x_eval)
def compute_full_lagrange_function(x_train, y_train, x_eval):
    L = np.zeros(np.size(x_eval))
    for i in range(len(x_train)):
        L += y_train[i]*compute_basis_polynomial(x_train, x_eval, i)
    return L

# Question 1:
x_low = -5
x_high = 5
n = 11

x_train = np.linspace(x_low,x_high,n)
y_train = fun(x_train)

x_eval = np.linspace(x_low,x_high,200)
y_true = fun(x_eval)

# full lagrange interpolation
y_lagrange = compute_full_lagrange_function(x_train, y_train, x_eval)

plt.plot(x_train, y_train, 'b*')
plt.plot(x_eval, y_true, 'm-')
plt.plot(x_eval, y_lagrange, 'r-')
plt.grid()
plt.xlabel('X Axis', fontsize=11)
plt.ylabel('Y Axis', fontsize=11)
plt.title('Lagrange Interpolation', fontsize=12);
plt.legend(['Sample Points', 'True Function', 'Lagrange Interpolation'])
plt.savefig("q1_sol.pdf", bbox_inches='tight')

