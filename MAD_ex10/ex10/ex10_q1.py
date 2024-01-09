#####################################################
#             MAD - Exercise Set 10                 #
#        Adaptive and Gauss Quadratures             #
#                                                   #
#                 Question 1                        #
#####################################################

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from helpers import *

x_s = -7
x_e = 7
n = 501
x = np.linspace(x_s,x_e,n)
y_u = batman_upper(x)
y_l = batman_lower(x)

init_intervals = np.array([x_s, x_e])
tol = 1e-6

def adaptive_simpson(intervals,cur_int,tol,fun,cur_depth,max_depth):
    s1 = simpson(cur_int, fun)
    c = (cur_int[0]+cur_int[1])/2.0
    s2 = simpson(np.array([cur_int[0], c]), fun)
    s3 = simpson(np.array([c, cur_int[1]]), fun)
    
    if abs(s1 - s2 - s3)/15.0 < tol or cur_depth >= max_depth:
        opt_integral = s1
        opt_intervals = intervals
        return opt_intervals, opt_integral
    else:
        cur_depth += 1

        new_intervals = np.concatenate((intervals[0:np.where(intervals==cur_int[0])[0][0]+1], 
            np.array([c]), intervals[np.where(intervals==cur_int[1])[0][0]:]))
        
        # First check first new interval
        ind_cur_int_1 = np.where(intervals==cur_int[0])[0][0]
        new_cur_int = new_intervals[ind_cur_int_1:(ind_cur_int_1+2)]
        intervals_1, integral_1 = adaptive_simpson(new_intervals, new_cur_int, tol, fun, cur_depth,
                max_depth)
        
        # Now check second interval
        ind_cur_int_2 = np.where(intervals_1==cur_int[1])[0][0]
        new_cur_int = intervals_1[(ind_cur_int_2-1):(ind_cur_int_2+1)]
        opt_intervals, integral_2 = adaptive_simpson(intervals_1, new_cur_int, tol, fun, cur_depth,
                max_depth)
        
        opt_integral = integral_1 + integral_2

        return opt_intervals, opt_integral

def simpson(interval, fun):
    a = interval[0]
    b = interval[1]
    s = (b-a)*(fun(a) + 4.0*fun((a+b)/2.0) + fun(b))/6.0
    return s

max_depth = 100

# Compute optimal intervals and integral value for both curves
opt_intervals_u, opt_integral_u = \
adaptive_simpson(init_intervals,init_intervals,tol,batman_upper,0,max_depth) 
opt_intervals_l, opt_integral_l = \
adaptive_simpson(init_intervals,init_intervals,tol,batman_lower,0,max_depth) 

# Find the interval widths
widths_u = opt_intervals_u[1:] - opt_intervals_u[0:-1]
min_u = np.min(widths_u)

print('{} Intervals Generated for Adaptive Simpsons Method'.format(len(widths_u)))
print('Smallest Interval Width: {}'.format(min_u))
print('Number of Intervals for uniform composite Simpsons Method: {}'.format(np.ceil((x_e-x_s)/min_u)))

print('Upper Area: ', opt_integral_u)
print('Lower Area: ', opt_integral_l)
A_tot = opt_integral_u + abs(opt_integral_l)
print('Total Area: ', A_tot)

print('\n')

A_true = (955./48.)-(2./7.)*(2*33**0.5 + 7*np.pi +
        3*10**0.5*(np.pi-1))+21*(np.arccos(3./7.)+np.arccos(4./7.))
print('True Area: ', A_true)
print('Percentage Error: {:f}%'.format(100*(A_true- A_tot)/A_true))

## Plotting!
plt.plot(x, y_u, 'k', linewidth=2)
plt.plot(x, y_l, 'k', linewidth=2)
plt.fill(x, y_u, color=(0.1,0.2,0.5,0.2))
plt.fill(x, y_l, color=(0.75,0.25,0.75,0.2))

for i in range(len(opt_intervals_u)):
    x_vec = np.array([opt_intervals_u[i], opt_intervals_u[i]])
    y_vec = np.array([0, batman_upper(opt_intervals_u[i])])
    plt.plot(x_vec, y_vec, 'k', linewidth=1)
    
for i in range(len(opt_intervals_l)):
    x_vec = np.array([opt_intervals_l[i], opt_intervals_l[i]])
    y_vec = np.array([0, batman_lower(opt_intervals_l[i])])
    plt.plot(x_vec, y_vec, 'k', linewidth=1)

plt.title("I'm Batman!!!",fontsize=24)
plt.axis('equal')
plt.savefig("batman.pdf", format="pdf", bbox_inches="tight")
plt.show()


