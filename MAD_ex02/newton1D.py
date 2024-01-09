#!/usr/bin/python
import math
import matplotlib.pyplot as plt
import numpy as np

#####################################################
#			MAD - Exercise set 2					#
#			Newton's Method      	                #
#####################################################

# notation: 
# func ... function for which we want to find root
# Dfunc ... derivative of the function for which we want to find root
# x ... np.array containing the values from the Newton iterations
# xAnalytic .. np.array containing the values computed in exercise 2.b)
# Newton update x[k+1] = x[k]-func(x[k])/Dfunc(x[k])

#####################
# 	question 2.c)	#
#####################

# define values from exercise 2.b)
xAnalytic = np.array([6,35/6, 2449/420])

# define functions
def func(x):
    return x**2-34

def Dfunc(x):
    return 2*x

# intialize np.array with 10 zeros (assuming convergence in less than 10 iterations)
x = np.zeros(10)
# set intial values
k = 0
error = 9999
x[0] = 6
# perform Newton iterations until convergence
while( error >= 1e-15):
    k += 1
    x[k]=x[k-1]-func(x[k-1])/Dfunc(x[k-1])
    error = np.linalg.norm(x[k]-x[k-1])
    print("iteration {:d}: x={:f}, error={:.15f}".format(k,x[k],error))

plt.figure(1)
plt.xlabel('iteration k')
plt.ylabel('value of x')
plt.plot(np.arange(k),x[:k], 'bo', )
plt.savefig("newton_converge.pdf", bbox_inches='tight')
