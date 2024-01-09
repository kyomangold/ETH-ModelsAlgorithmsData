import numpy as np
from math import sqrt, pi, exp, pow
from scipy import linalg

#####################################################
#			MAD - Exercise set 3					#   
#			Newton's method	- question 2.c			#
#####################################################

if __name__ == "__main__":

    #Newton's method:
    print("iter        r            error")

    #Values from 2.b
    K = np.array([87.7129,2.59695,-137.228])

    #initial guess
    x = 2.0

    #Parameters
    tol = 0.00001
    k_max = 20

    k = 1
    while( k <= k_max ):

        #Calculate Residuals
        F = -50000/(pi*pow(x,2)) + K[0]*exp(K[1]*x) + K[2]*x

        #Compute the Jacobian
        J = 100000/(pi*pow(x,3)) + K[0]*K[1]*exp(K[1]*x) + K[2]

        #Update of solution
        y = -1.0 * F / J
        x += y

        #Calculate error
        error = abs(y)

        #print current interation
        print("%d\t%f\t%f"% ((k, x, error)) )

        #Check if we have comvergence of the solution
        if( error < tol ): break

        k = k + 1

