import numpy as np
from math import sqrt, pi, exp
from scipy import linalg

#####################################################
#			MAD - Exercise set 3					#   
#			Newton's method	- question 2.b			#
#####################################################

def solve_system(J, F):
    y = np.linalg.solve(J, -1.0*F)
    return y

def calculate_Jacobian(x):
    J = np.zeros((3, 3))
    for k in range(1,4,1):
        J[k-1,0] = exp(k/10.0*x[1])
        J[k-1,1] = k/10.0*x[0]*exp(k/10.0*x[1])
        J[k-1,2] = k/10.0
    return J

def calculate_Residuals(x):
    F = np.zeros(3)
    F[0] = x[0]*exp(0.1*x[1]) + 0.1*x[2] - 100.0
    F[1] = x[0]*exp(0.2*x[1]) + 0.2*x[2] - 120.0
    F[2] = x[0]*exp(0.3*x[1]) + 0.3*x[2] - 150.0
    return F

#################################################################################
if __name__ == "__main__":


    #Newton's method:
    print("iter        k_1            k_2            k_3            error")

    #Initial guess
    x = np.array([80,10,1])

    #Parameters
    tol = 0.00001
    k_max = 20

    k = 1
    while( k <= k_max ):

        #Calculate Residuals
        F = calculate_Residuals(x)

        #Compute the Jacobian
        J = calculate_Jacobian(x)

        #Solve the linear system
        y = solve_system(J, F)

        #Update of solution
        x = x + y

        #Calculate error
        error = sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2])

        #print current interation
        print("%d\t%f\t%f\t%f\t%f"% ((k, x[0], x[1], x[2], error)) )

        #Check if we have comvergence of the solution
        if( error < tol ): break

        k = k + 1

