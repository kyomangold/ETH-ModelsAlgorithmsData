#####################################################
#           MAD - Exercise Set 4                    #
#      Lagrange Interpolation and Splines           #
#                                                   #
#                 Question 3                        #
#####################################################

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# Helper function to evaluate cubic polynomial with appropriate parameters
def cubic(x, fi, fi1, xi, xi1, yi, yi1, D):
    f = fi*(xi1-x)**3/(6*D) + fi1*(x-xi)**3/(6*D) + ((yi1-yi)/D - (fi1-fi)*D/6)*(x-xi) + (yi - fi*D**2/6)
    print( "ak0=", (yi - fi*D**2/6),", ak1=",((yi1-yi)/D - (fi1-fi)*D/6),", ak2=",fi1/(6*D),", ak3=", fi/(6*D) )
    return f

# data points
x = np.array([0.0,1.0,2.0,3.0])
y = np.array([1.0,0.0,0.3,0.0])

D = 1.0 # this is the (constant) difference between x values

## Case A: Natural End Conditions ##
B2 = 2*D/3
C2 = D/6
A3 = D/6
B3 = 2*D/3

D2 = (y[2]-y[1])/D - (y[1]-y[0])/D # note index mismatch due to python 0 indexing convention
D3 = (y[3]-y[2])/D - (y[2]-y[1])/D

# This is the solution for the curvatures at the second and third points
F_A = np.linalg.inv(np.matrix([[B2, C2], [A3, B3]]))*np.matrix([[D2],[D3]])

x1 = np.linspace(x[0], x[1], 100)
x2 = np.linspace(x[1], x[2], 100)
x3 = np.linspace(x[2], x[3], 100)

# Now compute the function evaluated in the 3 intervals
print("Case A: Natural End Conditions")
print("Coefficients of the cubic polynomial Sk(x)")
print("segment 1")
f1A = cubic(x1, 0, F_A[0,0], x[0], x[1], y[0], y[1], D)
print("segment 2")
f2A = cubic(x2, F_A[0,0], F_A[1,0], x[1], x[2], y[1], y[2], D)
print("segment 3")
f3A = cubic(x3, F_A[1,0], 0, x[2], x[3], y[2], y[3], D)

#####################################################
## Case B: Right End Clamped (Left End Free) ##
A4 = D/6
B4 = D/3
C3 = D/6

D4 = -(y[3]-y[2])/D

F_B = np.linalg.inv(np.matrix([[B2, C2, 0], [A3, B3, C3], [0, A4, B4]]))*np.matrix([[D2],[D3],[D4]])

# Now compute the function evaluated in the 3 intervals
print("\nCase B: Right End Clamped (Left End Free)")
print("Coefficients of the cubic polynomial Sk(x)")
print("segment 1")
f1B = cubic(x1, 0, F_B[0,0], x[0], x[1], y[0], y[1], D)
print("segment 2")
f2B = cubic(x2, F_B[0,0], F_B[1,0], x[1], x[2], y[1], y[2], D)
print("segment 3")
f3B = cubic(x3, F_B[1,0], F_B[2,0], x[2], x[3], y[2], y[3], D)

#####################################################
## Case C: Both Ends Clamped ##
B1 = D/3
C1 = D/6
A2 = D/6

D1 = (y[1]-y[0])/D

F_C = np.linalg.inv(np.matrix([[B1, C1, 0, 0], [A2, B2, C2, 0], 
                               [0, A3, B3, C3], [0, 0, A4, B4]]))*np.matrix([[D1],[D2],[D3],[D4]])

# Now compute the function evaluated in the 3 intervals
print("\nCase C: Both Ends Clamped")
print("Coefficients of the cubic polynomial Sk(x)")
print("segment 1")
f1C = cubic(x1, F_C[0,0], F_C[1,0], x[0], x[1], y[0], y[1], D)
print("segment 2")
f2C = cubic(x2, F_C[1,0], F_C[2,0], x[1], x[2], y[1], y[2], D)
print("segment 3")
f3C = cubic(x3, F_C[2,0], F_C[3,0], x[2], x[3], y[2], y[3], D)

#####################################################
# Plotting!
plt.plot(x, y, 'b*', markersize=10)
plt.plot(np.concatenate((x1,x2,x3)), np.concatenate((f1A,f2A,f3A)), 'b')
plt.plot(np.concatenate((x1,x2,x3)), np.concatenate((f1B,f2B,f3B)), 'm')
plt.plot(np.concatenate((x1,x2,x3)), np.concatenate((f1C,f2C,f3C)), 'r')

plt.xlabel('X Axis', fontsize=11)
plt.ylabel('Y Axis', fontsize=11)
plt.title('Various Cubic Spline End Conditions', fontsize=12)
plt.legend(['Data Points', 'Natural End Conditions', 'Right Side Clamped', 'Both Sides Clamped'])
plt.xlim([-0.5, 3.5])
plt.ylim([-0.5, 1.5])
plt.grid()
plt.savefig("q3_sol.pdf", bbox_inches='tight')

