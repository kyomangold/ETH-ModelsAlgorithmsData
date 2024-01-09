#!/usr/bin/python
import math
import matplotlib.pyplot as plt
import numpy as np

#####################################################
#			MAD - Exercise set 2					#
#			Normal Equation vs. SVD	                #
#####################################################

# notation: 
# A ... (Nx2) matrix first column contains numbers 1, second column contains the current measurements Ij
# x ... (2x1) matrix containing the unknowns R and V0
# b ... (Nx1) matrix containing the voltages Vj
# formulation of the problem in the matrix form: A x = b
# normal equation: x = (A_T A)^(-1) A_T b, where A_T denotes the transpose of matrix A and ^(-1) denotes the inverse
# solution using SVD: x=A_+b = VE^+U_T b, where U,V are the orthogonal matrices obtained from the singular value decomposition (A=UEV_T) and A_+ the pseudo-inverse

#################
# 	question 1	#
#################
# define array for measurement-times
t = np.linspace(0,1,50)
# define the Vandermonde matrix
A = np.vander(t, N=15, increasing=True)
# define array for coefficients
x = np.linspace(1,15,15)
print("Exact solution to the LLS problem: ", x)
# compute the perfect measurements
b = np.dot(A,x)

###################
#  question 1.a)  #
###################

condA = np.linalg.cond(A)
print("The condition number of A is {:.2E}".format(condA))

#####################
# 	question 1.b)	#
#####################

# compute solution according to normal equation
A_T = np.transpose(A)
product1 = np.dot(A_T,A)
product1_inv = np.linalg.inv(product1)
product2 = np.dot(A_T,b)

xBar = np.dot(product1_inv,product2)
print("xBar for normal equation: ", xBar)

#compute relative residual
residual = np.linalg.norm(b-np.dot(A,xBar))
normA = np.linalg.norm(A,2)
normb = np.linalg.norm(b,2)
rel_residual = residual / (normA*normb)

#compute error
error = np.linalg.norm(xBar-x,2)

print("The normalized residual when computing the LLS with the normal equation gives {:.2E}, whereas the absolute error is {:.2E}".format(rel_residual, error))


#####################
# 	question 1.c)	#
#####################

#compute solution using pseudo-inverse
Aplus = np.linalg.pinv(A)
xBar = np.dot(Aplus,b)
print("xBar for SVD: ", xBar)

#compute relative residual
residual = np.linalg.norm(b-np.dot(A,xBar))
normA = np.linalg.norm(A,2)
normb = np.linalg.norm(b,2)
rel_residual = residual / (normA*normb)

#compute error
error = np.linalg.norm(xBar-x,2)

print("The normalized residual when computing the LLS with the SVD gives {:.2E}, whereas the absolute error is {:.2E}".format(rel_residual, error))
