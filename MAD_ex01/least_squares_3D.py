#!/usr/bin/python
import math
import matplotlib.pyplot as plt
import numpy as np
import random

#####################################################
#			MAD - Exercise set 1					#
#			Least squares fitting 3D				#
#####################################################

# notation: 
# A ... (Nx3) matrix first column contains numbers 1, second column contains the x, third column contains the y
# x ... (3x1) matrix containing the unknowns alpha, beta, gamma
# b ... (Nx1) matrix containing the z
# formulation of the problem in the matrix form: A x = b
# least squares solution: x = (A_T A)^(-1) A_T b, where A_T denotes the transpose of matrix A and ^(-1) denotes the inverse

#####################
# 	question 2.a	# 
#####################
# define matrix A, b 
N = 100 #data points
G = 1.0 #noise level
A = np.zeros((N,3))
b = np.zeros((N,1))

#fill matrix
for i in range(0,10):
	for j in range(0,10):
		noise = random.uniform(-0.1,0.1)
		A[10*i+j,0] = 1.0
		A[10*i+j,1] = 0.1*i
		A[10*i+j,2] = 0.1*j
		b[10*i+j,0] = 0.1*i + 0.1*j + G*noise

A_T = np.transpose(A)
product1 = np.dot(A_T,A)
product1_inv = np.linalg.inv(product1)
product2 = np.dot(A_T,b)

solution = np.dot(product1_inv,product2)
alpha = solution[0,0]
beta  = solution[1,0]
gamma = solution[2,0]

print(("LSQ gives: alpha = ",alpha," and beta = ",beta," and gamma = ",gamma," for N = ",N ,"data points and noise level G = ",G))


#####################
# 	question 2.b	# 
#####################
# define matrix A, b 
N = 10 #data points
G = 1.0 #noise level
A = np.zeros((N,3))
b = np.zeros((N,1))

#fill matrix
for i in range(0,N):
	noise = random.uniform(-0.1,0.1)
	xrand = random.uniform(0.0,1.0)
	yrand = random.uniform(0.0,1.0)
	A[i,0] = 1.0
	A[i,1] = xrand
	A[i,2] = yrand
	b[i,0] = xrand + yrand + G*noise

A_T = np.transpose(A)
product1 = np.dot(A_T,A)
product1_inv = np.linalg.inv(product1)
product2 = np.dot(A_T,b)

solution = np.dot(product1_inv,product2)
alpha = solution[0,0]
beta  = solution[1,0]
gamma = solution[2,0]

print(("LSQ gives: alpha = ",alpha," and beta = ",beta," and gamma = ",gamma," for N = ",N ,"data points and noise level G = ",G ))




