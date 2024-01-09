#!/usr/bin/python
import math
import matplotlib.pyplot as plt
import numpy as np
import random

#####################################################
#			MAD - Exercise set 1					#
#			Least squares fitting 2D				#
#####################################################

# notation: 
# A ... (Nx2) matrix first column contains numbers 1, second column contains the current measurements Ij
# x ... (2x1) matrix containing the unknowns R and V0
# b ... (Nx1) matrix containing the voltages Vj
# formulation of the problem in the matrix form: A x = b
# least squares solution: x = (A_T A)^(-1) A_T b, where A_T denotes the transpose of matrix A and ^(-1) denotes the inverse

#####################
# 	question 1.a	# 
#####################
# define array of data points
current = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
voltage = [3.92,6.23,5.52,8.45,10.11,15.11,15.11,18.22,19.97,12.77]

#plot data
plt.figure(0)
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.plot(current,voltage,'bo')
plt.savefig("data_exp.pdf", bbox_inches='tight')

#####################
# 	question 1.b	# 
#####################
# define matrix A, b 
A = np.zeros((10,2))
b = np.zeros((10,1))

#fill matrix
for i in range(0,10):
	A[i,0] = 1.0
	A[i,1] = current[i]
	b[i,0] = voltage[i]

#Normalequation A_T*A*x = A*b

A_T = np.transpose(A)
product1 = np.dot(A_T,A)
product1_inv = np.linalg.inv(product1)
product2 = np.dot(A_T,b)

x = np.dot(product1_inv,product2)
V0 = x[0,0]
R = x[1,0]

print("LSQ gives: V0 = ",V0,"[V] and R = ",R," [ohm] for experimental data")

t = np.arange(0.0, 0.12, 0.02)
plt.figure(1)
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.plot(t, V0+R*t,'b--',current,b,'bo')
plt.savefig("lsq_exp.pdf", bbox_inches='tight')

#####################
# 	question 1.c	# 
#####################
# generate data without noise
# using V0 = -0.5 [V] and R = 150 [ohm]

#change b
for i in range(0,10):	
	b[i,0] = -0.5 + 150.0*current[i]

#recalculate LSQ
product2 = np.dot(A_T,b)
x = np.dot(product1_inv,product2)
V0 = x[0,0]
R = x[1,0]

print("LSQ gives: V0 = ",V0,"[V] and R = ",R," [ohm] for generated data with no noise and V0 = -0.5 [V] and R = 150 [ohm]")

plt.figure(2)
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.plot(t, V0+R*t,'b--',current,b,'bo')
plt.savefig("lsq_noNoise.pdf", bbox_inches='tight')


# generate data with uniform noise with different G
G = 1.5
#change b
for i in range(0,10):	
	noise = random.uniform(-1.0,1.0)
	b[i,0] = -0.5 + 150.0*current[i] + G*noise

#recalculate LSQ
product2 = np.dot(A_T,b)
x = np.dot(product1_inv,product2)
V0 = x[0,0]
R = x[1,0]

print("LSQ gives: V0 = ",V0,"[V] and R = ",R," [ohm] for generated data with noise ", G, "*[-1,1] and V0 = -0.5 [V] and R = 150 [ohm]")

plt.figure(3)
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.plot(t, V0+R*t,'r--',current,b,'ro')
plt.savefig("lsq_withNoise.pdf", bbox_inches='tight')

#####################
# 	question 1.d	# 
#####################
# generate data with outlier
# using V0 = -0.5 [V] and R = 150 [ohm]
#change b
for i in range(0,10):	
	b[i,0] = -0.5 + 150.0*current[i]

b[7,0] = 0.5*b[7,0]

#recalculate LSQ
product2 = np.dot(A_T,b)
x = np.dot(product1_inv,product2)
V0 = x[0,0]
R = x[1,0]

print("LSQ gives: V0 = ",V0,"[V] and R = ",R," [ohm] for generated data with outlier and V0 = -0.5 [V] and R = 150 [ohm]")

plt.figure(4)
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.plot(t, V0+R*t,'b--',current,b,'bo')
plt.savefig("lsq_outlier.pdf", bbox_inches='tight')

