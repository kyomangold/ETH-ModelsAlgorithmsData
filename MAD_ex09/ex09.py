import sys
import numpy as np
import math
import matplotlib.pyplot as plt

#################################################################################
#			MAD - Exercise set 9                                                #
#			Romberg integration 	- question 3                                #
#################################################################################

def printf(format, *args):
  sys.stdout.write(format % args)

def ErfFunct(x):
    PI=3.141592653589
    return (2/math.sqrt(PI))*math.exp(-x*x);

def PlanarMixingLayer(Uexh,Ufree,r_R):
    return Uexh+(Ufree-Uexh)*Romberg(0,2*r_R);


def Romberg(a,b):
    K=4
    PI=3.141592653589
    Integr=[0, 0, 0, 0, 0]
    fa=ErfFunct(a)
    fb=ErfFunct(b)
    s=0
    n = 1
    for l in range(0,K+1):
       h=(b-a)/n
       for i in range(1,n,2):
           s=s+ErfFunct(a+i*h)
       Integr[l]=(h/2)*(fa+fb+2*s)
       n=n*2

    for ka in range(K,-1,-1):
        k4=math.pow(4,ka)
        for l in range(0,ka):
          Integr[l]=(k4*Integr[l+1]-Integr[l])/(k4-1);

    return Integr[0];
 


# Call the Romberg integration to compute I^16_0
RombergValue=Romberg(0,0.5);
printf("The Romberg computation is %.12f", RombergValue)

# Compute the planar mixing layer value for different r/R 
r_R=np.linspace(0,1,11)
U_r=[]
for i in range(0,len(r_R)):
    U_r.append(PlanarMixingLayer(1200,800,r_R[i]));

plt.plot(r_R,U_r)
plt.xlabel('r/R')
plt.ylabel('Velocity, km/h')
plt.show()
   # print(r_R[i],U_r[i])


