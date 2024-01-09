#####################################################
#             MAD - Exercise Set 10                 #
#        Adaptive and Gauss Quadratures             #
#                                                   #
#               Helper Functions                    #
#####################################################

import math
import numpy as np
import sys

def batman_upper(x):
    f = (h(x)-l(x))*np.heaviside(x+1.0, 0.5) + \
    (r(x)-h(x))*np.heaviside(x-1.0, 0.5) + \
    (l(x)-w(x))*np.heaviside(x+3.0, 0.5) + \
    (w(x)-r(x))*np.heaviside(x-3.0, 0.5) + \
    w(x)
    
    return np.real(f)

def batman_lower(x):
    f = 0.5*(np.abs(0.5*x) + sqrt(1.0 - (np.abs(np.abs(x)-2.0)-1.0)**2) - \
            (1.0/112.0)*(3.0*sqrt(33.0)-7.0)*x**2 \
            + 3.0*sqrt(1.0-(x/7.0)**2)-3.0) * \
            (np.sign(x+4.0)-np.sign(x-4.0))-3.0*sqrt(1.0-(x/7.0)**2)
    return np.real(f)

def w(x):
    f = 3.0*sqrt(1.0-(x/7.0)**2);
    return f

def l(x):
    f = 0.5*(x+3.0)-(3.0/7.0)*sqrt(10.0)*sqrt(4.0-(x+1.0)**2) + (6.0/7.0)*sqrt(10.0);
    return f
    
def h(x):
    f = 0.5*(3.0*(np.abs(x-0.5)+np.abs(x+0.5)+6.0)-11.0*(np.abs(x+0.75)+np.abs(x-0.75)));
    return f

def r(x):
    f = 0.5*(3.0-x) - (3.0/7.0)*sqrt(10.0)*sqrt(4.0-(x-1.0)**2) + (6.0/7.0)*sqrt(10.0);
    return f

def sqrt(x):
    return np.lib.scimath.sqrt(x)


