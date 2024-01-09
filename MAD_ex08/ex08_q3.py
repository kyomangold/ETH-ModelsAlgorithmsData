import argparse
import numpy as np

#################################################################################
#			MAD - Exercise set 8					                            #   
#			Numerical integration	- question 3			                    #
#################################################################################

def T_function(r):
    """Function relating radius to Temperature.

    :param r: Radius.
    :return: Function value.
    """
    a, b, c = -0.125, -3.455, 770.3
    return a * np.power(r, b) + c


def nominator(r):
    """Nominator term.

    :param r: Radius.
    :return:  Nominator term.
    """
    return T_function(r) * r * 0.7051

def denominator(r):
    """Denominator term.

    :param r: Radius.
    :return: Denominator term.
    """
    return r * 0.7051

def IntegrateTR(f,x):
    """Function that integrates over f with the trapezoidal rule.

    :param f: Array containing function values.
    :param x: Array containing x values.
    :return I: Integral value.
    """
    I = 0
    for i in range(len(x) - 1):
        I += (f(x[i]) + f(x[i+1])) / 2. * (x[i+1] - x[i])
    return I

def IntegrateSM(f, x):
    """Function that integrates over f with the simpson rule.

    :param f: Array containing function values.
    :param x: Array containing x values.
    :return I: Integral value.
    """
    I = 0
    for i in range(len(x) - 1):
        x_inter = (x[i+1] + x[i]) / 2.
        I += (f(x[i]) + 4. * f(x_inter) + f(x[i+1])) / 6. * (x[i+1] - x[i])
    return I

def main(args):

    # arrays to store the values
    r = np.linspace(0.095, 0.145, num=6)

    # integrate denominator and nominator with Trapezoidal rule
    TR_I_nom = IntegrateTR(nominator, r)
    TR_I_den = IntegrateTR(denominator, r)

    # integrate denominator and nominator functions
    SM_I_nom = IntegrateSM(nominator, r)
    SM_I_den = IntegrateSM(denominator, r)

    # T_bar
    TR_T_bar = TR_I_nom / TR_I_den
    print("TR_T_bar value: " + str(TR_T_bar))
    SM_T_bar = SM_I_nom / SM_I_den
    print("SM_T_bar value: " + str(SM_T_bar))

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser(description='MAD_ex8')
    args = vars(parser.parse_args())
    main(args)