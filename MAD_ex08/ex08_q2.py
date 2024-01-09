import argparse
import csv

#################################################################################
#			MAD - Exercise set 8					                            #   
#			Numerical integration	- question 2			                    #
#################################################################################

def nominator(T, r, theta):
    """List of nominator terms.

    :param T: Temperature list.
    :param r: Radius list.
    :param theta: Theta.
    :return: List of nominator terms.
    """
    f_nominator = []
    for (r_i, T_i) in zip(r, T):
        f_nominator.append(T_i * r_i * theta)
    return f_nominator

def denominator(r, theta):
    """List of denominator terms.

    :param r: Radius list.
    :param theta: Theta.
    :return: List of denominator terms.
    """
    f_denominator = []
    for r_i in r:
        f_denominator.append(r_i * theta)
    return f_denominator


def IntegrateTR(f, x):
    """Function that integrates over f with the trapezoidal rule.

    :param f: Array containing function values.
    :param x: Array containing x values.
    :return I: Integral value.
    """
    I = 0
    for i in range(len(f) - 1):
        I += (f[i] + f[i+1]) / 2. * (x[i+1] - x[i])
    return I

def IntegrateSM(f, x):
    """Function that integrates over f with the simpson rule.

    :param f: Array containing function values.
    :param x: Array containing x values.
    :return I: Integral value.
    """
    I = 0
    for i in range(0, len(f) - 1, step=2):
        I = I + (f[i] + 4*f[i+1] + f[i+2]) / 6. * (x[i+2] - x[i])
    return I

def main(args):

    # arrays to store the values
    T = []
    r = []

    # read data
    with open(args['data_path'], 'r') as file:
        data = csv.reader(file, delimiter=' ')
        for row in data:
            r.append(float(row[0]))
            T.append(float(row[1]))

    # get the denominator and nominator values
    f_nominator = nominator(T, r, args['theta'])
    f_denominator = denominator(r, args['theta'])

    # inegrate denominator and nominator functions
    I_nominator = IntegrateTR(f_nominator, r)
    I_denominator = IntegrateTR(f_denominator, r)

    # T_bar
    T_bar = I_nominator / I_denominator
    print("T_bar value: " + str(T_bar))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAD_ex8')

    #add arguments here
    parser.add_argument('--theta', help='thata value', default=0.7051, type=float)
    parser.add_argument('--data_path', help='path for data', default='./input_values.txt')

    args = vars(parser.parse_args())
    main(args)