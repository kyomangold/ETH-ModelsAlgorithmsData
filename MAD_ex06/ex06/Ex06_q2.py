import numpy as np
from math import sqrt, pi, exp
from scipy import linalg
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm

def interpolate_rbf(x, y, z, sigma_x, sigma_y, x_vis, y_vis):
    """ Interpolate the data set (x,y,z) with the data-driven RBF.
    Return the values of the interpolation function in the points (x_vis,y_viz).
    """

    N = np.size(x)

    A = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            A[i, j] = normpdf(x[i]-x[j], scale=sigma_x) * \
                        normpdf(y[i]-y[j], scale=sigma_y)

    # Solve LSE
    lu, piv = linalg.lu_factor(A)
    d = linalg.lu_solve((lu, piv), z)

    # Evaluate the interpolation function on the visualisation grid
    N_vis_x = np.size(x_vis)
    N_vis_y = np.size(y_vis)
    z_vis = np.zeros((N_vis_x, N_vis_y))
    for i in range(N_vis_x):
        for j in range(N_vis_y):
                t = 0
                # loop over all the data points
                for k in range(N):
                    t += d[k] * normpdf(
                            x_vis[i] - x[k], scale=sigma_x) * \
                            normpdf(y_vis[j] - y[k], scale=sigma_y)
                z_vis[i, j] = t
    return z_vis

def normpdf(x, scale=1 ):
    u = (x)/abs(scale)
    y = (1/(sqrt(2*pi)*abs(scale)))*exp(-u*u/2)
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Usage: q2.py <sigma_x> <sigma_y>')
    parser.add_argument('sigma_x', type = float)
    parser.add_argument('sigma_y', type = float)
    args = parser.parse_args()

    x = np.loadtxt("q2-data.txt", dtype=float, usecols=0)
    y = np.loadtxt("q2-data.txt", dtype=float, usecols=1)
    z = np.loadtxt("q2-data.txt", dtype=float, usecols=2)

    N_vis_x = 100
    N_vis_y = 100

    x_vis = np.linspace(x.min(), x.max(), num=N_vis_x)
    y_vis = np.linspace(y.min(), y.max(), num=N_vis_y)

    # Plot results
    z_vis = interpolate_rbf(x, y, z, args.sigma_x, args.sigma_y,
                            x_vis, y_vis)
    grid_x, grid_y = np.meshgrid(x_vis, y_vis,indexing='ij')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    z_min = z_vis.min()
    z_max = z_vis.max()
    color_map = cm.RdYlGn
    scalarMap = cm.ScalarMappable(
                    norm=Normalize(vmin=z_min, vmax=z_max),
                    cmap=color_map)
    C_colored = scalarMap.to_rgba(z_vis)

    ax.plot_surface(grid_x, grid_y, z_vis, facecolors=C_colored)

    ax.view_init(elev=80, azim=-80)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.axis('equal')

    ax.scatter(x, y, z)

    plt.show()
