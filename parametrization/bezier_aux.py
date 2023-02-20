from scipy.special import comb
import numpy as np


# def bpoly(n, t, i):
#     """ Polinômios de Bernstein  """
#     return comb(n, i) * (t**i) * (1 - t)**(n-i)


def bmatrix(T, degree):
    """ Bernstein matrix for Bézier curves. """
    return np.matrix([[bernstein_poly(i, degree, t) for i in range(degree + 1)] for t in T])


def least_square_fit(points, M):
    M_ = np.linalg.pinv(M)
    return M_ * points


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t**i) * (1 - t)**(n-i)


def generate_bezier_curve(points, nTimes=50):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals
