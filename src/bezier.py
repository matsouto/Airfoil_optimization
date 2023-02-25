import bezier_aux as aux
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.signal import resample
from xfoil_runner.xfoil import run_xfoil, plot_polar

"""
Baseado em https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
e provavelmente na tese de Tim Andrew Pastva, "Bézier Curve Fitting", 1998.

Outros materiais:
Geometric Modeling - Mortenson
Innovative Design and Development Practices in Aerospace and Automotive Engineering - Pg.  79

"""


class bezier_airfoil:

    all = []

    def __init__(self, airfoil_path: str):
        """ Converte o .dat para coordenadas np.array X e Y """
        self.airfoil_path = airfoil_path
        df = read_csv(airfoil_path, names=("X", "Y"), sep='\s+')
        self.original_name = df.iloc[0]["X"]
        self.X = df["X"].drop(0).to_numpy(float)
        self.Y = df["Y"].drop(0).to_numpy(float)

        self.sim = False  # Informa se o perfil já foi simulado
        self.alpha = None
        self.Cl = None
        self.Cd = None
        self.ClCd = None
        self.Cl3Cd2 = None
        self.stall_angle = None
        self.alpha_range = None

    def set_X(self, xvalue):
        self.X = xvalue

    def set_Y(self, yvalue):
        self.Y = yvalue

    """Calcula os parâmetros de bezier"""

    def get_bezier_parameters(self, degree=3):

        self.degree = degree

        if self.degree < 1:
            raise ValueError('degree must be 1 or greater.')

        if len(self.X) != len(self.Y):
            raise ValueError('X and Y must be of the same length.')

        if len(self.X) < self.degree + 1:
            raise ValueError(f'There must be at least {self.degree + 1} points to '
                             f'determine the parameters of a degree {self.degree} curve. '
                             f'Got only {len(self.X)} points.')

        T = np.linspace(0, 1, len(self.X))
        M = aux.bmatrix(T, self.degree)
        points = np.array(list(zip(self.X, self.Y)))

        parameters = aux.least_square_fit(points, M).tolist()
        parameters[0] = [self.X[0], self.Y[0]]
        parameters[len(parameters)-1] = [self.X[len(self.X)-1],
                                         self.Y[len(self.Y)-1]]

        self.parameters = parameters
        return self.parameters

    """Roda simulação pelo XFOIL"""

    def simulate(self, alpha_i=0, alpha_f=10, alpha_step=0.25, Re=1000000, n_iter=100):
        run_xfoil(self.airfoil_path, self.original_name,
                  alpha_i, alpha_f, alpha_step, Re, n_iter)

    def get_opt_params(self, polar_path="src/xfoil_runner/data/polar_file.txt"):

        if not self.sim:
            raise ValueError("O perfil precisa ser simulado antes")

        with open(polar_path) as file:
            """ Ref: https://github.com/ashokolarov/Genetic-airfoil"""
            polar_data = np.array(
                [np.array([float(x) for x in line.split()]) for line in file.readlines()[12:]])
            alpha = polar_data[:, 0]
            Cl = polar_data[:, 1]
            Cd = polar_data[:, 2]

            idx = np.argmax(Cl*Cl*Cl / Cd / Cd)
            ClCd = Cl[idx] / Cd[idx]
            Cl3Cd2 = (Cl[idx])**3 / (Cd[idx])**2

            stall_idx = np.argmax(Cl)
            alpha_range = alpha[stall_idx] - alpha[idx]

            self.alpha = alpha
            self.Cl = Cl
            self.Cd = Cd
            self.ClCd = ClCd
            self.Cl3Cd2 = Cl3Cd2
            self.stall_angle = stall_idx
            self.alpha_range = alpha_range

        return ClCd, Cl3Cd2, alpha_range

    def __str__(self):
        return self.original_name


def _example():
    airfoil = bezier_airfoil("airfoils/s1223.dat")
    # airfoil.set_X(np.linspace(0, 15))
    # airfoil.set_Y(np.cos(np.linspace(0, 15)))

    plt.plot(airfoil.X, airfoil.Y, "r", label='Original Points')

    params = airfoil.get_bezier_parameters(20)  # Args: Grau do polinômio
    # params[3] = [0.5, 0.23]
    # print(params)

    """Plota pontos de controle"""
    x_params_list = [param[0] for param in params]
    y_params_list = [param[1] for param in params]
    x_params = np.array(x_params_list)
    y_params = np.array(y_params_list)

    # plt.plot(x_params, y_params, 'k--o', label='Control Points')

    X_bezier, Y_bezier = aux.generate_bezier_curve(
        params, nTimes=len(airfoil.X))

    # Plota a curva de bezier
    plt.plot(X_bezier, Y_bezier, 'b-', label='Bezier')

    Y_error = np.abs(Y_bezier - resample(airfoil.Y, len(Y_bezier)))
    print(f'Erro máximo: {max(Y_error)}')
    # plt.plot(X_bezier, Y_error, 'g--', label="Erro")

    plt.legend()
    plt.show()

    # airfoil.simulate()
    # plot_polar()


    # Se esse arquivo for executado, rode _example()
if __name__ == "__main__":
    _example()
