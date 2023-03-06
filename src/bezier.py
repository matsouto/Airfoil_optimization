import aux
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

    def __init__(self):
        """Para debug"""
        self.sim = False  # Informa se o perfil já foi simulado

    def set_coords_from_dat(self, airfoil_path: str):
        """ Converte o .dat para coordenadas np.array X e Y """
        self.airfoil_path = airfoil_path
        df = read_csv(airfoil_path, names=("X", "Y"), sep='\s+')
        self.original_name = df.iloc[0]["X"]
        self.X = df["X"].drop(0).to_numpy(float)
        self.Y = df["Y"].drop(0).to_numpy(float)
        self.X_upper = self.X[:int(len(self.X)/2)]
        self.X_lower = self.X[int(len(self.X)/2):]
        self.Y_upper = self.Y[:int(len(self.Y)/2)]
        self.Y_lower = self.Y[int(len(self.Y)/2):]

    def set_X_upper(self, xvalue):
        self.X_upper = xvalue

    def set_X_lower(self, xvalue):
        self.X_lower = xvalue

    def set_Y_upper(self, yvalue):
        self.Y_upper = yvalue

    def set_Y_lower(self, yvalue):
        self.Y_lower = yvalue

    def get_bezier_cp(self, degree_upper: int, degree_lower: int):
        """
        Calcula os parâmetros de bezier.

        Recebe graus diferentes para o intra e extradorso já que 
        para perfis arquiados, o intradorso necessita de mais pontos que o
        extradorso, assim, pode-se diminuir a quantidade de pontos totais 
        e melhorar a velocidade de convergência do algoritmo otimizador.
        """

        self.degree_upper = degree_upper
        self.degree_lower = degree_lower

        if (self.degree_upper or self.degree_lower) < 1:
            raise ValueError('Grau precisa ser 1 ou maior.')

        if len(self.X) != len(self.Y):
            raise ValueError('X e Y precisam ter o mesmo tamanho.')

        if len(self.X) < (self.degree_lower + 1 or self.degree_upper+1):
            raise ValueError(f'É necessário ter pelo menos {self.degree + 1} pontos para '
                             f'determinar os parâmetros de uma curva de grau {self.degree}. '
                             f'Foram dados apenas {len(self.X)} pontos.')

        T = np.linspace(0, 1, len(self.X_upper))
        M_upper = aux.bmatrix(T, self.degree_upper)
        points_upper = np.array(list(zip(self.X_upper, self.Y_upper)))
        points_lower = np.array(list(zip(self.X_lower, self.Y_lower)))

        cp_upper = aux.least_square_fit(points_upper, M_upper).tolist()
        cp_upper[0] = [self.X_upper[0], self.Y_upper[0]]
        cp_upper[len(cp_upper)-1] = [self.X_upper[len(self.X_upper)-1],
                                     self.Y_upper[len(self.Y_upper)-1]]

        M_lower = aux.bmatrix(T, self.degree_lower)
        cp_lower = aux.least_square_fit(points_lower, M_lower).tolist()
        cp_lower[0] = [self.X_lower[0], self.Y_lower[0]]
        cp_lower[len(cp_lower)-1] = [self.X_lower[len(self.X_lower)-1],
                                     self.Y_lower[len(self.Y_lower)-1]]

        self.cp_upper = cp_upper
        self.cp_lower = cp_lower
        return self.cp_upper, self.cp_lower

    def simulate(self, alpha_i=0, alpha_f=10, alpha_step=0.25, Re=1000000, n_iter=100):
        """Roda simulação pelo XFOIL"""
        run_xfoil(self.airfoil_path, self.original_name,
                  alpha_i, alpha_f, alpha_step, Re, n_iter)
        self.sim = True

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

    def save_as_dat_from_bezier(self, name="generated_airfoil"):
        """ Salva o perfil de bezier como um arquivo .dat"""

        self.X_bezier_upper, self.Y_bezier_upper = aux.generate_bezier_curve(
            self.cp_upper, nTimes=len(self.X_upper))
        self.X_bezier_lower, self.Y_bezier_lower = aux.generate_bezier_curve(
            self.cp_lower, nTimes=len(self.X_lower))

        data_upper = np.array([np.around(self.X_bezier_upper, 6).astype(
            'str'), np.around(self.Y_bezier_upper, 6).astype('str')]).transpose()
        data_lower = np.array([np.around(self.X_bezier_lower, 6).astype(
            'str'), np.around(self.Y_bezier_lower, 6).astype('str')]).transpose()

        if '.dat' not in name:
            name += '.dat'

        data = np.concatenate((data_upper, data_lower))

        header = "Airfoil"  # Melhorar isso aqui
        np.savetxt(f'airfoils/{name}', data,
                   header=header, comments="", fmt="%s")

    def __str__(self):
        return self.original_name


def _example():
    airfoil = bezier_airfoil()
    airfoil.set_coords_from_dat("airfoils/s1223.dat")
    # airfoil.set_X(np.linspace(0, 15))
    # airfoil.set_Y(np.cos(np.linspace(0, 15)))

    plt.figure(figsize=(9, 3))

    plt.plot(airfoil.X_upper, airfoil.Y_upper,
             "r", label='Original Points - Upper')
    plt.plot(airfoil.X_lower, airfoil.Y_lower,
             "b", label='Original Points - Lower')

    cp_upper, cp_lower = airfoil.get_bezier_cp(
        8, 16)  # Args: Grau do polinômio
    # cp_lower[7] = [cp_lower[7][0]+0.1, cp_lower[7][1]]
    print(cp_lower)

    """Gera listas com os pontos de controle"""
    x_cp_list_upper = [i[0] for i in cp_upper]
    y_cp_list_upper = [i[1] for i in cp_upper]
    x_cp_list_lower = [i[0] for i in cp_lower]
    y_cp_list_lower = [i[1] for i in cp_lower]

    """Converte a lista para array"""
    x_cp_upper = np.array(x_cp_list_upper)
    y_cp_upper = np.array(y_cp_list_upper)
    x_cp_lower = np.array(x_cp_list_lower)
    y_cp_lower = np.array(y_cp_list_lower)

    """Plota pontos de controle"""
    # plt.plot(x_cp_upper, y_cp_upper, 'k--o', label='Control Points')
    # plt.plot(x_cp_lower, y_cp_lower, 'k--o')

    """Plota a curva de bezier"""
    X_bezier_upper, Y_bezier_upper = aux.generate_bezier_curve(
        cp_upper, nTimes=len(airfoil.X_upper))
    # plt.plot(X_bezier_upper, Y_bezier_upper, 'g--', label='Bezier')

    X_bezier_lower, Y_bezier_lower = aux.generate_bezier_curve(
        cp_lower, nTimes=len(airfoil.X_lower))
    plt.plot(X_bezier_lower, Y_bezier_lower, 'g--', label='Bezier')

    X_bezier = np.concatenate((X_bezier_upper, X_bezier_lower))
    Y_bezier = np.concatenate((Y_bezier_upper, Y_bezier_lower))
    # plt.plot(X_bezier, Y_bezier, 'g--', label='Bezier')

    plt.legend()
    plt.xlabel("x/c")
    # plt.ylabel("y")

    """Calcula o erro - PRECISA SER MELHORADO (Ver o artigo)"""
    Y_error = np.abs(Y_bezier_lower -
                     resample(airfoil.Y_lower, len(Y_bezier_lower)))
    print(f'Erro máximo (Curva inferior): {max(Y_error)}')
    # plt.figure()
    # plt.plot(X_bezier_lower, Y_error, 'g--', label="Erro")
    # plt.title("Erro em Y (Curva inferior)")
    # plt.xlabel("x/c")

    plt.show()

    airfoil.save_as_dat_from_bezier()

    # airfoil.simulate()
    # plot_polar()


    # Se esse arquivo for executado, rode _example()
if __name__ == "__main__":
    _example()
