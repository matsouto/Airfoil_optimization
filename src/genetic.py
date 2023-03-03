from bezier import bezier_airfoil
import aux
from random import uniform
import matplotlib.pyplot as plt

"""Baseado em https://github.com/ashokolarov/Genetic-airfoil, 
no paper Two-dimensional airfoil shape optimization for airfoils at low speeds - Ruxandra Mihaela Botez
e no vídeo https://www.youtube.com/watch?v=nhT56blfRpE"""


class genetic_algorithm:
    def __init__(self, initial_airfoil: bezier_airfoil, MAX_CHANGE: float):
        self.initial_cp = initial_airfoil.cp
        self.MAX_CHANGE = MAX_CHANGE

    def generate_genome(self):
        """Gera um perfil e atribui caracteristicas aleatórias"""
        genome = []
        for i in self.initial_cp:
            """Cria uma lista baseada no cp original, variando os parâmetros aleatoriamente"""
            genome.extend([[i[0] + uniform(-self.MAX_CHANGE, self.MAX_CHANGE),
                          i[1] + uniform(-self.MAX_CHANGE, self.MAX_CHANGE)]])

        return genome

    def generate_population(self, pop_size: int):
        return [self.generate_genome() for _ in range(pop_size)]

    def run_opt_genetic(cp):
        pass


def _example():
    """Cria um perfil a partir de um .dat e obtem os pontos de controle de Bezier"""
    initial_airfoil = bezier_airfoil()
    initial_airfoil.set_coords_from_dat("airfoils/s1223.dat")
    initial_airfoil.get_bezier_cp(18)

    gen = genetic_algorithm(initial_airfoil, MAX_CHANGE=0.1)

    genome = gen.generate_genome()
    X_bezier, Y_bezier = aux.generate_bezier_curve(genome)
    plt.plot(X_bezier, Y_bezier, 'b-', label='Bezier')
    plt.show()
    # gen.generate_population(2)


if __name__ == "__main__":
    _example()
