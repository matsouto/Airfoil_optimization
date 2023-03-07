from bezier import bezier_airfoil
import aux
from random import uniform
import numpy as np
import matplotlib.pyplot as plt

"""Baseado em https://github.com/ashokolarov/Genetic-airfoil,
no paper Two-dimensional airfoil shape optimization for airfoils at low speeds - Ruxandra Mihaela Botez
e no vídeo https://www.youtube.com/watch?v=nhT56blfRpE"""


class genetic_algorithm:
    def __init__(self, initial_airfoil: bezier_airfoil, MAX_CHANGE: float):
        self.initial_cp_upper = initial_airfoil.cp_upper
        self.initial_cp_lower = initial_airfoil.cp_lower
        self.degree_upper = initial_airfoil.degree_upper
        self.degree_lower = initial_airfoil.degree_lower
        self.MAX_CHANGE = MAX_CHANGE

    def generate_genome(self):
        """Gera um perfil e atribui caracteristicas aleatórias"""
        genome_upper = []
        genome_lower = []

        """
        Mantém alguns pontos no bordo de fuga e de ataque
        inalterados com base na variável 'skip_cp'.
        Isso ajuda a diminuir a ocorrência de perfis impossíveis,
        e agiliza o processo de otimização.
        """

        skip_cp_upper = 2  # Quantos pontos de controle vão ser inalterados
        skip_cp_lower = 4

        if (skip_cp_upper > self.degree_upper//2) or (skip_cp_lower > self.degree_lower//2):
            raise ValueError(
                '"skip_cp" precisa ser menor que a metade do grau de bezier')

        for i in range(skip_cp_upper):
            genome_upper.extend(
                [self.initial_cp_upper[i]])

        for i in range(skip_cp_lower):
            genome_lower.extend(
                [self.initial_cp_lower[i]])

        """Realiza a variação dos parâmetros aleatoriamente"""

        for i in self.initial_cp_upper[skip_cp_upper:self.degree_upper - skip_cp_upper + 1]:
            genome_upper.extend([[i[0] + uniform(-self.MAX_CHANGE, self.MAX_CHANGE),
                                  i[1] + uniform(-self.MAX_CHANGE, self.MAX_CHANGE)]])

        for i in self.initial_cp_lower[skip_cp_lower:self.degree_lower - skip_cp_lower + 1]:
            genome_lower.extend([[i[0] + uniform(-self.MAX_CHANGE, self.MAX_CHANGE),
                                  i[1] + uniform(-self.MAX_CHANGE, self.MAX_CHANGE)]])

        for i in range(skip_cp_upper):
            genome_upper.extend(
                [self.initial_cp_upper[self.degree_upper - skip_cp_upper + 1 + i]])

        for i in range(skip_cp_lower):
            genome_lower.extend(
                [self.initial_cp_lower[self.degree_lower - skip_cp_lower + 1 + i]])

        return genome_upper, genome_lower

    def generate_population(self, pop_size: int):
        population = []
        for _ in range(pop_size):
            population.append(self.generate_genome())
        return population

    def plot_population(self, population: list):
        """Define o layout a partir da raiz quadrada do tamanho da população"""
        total_subplots = len(population)
        cols_subplots = int(total_subplots**0.5)
        rows_subplots = total_subplots//cols_subplots

        if total_subplots % cols_subplots != 0:
            rows_subplots += 1

        position = range(1, total_subplots + 1)
        fig = plt.figure(1, figsize=(14, 5))
        plt.title(f"População de {len(population)} genomas")
        plt.axis('off')

        """Gera o perfil original para demonstração"""
        X_bezier_upper, Y_bezier_upper = aux.generate_bezier_curve(
            self.initial_cp_upper)

        X_bezier_lower, Y_bezier_lower = aux.generate_bezier_curve(
            self.initial_cp_lower)

        for k in range(total_subplots):

            ax = fig.add_subplot(rows_subplots, cols_subplots, position[k])
            ax.yaxis.set_visible(False)
            X_upper, Y_upper = aux.generate_bezier_curve(
                population[k][0])
            X_lower, Y_lower = aux.generate_bezier_curve(
                population[k][1])
            ax.plot(X_upper, Y_upper, 'b--')
            ax.plot(X_lower, Y_lower, 'b--')

            # """Plota o perfil original"""
            ax.plot(X_bezier_upper, Y_bezier_upper, 'g')
            ax.plot(X_bezier_lower, Y_bezier_lower, 'g')
        plt.show()

    def run_opt_genetic(cp):
        pass


def _example():
    """Cria um perfil a partir de um .dat e obtem os pontos de controle de Bezier"""
    initial_airfoil = bezier_airfoil()
    initial_airfoil.set_coords_from_dat("airfoils/s1223.dat")
    initial_airfoil.get_bezier_cp(8, 16)

    gen = genetic_algorithm(initial_airfoil, MAX_CHANGE=0.04)
    # plt.figure(figsize=(9, 3))

    population = gen.generate_population(20)
    gen.plot_population(population)


if __name__ == "__main__":
    _example()
