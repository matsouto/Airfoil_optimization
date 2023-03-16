import aux
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import copy
from xfoil_runner.xfoil import run_xfoil, plot_polar
from bezier import bezier_airfoil

"""Baseado em https://github.com/ashokolarov/Genetic-airfoil,
no paper Two-dimensional airfoil shape optimization for airfoils at low speeds - Ruxandra Mihaela Botez
e no vídeo https://www.youtube.com/watch?v=nhT56blfRpE"""


class genetic_algorithm:
    def __init__(self, initial_airfoil: bezier_airfoil, MAX_CHANGE: float, simulation_params: list, skip_cp_upper: int = 2, skip_cp_lower: int = 4):
        self.initial_cp_upper = initial_airfoil.cp_upper
        self.initial_cp_lower = initial_airfoil.cp_lower
        self.degree_upper = initial_airfoil.degree_upper
        self.degree_lower = initial_airfoil.degree_lower

        self.skip_cp_upper = skip_cp_upper
        self.skip_cp_lower = skip_cp_lower
        self.MAX_CHANGE = MAX_CHANGE

        self.alpha_f = simulation_params[0]
        self.alpha_step = simulation_params[1]
        self.Re = simulation_params[2]

        self.n_genomes = 0  # Número de genomas criados
        self.n_crossovers = 0  # Número de vezes que o crossover ocorreu
        self.n_mutation = 0  # Número de mutações em genes

    def generate_genome(self):
        """Gera um perfil e atribui caracteristicas aleatórias.
        Mantém alguns pontos no bordo de fuga e de ataque
        inalterados com base na variável 'skip_cp'.
        Isso ajuda a diminuir a ocorrência de perfis impossíveis,
        e agiliza o processo de otimização.
        """
        self.n_genomes = self.n_genomes + 1

        genome_upper = []
        genome_lower = []

        if (self.skip_cp_upper > self.degree_upper//2) or (self.skip_cp_lower > self.degree_lower//2):
            raise ValueError(
                '"skip_cp" precisa ser menor que a metade do grau de bezier')

        for i in range(self.skip_cp_upper):
            genome_upper.extend(
                [self.initial_cp_upper[i]])

        for i in range(self.skip_cp_lower):
            genome_lower.extend(
                [self.initial_cp_lower[i]])

        """Realiza a variação dos parâmetros aleatoriamente"""

        for i in self.initial_cp_upper[self.skip_cp_upper:self.degree_upper - self.skip_cp_upper + 1]:
            genome_upper.extend([[i[0] + random.uniform(-self.MAX_CHANGE, self.MAX_CHANGE),
                                  i[1] + random.uniform(-self.MAX_CHANGE, self.MAX_CHANGE)]])

        for i in self.initial_cp_lower[self.skip_cp_lower:self.degree_lower - self.skip_cp_lower + 1]:
            genome_lower.extend([[i[0] + random.uniform(-self.MAX_CHANGE, self.MAX_CHANGE),
                                  i[1] + random.uniform(-self.MAX_CHANGE, self.MAX_CHANGE)]])

        for i in range(self.skip_cp_upper):
            genome_upper.extend(
                [self.initial_cp_upper[self.degree_upper - self.skip_cp_upper + 1 + i]])

        for i in range(self.skip_cp_lower):
            genome_lower.extend(
                [self.initial_cp_lower[self.degree_lower - self.skip_cp_lower + 1 + i]])

        genome = bezier_airfoil()
        genome.set_name(f"Generated Airfoil {self.n_genomes}")
        genome.set_genome_points(genome_upper, genome_lower)

        """Gera o arquivo .dat e simula o genoma"""
        aux.save_as_dat_from_bezier(
            genome.cp_upper, genome.cp_lower, "generated_airfoil", header=f"Generated Airfoil {self.n_genomes}")
        genome.simulate(airfoil_path="airfoils/generated_airfoil.dat",
                        name=f"Generated Airfoil {self.n_genomes}",
                        alpha_f=self.alpha_f,
                        alpha_step=self.alpha_step,
                        Re=self.Re,
                        polar_path="src/xfoil_runner/data/genome_polar.txt")
        genome.get_opt_params()

        return genome

    def generate_population(self, pop_size: int):
        """Gera uma lista de objetos genomas"""
        population = []
        for _ in range(pop_size):
            population.append(self.generate_genome())
        return population

    def plot_population(self, population: list):
        """TEM QUE CONSERTAR AGORA QUE A POPULAÇÃO É DE OBJETOS"""
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

    @staticmethod
    def fitness(genome: bezier_airfoil, initial_airfoil: bezier_airfoil, weights=(0.4, 0.4, 0.2)):
        if genome.converged == True:
            """Calcula o custo baseado no perfil inicial"""
            fitness = weights[0]*(genome.Cl_integration / initial_airfoil.Cl_integration) + \
                weights[1]*(initial_airfoil.Cd_integration / genome.Cd_integration) + \
                weights[2]*(genome.stall_angle/initial_airfoil.stall_angle)
            genome.set_fitness(fitness)
            return fitness
        else:
            """Se o genoma não convergir na análise, ele é descartado"""
            genome.set_fitness(0)
            return 0

    def select_pair(self, population: list, initial_airfoil: bezier_airfoil, fitness_weights=(0.4, 0.4, 0.2)):
        """Seleciona um par dentre a população. Quanto maior o fitness, maior
        é a chance de ser escolhido"""
        if np.sum(fitness_weights) != 1:
            raise ValueError('Os pesos fornecidos precisam somar 1.')

        if len(population) < 2:
            raise ValueError(
                f"Não foi possível selecionar um par, pois a população possui {len(population)} genomas apenas.")

        """Os pesos utilizados para a seleção são os valores de fitness
        de cada perfil elevado à quinta potência, isso faz com que perfis com maior
        fitness tenham ainda mais probabilidade de serem escolhidos"""
        for genome in population:
            if genome.converged == False:
                population.remove(genome)

        return random.choices(
            population=population,
            weights=[self.fitness(genome, initial_airfoil, fitness_weights)**6
                     for genome in population],
            k=2
        )

    def crossover(self, parents_genome: list, probability: float):
        """Gera dois genomas cruzados a partir de dois genomas parentes a partir de uma certa probabilidade"""
        genome_a, genome_b = copy(parents_genome[0]), copy(parents_genome[1])

        if (len(genome_a.cp_upper) != len(genome_b.cp_upper)) or (len(genome_a.cp_lower) != len(genome_b.cp_lower)):
            raise ValueError(
                "Pontos de controle de genomas pais precisam ter o mesmo tamanho.")

        length_upper = len(genome_a.cp_upper)
        length_lower = len(genome_a.cp_lower)

        if random.random() <= probability:
            self.n_crossovers = self.n_crossovers + 1
            """Index é o ponto onde será realizado o cross over"""
            index_upper = random.randint(1, length_upper - 1)
            index_lower = random.randint(1, length_lower - 1)

            a_cp_upper = genome_a.cp_upper
            a_cp_lower = genome_a.cp_lower
            b_cp_upper = genome_b.cp_upper
            b_cp_lower = genome_b.cp_lower

            genome_a.set_genome_points(
                a_cp_upper[0:index_upper] + b_cp_upper[index_upper:], a_cp_lower[0:index_lower] + b_cp_lower[index_lower:])

            genome_b.set_genome_points(
                b_cp_upper[0:index_upper] + a_cp_upper[index_upper:], b_cp_lower[0:index_lower] + a_cp_lower[index_lower:])

            return [genome_a, genome_b]
        else:
            return parents_genome

    def mutate(self, genome: bezier_airfoil, probability: float, n_genes: int = 1):
        """Realiza a mutação de um gene a partir de uma certa probabilidade.
            - n_genes: Número de genes que serão mutados
        """

        if random.random() <= probability:
            self.n_mutation = self.n_mutation + 1

            cp_upper = copy(genome.cp_upper)
            cp_lower = copy(genome.cp_lower)

            for _ in range(n_genes):

                index_upper = random.randrange(
                    self.skip_cp_upper, len(cp_upper) - self.skip_cp_upper)

                index_lower = random.randrange(
                    self.skip_cp_lower, len(cp_lower) - self.skip_cp_lower)

                cp_upper[index_upper] = [cp_upper[index_upper][0] + random.uniform(-self.MAX_CHANGE, self.MAX_CHANGE),
                                         cp_upper[index_upper][1] + random.uniform(-self.MAX_CHANGE, self.MAX_CHANGE)]

                cp_lower[index_lower] = [cp_lower[index_lower][0] + random.uniform(-self.MAX_CHANGE, self.MAX_CHANGE),
                                         cp_lower[index_lower][1] + random.uniform(-self.MAX_CHANGE, self.MAX_CHANGE)]

            genome.set_genome_points(cp_upper, cp_lower)

    @ staticmethod
    def run_opt_genetic(cp):
        pass


def _example():
    alpha_step = 0.5
    Re = 200000
    alpha_f = 16

    """Cria um perfil a partir de um .dat e obtem os pontos de controle de Bezier"""
    initial_airfoil = bezier_airfoil()
    initial_airfoil.set_coords_from_dat("airfoils/s1223.dat")
    initial_airfoil.get_bezier_cp(8, 16)
    initial_airfoil.simulate(initial_airfoil.airfoil_path,
                             initial_airfoil.name,
                             alpha_f=alpha_f,
                             alpha_step=alpha_step,
                             Re=Re,
                             polar_path="src/xfoil_runner/data/initial_polar.txt")
    initial_airfoil.get_opt_params(
        polar_path="src/xfoil_runner/data/initial_polar.txt")

    gen = genetic_algorithm(
        initial_airfoil,
        MAX_CHANGE=0.04,
        simulation_params=[alpha_f, alpha_step, Re]
    )

    population = gen.generate_population(3)
    # plt.figure(figsize=(9, 3))
    # gen.plot_population(population)

    fitness_progression = [[1]]

    pair = gen.select_pair(
        population=population,
        initial_airfoil=initial_airfoil,
        fitness_weights=(0.4, 0.4, 0.2)
    )

    os.system('cls' if os.name == 'nt' else 'clear')
    print(pair)

    # gen.crossover(pair, 1)

    initial_airfoil_copy = copy(initial_airfoil)
    gen.mutate(initial_airfoil, 1)

    plt.show()

    """Cria uma instancia de perfil nova para o genoma"""
    # genome = bezier_airfoil()
    # genome_upper, genome_lower = gen.generate_genome()
    # genome.set_genome_points(genome_upper, genome_lower)

    """Gera o arquivo .dat e simula o genoma"""
    # aux.save_as_dat_from_bezier(
    #     genome.cp_upper
    #     , genome.cp_lower, "generated_airfoil")
    # genome.simulate("airfoils/generated_airfoil.dat",
    #                 "Genome Airfoil", alpha_f=alpha_f, alpha_step=alpha_step, Re=Re, polar_path="src/xfoil_runner/data/genome_polar.txt")
    # genome.get_opt_params()

    """Calcula o fitness do genoma"""
    # fitness = gen.fitness(genome, initial_airfoil)
    # os.system('cls' if os.name == 'nt' else 'clear')
    # print(fitness)
    # fig, axs = plt.subplots(2, 2)
    # plot_polar(axs, "src/xfoil_runner/data/initial_polar.txt")
    # plot_polar(axs, "src/xfoil_runner/data/genome_polar.txt")
    # plt.show()


if __name__ == "__main__":
    _example()
