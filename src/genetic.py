import aux
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tabulate import tabulate
from copy import copy
from xfoil_runner.xfoil import run_xfoil, plot_polar
from bezier import bezier_airfoil

"""Baseado em https://github.com/ashokolarov/Genetic-airfoil,
no paper Two-dimensional airfoil shape optimization for airfoils at low speeds - Ruxandra Mihaela Botez
e no vídeo https://www.youtube.com/watch?v=nhT56blfRpE"""


class genetic_algorithm:
    def __init__(
        self, initial_airfoil: bezier_airfoil,
        simulation_params: tuple,
        MAX_CHANGE: float = 0.04,
        skip_cp_upper: int = 2, skip_cp_lower: int = 4,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.05, mutation_n_genes: int = 1,
        fitness_weights: tuple = (0.4, 0.4, 0.2)
    ):

        self.initial_airfoil = initial_airfoil
        self.initial_cp_upper = initial_airfoil.cp_upper
        self.initial_cp_lower = initial_airfoil.cp_lower
        self.degree_upper = initial_airfoil.degree_upper
        self.degree_lower = initial_airfoil.degree_lower

        # ---Opt. Params---
        self.skip_cp_upper = skip_cp_upper  # Pontos de controle que serão inalterados
        self.skip_cp_lower = skip_cp_lower  # Pontos de controle que serão inalterados
        self.MAX_CHANGE = MAX_CHANGE  # Variação máxima da posição dos pontos de controle
        self.crossover_prob = crossover_prob  # Probabilidade de ocorrência de crossover
        self.mutation_prob = mutation_prob  # Probabilidade de ocorrência de mutação
        self.mutation_n_genes = mutation_n_genes  # Número de genes que serão mutados
        # ------------------

        # ---Simulation Params---
        self.alpha_f = simulation_params[0]  # Alfa final simulado
        self.alpha_step = simulation_params[1]  # Variação do alfa
        self.Re = simulation_params[2]  # Número de Reynolds da simulação
        self.fitness_weights = fitness_weights  # Pesos para avaliação dos genomas
        # ------------------------

        # ---Running Data---
        self.n_genomes = 0  # Número de genomas criados
        self.n_crossovers = 0  # Número de vezes que o crossover ocorreu
        self.n_mutation = 0  # Número de mutações em genes
        self.n_generation = 0  # Número da geração atual
        self.fitness_progression = [1]  # Progressão da função custo
        # ------------------

    def generate_genome(self):
        """Gera um perfil e atribui caracteristicas aleatórias.
        Mantém alguns pontos no bordo de fuga e de ataque
        inalterados com base na variável 'skip_cp'.
        Isso ajuda a diminuir a ocorrência de perfis impossíveis,
        e agiliza o processo de otimização.
        """
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

        self.n_genomes = self.n_genomes + 1
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
    def fitness(genome: bezier_airfoil, initial_airfoil: bezier_airfoil, fitness_weights: tuple = (0.4, 0.4, 0.2)):
        if genome.converged == True:
            """Calcula o custo baseado no perfil inicial"""
            fitness = fitness_weights[0]*(genome.Cl_integration / initial_airfoil.Cl_integration) + \
                fitness_weights[1]*(initial_airfoil.Cd_integration / genome.Cd_integration) + \
                fitness_weights[2]*(genome.stall_angle /
                                    initial_airfoil.stall_angle)
            genome.set_fitness(fitness)
            return fitness
        else:
            """Se o genoma não convergir na análise, ele é descartado"""
            genome.set_fitness(0)
            return 0

    def select_pair(self, population: list):
        """Seleciona um par dentre a população. Quanto maior o fitness, maior
        é a chance de ser escolhido"""

        _population = copy(population)

        if np.sum(self.fitness_weights) != 1:
            raise ValueError('Os pesos fornecidos precisam somar 1.')

        if len(_population) < 2:
            raise ValueError(
                f"Não foi possível selecionar um par, pois a população possui {len(_population)} genomas apenas.")

        """Os pesos utilizados para a seleção são os valores de fitness
        de cada perfil elevado à sexta potência, isso faz com que perfis com maior
        fitness tenham ainda mais probabilidade de serem escolhidos"""
        for genome in _population:
            if genome.converged == False:
                _population.remove(genome)

        pair = []
        pair += random.choices(
            population=_population,
            weights=[self.fitness(genome, self.initial_airfoil, self.fitness_weights)**6
                     for genome in _population],
            k=1
        )

        """Remove o genoma escolhido para impedir 2 genomas iguais no crossover"""
        _population.remove(pair[0])

        pair += random.choices(
            population=_population,
            weights=[self.fitness(genome, self.initial_airfoil, self.fitness_weights)**6
                     for genome in _population],
            k=1
        )

        return pair

    def crossover(self, parents_genome: list, probability: float):
        """Gera dois genomas cruzados a partir de dois genomas parentes com uma probabilidade"""
        genome_a, genome_b = copy(parents_genome[0]), copy(parents_genome[1])

        if (len(genome_a.cp_upper) != len(genome_b.cp_upper)) or (len(genome_a.cp_lower) != len(genome_b.cp_lower)):
            raise ValueError(
                "Pontos de controle de genomas pais precisam ter o mesmo tamanho.")

        length_upper = len(genome_a.cp_upper)
        length_lower = len(genome_a.cp_lower)

        if random.random() <= probability:
            self.n_crossovers += 1
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

            return genome_a, genome_b
        else:
            return parents_genome[0], parents_genome[1]

    def mutate(self, genome: bezier_airfoil, probability: float, n_genes: int = 1):
        """Realiza a mutação de um gene a partir de uma certa probabilidade.
            - n_genes: Número de genes que serão mutados
        """

        if random.random() <= probability:
            self.n_mutation += 1

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

    def run_evolution(self, fitness_limit: int = 2, population_size: int = 10, generation_limit: int = 100):
        population = self.generate_population(population_size)
        self.initial_population = sorted(
            population,
            key=lambda genome: self.fitness(genome, self.initial_airfoil),
            reverse=True
        )

        # plt.figure(figsize=(9, 3))
        # gen.plot_population(population)

        for _ in range(generation_limit):
            self.n_generation += 1

            population = sorted(
                population,
                key=lambda genome: self.fitness(genome, self.initial_airfoil),
                reverse=True
            )

            self.fitness_progression.append(population[0].fitness)

            """Critério de parada por limite de custo"""
            if self.fitness(population[0], self.initial_airfoil) >= fitness_limit:
                break

            """Mantém os 2 melhores genomas para a proxima população"""
            next_generation = population[:2]

            """Adiciona os genomas restantes realizando crossover e mutações"""
            for j in range(int(population_size/2) - 1):
                parents = self.select_pair(population)

                child_a, child_b = self.crossover(parents, self.crossover_prob)
                self.mutate(child_a, self.mutation_prob, self.mutation_n_genes)
                self.mutate(child_b, self.mutation_prob, self.mutation_n_genes)

                self.n_genomes += 1

                aux.save_as_dat_from_bezier(
                    child_a.cp_upper,
                    child_a.cp_lower,
                    name="generated_airfoil",
                    header=f"Generated Airfoil G{self.n_generation}N{self.n_genomes}")
                child_a.simulate("airfoils/generated_airfoil.dat",
                                 "Genome Airfoil",
                                 alpha_f=self.alpha_f,
                                 alpha_step=self.alpha_step,
                                 Re=self.Re,
                                 polar_path="src/xfoil_runner/data/genome_polar.txt")
                child_a.get_opt_params()

                self.n_genomes += 1

                aux.save_as_dat_from_bezier(
                    child_b.cp_upper,
                    child_b.cp_lower,
                    name="generated_airfoil",
                    header=f"Generated Airfoil G{self.n_generation}N{self.n_genomes}")
                child_b.simulate("airfoils/generated_airfoil.dat",
                                 "Genome Airfoil",
                                 alpha_f=self.alpha_f,
                                 alpha_step=self.alpha_step,
                                 Re=self.Re,
                                 polar_path="src/xfoil_runner/data/genome_polar.txt")
                child_b.get_opt_params()

                next_generation.append(child_a)
                next_generation.append(child_b)

            population = next_generation

        """Última população encontrada"""
        self.final_population = sorted(
            population,
            key=lambda genome: self.fitness(genome, self.initial_airfoil),
            reverse=True
        )


def _example():
    """Parâmetros para a simulação"""
    alpha_step = 0.5
    Re = 200000
    alpha_f = 15

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
        initial_airfoil=initial_airfoil,
        MAX_CHANGE=0.04,
        simulation_params=(alpha_f, alpha_step, Re),
        skip_cp_upper=2,
        skip_cp_lower=4,
        crossover_prob=0.8,
        mutation_prob=0.15,
        mutation_n_genes=2,
        fitness_weights=(0.5, 0.3, 0.2)
    )

    start = time.time()

    gen.run_evolution(
        fitness_limit=2,
        population_size=20,
        generation_limit=20
    )

    end = time.time()

    os.system('cls' if os.name == 'nt' else 'clear')

    table = tabulate([['Parameter', 'Initial airfoil', 'Optimized airfoil'],
                      ['Cl_Integration', gen.initial_airfoil.Cl_integration,
                          gen.final_population[0].Cl_integration],
                      ['Cd_Integration', gen.initial_airfoil.Cd_integration,
                          gen.final_population[0].Cd_integration],
                      ['Stall_Angle', initial_airfoil.stall_angle, gen.final_population[0].stall_angle]],
                     headers='firstrow')
    print(table)

    print()
    print(gen.fitness_progression)
    print()
    print(f"Time: {end-start}s")

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
