from src.bezier import bezier_airfoil
import numpy as np
import random
import matplotlib.pyplot as plt
import os


def generate_genome(length: int):
    genome = bezier_airfoil()
    return genome


def run_opt_genetic(cp):
    pass


def _example():
    airfoil = bezier_airfoil("airfoils/s1223.dat")
    cp = airfoil.get_bezier_cp(20)
    airfoil.simulate()
    airfoil.get_opt_params()
    run_opt_genetic(cp)


if __name__ == "__main__":
    _example()
