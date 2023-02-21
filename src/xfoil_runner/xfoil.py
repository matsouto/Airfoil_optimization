import os
import subprocess
import numpy as np
import sys

"""
Baseado em https://github.com/JARC99/xfoil-runner e https://www.youtube.com/watch?v=zGZin_PPLdc.

Outros materiais:
https://github.com/karanchawla/Airfoil-Optimization/tree/master/xfoil
"""


def run_xfoil(airfoil_path, airfoil_name, alpha_i=0, alpha_f=10, alpha_step=0.25, Re=1000000, n_iter=100):

    if sys.platform.startswith('win32'):
        XFOIL_BIN = "xfoil.exe"
    elif sys.platform.startswith('darwin'):
        XFOIL_BIN = "xfoil"
    elif sys.platform.startswith('linux'):
        XFOIL_BIN = "xfoil"

    """XFOIL input file writer"""
    if os.path.exists("data/polar_file.txt"):
        os.remove("data/polar_file.txt")

    with open("src/xfoil_runner/data/input_file.in", 'w') as file:
        file.write(f"LOAD {airfoil_path}\n")
        file.write(airfoil_name + '\n')
        file.write("PANE\n")
        file.write("OPER\n")
        file.write("Visc {0}\n".format(Re))
        file.write("PACC\n")
        file.write("src/xfoil_runner/data/polar_file.txt\n\n")
        file.write("ITER {0}\n".format(n_iter))
        file.write("ASeq {0} {1} {2}\n".format(alpha_i, alpha_f,
                                               alpha_step))
        file.write("\n\n")
        file.write("quit\n")

    subprocess.call(
        f"{XFOIL_BIN} < src/xfoil_runner/data/input_file.in", shell=True)

    polar_data = np.loadtxt(
        "src/xfoil_runner/data/polar_file.txt", skiprows=12)


def _example():
    run_xfoil("airfoils/s1223.dat", "s1223")


"""Se esse arquivo for executado, rode _example()"""
if __name__ == "__main__":
    _example()
