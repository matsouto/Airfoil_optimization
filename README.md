<div align="center">
  <H1> Genetic Airfoil Otimization Algorithm </H1>
</div>

<div align="center">
  <video src="https://user-images.githubusercontent.com/90715995/235260206-a1e6b36a-dd23-4d2d-a1ca-7b1e782c565d.mp4" width="200" />
</div>

This program was created as an attempt to find an optimized airfoil based on the S1223, but can be used in many different occasions.

Some of the features are:
- Parametrize airfoils using the Bezier Curves method;
- Simulate airfoils using XFOIL;
- Generate polar plots;
- Optimize the geometry for both symmetrical and non symmetrical airfoils.  

The default optimization goal of the algorithm is to find a better _Cl_ x _Cd_ ratio for a specific alpha range, considering a low Reynolds number.   
 
## Basic Functionalities:
- `/xfoil_runner`: Basic functions used to perform simulations using XFOIL (binaries included); 
- `/bezier`: Contains the _bezier_airfoil_ base class and all methods used for creating, plotting and parametrizing, using the Bezier Curves;

<p align="center">
    <img width="700" src=https://user-images.githubusercontent.com/90715995/235260061-d1c8816e-d494-4bc1-a2a6-a01ed4dd7717.jpg>
</p>

- `/genetic`: Contains the _genetic_algorithm_ class along with all the methods to perform the genetic optimization (mutation, crossover, etc...). 

## Genetic Optimization
The expected walkthrough of the genetic algorithm is as follows:

1. Parametrize the base airfoil (s1223 by default);
2. Generate the polar for the base airfoil;
3. Create the first population by varying slightly the control points of the base airfoil; 
4. Perform mutation and crossover based on the determined probabilities and create first generation;
5. Create the subsequent generations until the stop criteria is accomplished or the generation limit is achieved. 

The default stop criteria is to obtain a minimum fitness value, based on the fitness function. The fitness function evaluates the polar of the genomes and returns a float value based on a weighted average of _Cl_, _Cd_ slope integration and stall angle.
