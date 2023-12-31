# Portfolio Optimisation Problem

this repo contain two optimisation algorithms to optimise the Portfolio Optimisation Problem. The first algorithm is a Local Search Algorithm and the second is a Genetic Algorithm.

## Solution Representation
The solution to the Portfolio Optimization Problem is represented as a List, where each element corresponds to the weight or proportion of the investment allocated to a particular asset. The length of this List is equal to the number of assets (N). The weights are constrained to be  non-negative and sum to 1. The initial population is created by generating random solutions, which are then normalized to meet the constraints of the problem.


## Genetic Seach 
Genetic Algorithm will start after creating the population to perform four operations, the first operation is Selection: A binary tournament selection process is employed, which compares the fitness of two randomly chosen solutions and selects the better one to become a  parent for the next generation. The second operation is Crossover: A single-point crossover  mechanism is used, where a random crossover point is chosen, and the genes (asset weights) are exchanged between two parents to create offspring, followed by normalization to maintain the constraints. The third operation is Mutation: Mutation is introduced by randomly altering the asset weights in a solution, followed by normalization to maintain the constraints. The fourth operation is Weak Replacement: if the created offspring is fitter than the worst in the population, then overwrite the worst with the new solution.


## Local Search 
It will start by initializing the solution randomly and implementing the best improvement move. The i-th neighbor of a solution is generated by changing the i-th weight. When the algorithm gets stuck in a local maximum, it will keep a record of this local maximum and will restart the search again randomly.


Both algorithms will complete processing until they reach a 10,000-fitness evaluation.


