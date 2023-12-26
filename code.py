import random
import numpy as np
from tabulate import tabulate

#first function
def Create_Initial_Population(population_size, num_assets):
    population = []
    # generate random population
    for _ in range(population_size):
        solution = [random.uniform(0, 1) for _ in range(num_assets)]
        population.append(solution)

    return population


#Second function
def repair_solution(solution):
    # Calculate min and max will be used in normalization
    min_val = np.min(solution)
    max_val = np.max(solution)

    # Min-Max normalization constrain 1
    normalized_data = (solution - min_val) / (max_val - min_val)

    # sum of the weights is = 1
    normalized_data /= np.sum(normalized_data)

    return normalized_data

#third function
def compute_fitness_function(individual,expected_returns,risk_free_rate,covariance_matrix):

    # compute the numerator
    C = np.dot(individual, expected_returns)
    numerator=C-risk_free_rate

    # compute the denominator
    denominator=np.sqrt(np.dot(individual, np.dot(covariance_matrix, individual)))

    fitness_value=numerator/denominator

    return fitness_value


def main():

    # Read from the user
    budget = float(input("Enter the budget: "))

    num_assets = int(input("Enter the number of assets: "))

    population_size = int(input("Enter the population size: "))

    risk_free_rate = float(input("Enter the risk-free rate: "))

    expected_returns = []
    for i in range(num_assets):
        expected_return = float(input(f"Enter the expected return for asset {i + 1}: "))
        expected_returns.append(expected_return)

    covariance_matrix = []
    for i in range(num_assets):
        row = []
        for j in range(num_assets):
            covariance = float(input(f"Enter the value in covariance matrix between {i + 1} and {j + 1}: "))
            row.append(covariance)
        covariance_matrix.append(row)

    # Print all the entered values
    print(f"Budget: {budget}")
    print(f"Number of assets: {num_assets}")
    print(f"Population size: {population_size}")
    print(f"Risk-free rate: {risk_free_rate}")
    print(f"Expected returns: {expected_returns}")
    print("Covariance matrix:")
    for row in covariance_matrix:
        print(row)

    print("----------------------------------------------------------------------------")

    # Create the initial population the return value is matrix
    initial_population = Create_Initial_Population(population_size, num_assets)

    repaired_initial_population = []

    # repair each solution
    for solution in initial_population:
      repaired_solution = repair_solution(solution)
      repaired_initial_population.append(repaired_solution)


    # send each individual to calculate the fitness value and print the output in table format
    table_data = []
    for i, solution in enumerate(repaired_initial_population):
        fitness_value = compute_fitness_function(solution, expected_returns, risk_free_rate, covariance_matrix)
        table_data.append([i + 1, solution, fitness_value])

    headers = ["Solution", "Portfolio Weights", "Fitness Value"]
    table = tabulate(table_data, headers, tablefmt="pipe")
    print(table)


if __name__ == "__main__":
    main()