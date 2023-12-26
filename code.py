import random
import numpy as np
import matplotlib.pyplot as plt
termination_counter=0
fitness_avg=[]

def Create_Initial_Population(population_size, num_assets, seed=None):
    if seed is not None:
        random.seed(seed)
    population = []
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

def compute_fitness_function(individual,expected_returns,risk_free_rate,covariance_matrix):
    global termination_counter
    # compute the numerator
    C = np.dot(individual, expected_returns)
    numerator=C-risk_free_rate

    # compute the denominator
    denominator=np.sqrt(np.dot(individual, np.dot(covariance_matrix, individual)))

    fitness_value=numerator/denominator

    termination_counter=termination_counter+1

    return fitness_value



#==========================================================
#GA

def binary_tournament_selection(population,fitness_values):
    #Randomly choose two chromosomes from the population
    index_a=random.choice(range(len(population)))
    a=population[index_a]
    index_b = random.choice(range(len(population)))
    b = population[index_b]


    a_fitness_value = fitness_values[index_a]
    b_fitness_value = fitness_values[index_b]
    #The fittest of these two becomes the selected parent.
    parent=b
    if a_fitness_value>b_fitness_value:
        parent=a
    return parent



def single_point_crossover(parent1, parent2):

    #Randomly select a ‘crossover point’ which should be smaller than the total length of the chromosome
    crossover_point=random.randint(1,len(parent1)-1)
    #print(f"crossover_point= {crossover_point}")

    #Take the two parents, and swap the gene values between them only for those genes
    #which appear after the crossover point to create two new children
    child1=repair_solution(np.append(parent1[:crossover_point],parent2[crossover_point:]))
    child2 = repair_solution(np.append(parent2[:crossover_point], parent1[crossover_point:]))

    childern=[]
    childern.append(child1)
    childern.append(child2)

    return childern



def Mutation_function(chromosome, k):
    for _ in range(k):
        # Choose a random index in the
        rand_index = random.randint(0, len(chromosome) - 1)

        # Change it to a random new value within the permissible range
        chromosome[rand_index] = random.random()

    chromosome = repair_solution(chromosome)
    return chromosome



def weak_replace(chromosome,chromosome_fitness , population, fitness_values):
  min_fitness = min(fitness_values)  # Finding the minimum value
  min_index = fitness_values.index(min_fitness)  # Finding the index of the minimum value
  if(min_fitness<chromosome_fitness):
    population[min_index]=chromosome
    fitness_values[min_index]=chromosome_fitness

  return population ,fitness_values


def Genetic_search(population_size,k,num_assets,expected_returns, risk_free_rate, covariance_matrix,cross_condition,seed ):

#step1
#call create population and repair
#now we have 2 arrays popultaion and fitness

#for each run declear again
  global termination_counter
  global fitness_avg
  termination_counter=0
  fitness_avg=[]
  population=Create_Initial_Population(population_size, num_assets,seed)
  repaired_population = []
  for solution in population:
      repaired_solution = repair_solution(solution)
      repaired_population.append(repaired_solution)

  fitness_values=[]
  for individual in repaired_population:
    fitness_values.append(compute_fitness_function(individual, expected_returns, risk_free_rate, covariance_matrix))

  x_axis=[]
  avg_value=sum(fitness_values) / len(fitness_values)
  fitness_avg.append(avg_value)
  x_axis.append(termination_counter)

  print (f"T{seed}")
  print (f"the best fitness value in the population before the search: {max(fitness_values)}")

  #modify to 10000
  while(termination_counter!=10000):

    #array of avg to plot in graph
    avg_value=sum(fitness_values) / len(fitness_values)
    fitness_avg.append(avg_value)
    x_axis.append(termination_counter)



  #step2
  #choose 2 parent a and b
    a=binary_tournament_selection(repaired_population,fitness_values)
    b=binary_tournament_selection(repaired_population,fitness_values)

  #step3
  #single-point crossover give 2 children e and f.
  # e and f values is the cross-over calling return value
    if(cross_condition):
      childern=single_point_crossover(a, b)
      e=childern[0]
      f=childern[1]
    else:
      e=a
      f=b

  #step4
  # mutation give two new solutions u and v
    if (k != 0 ):
      u = Mutation_function(e, k);
      v = Mutation_function(f, k);
    else:
      u = e
      v = f

  #step 5
  #Run weakest replacement, firstly for u, then v.
  #when replace update the fitness value array also
    u_fitness= compute_fitness_function(u, expected_returns, risk_free_rate, covariance_matrix)
    v_fitness=compute_fitness_function(v, expected_returns, risk_free_rate, covariance_matrix)
    repaired_population, fitness_values= weak_replace(u,u_fitness,repaired_population,fitness_values);
    repaired_population, fitness_values= weak_replace(v,v_fitness,repaired_population,fitness_values);

    if (termination_counter==10000):
      avg_value=sum(fitness_values) / len(fitness_values)
      fitness_avg.append(avg_value)
      x_axis.append(termination_counter)
      print (f"the best fitness value in the population after the search: {max(fitness_values)}")
      print (f"the average value of population fitness before the search: {fitness_avg[0]}")
      print (f"the average value of population fitness after the search: {fitness_avg[len(fitness_avg)-1]}")
      print (f"the change in average: {fitness_avg[len(fitness_avg)-1]-fitness_avg[0]}")
      print(f"the best solution: {repaired_population[fitness_values.index(max(fitness_values))]}")
      plt.plot(x_axis,fitness_avg, label=f"T{seed}")


#the end of the genetic function =============================================================================


#Strart of LS ================================================================================================
# Generate ith Neighbor
def Generate_ith_Neighbor(solution, i):
    new_neighbor = solution.copy()
    new_neighbor[i] = 1 - solution[i]
    return new_neighbor

def Local_Search(population_size, num_assets, expected_returns, risk_free_rate, covariance_matrix, seed):
    best_local_maximum = None
    best_fitness = float('-inf')
    global termination_counter
    termination_counter = 0
    best_fitness_values = []
    termination = []


    while termination_counter != 10000:

        if termination_counter == 10000:
            break

        if termination_counter == 0:#if it the first time send the main's seed
           population = Create_Initial_Population(population_size, num_assets, seed)
           solution = population[0]
           normalized_data = repair_solution(solution)
           fitness_value = compute_fitness_function(normalized_data, expected_returns, risk_free_rate, covariance_matrix)

           best_fitness_values.append(best_fitness)
           termination.append(termination_counter)

           print(f"LS-T{seed}\n start solution:{solution}\n fitness_value:{fitness_value}")

        else:
          population = Create_Initial_Population(population_size, num_assets)
          solution = population[0]
          normalized_data = repair_solution(solution)
          fitness_value = compute_fitness_function(normalized_data, expected_returns, risk_free_rate, covariance_matrix)

          best_fitness_values.append(best_fitness)
          termination.append(termination_counter)

        if fitness_value > best_fitness:
            best_fitness = fitness_value
            best_local_maximum = normalized_data


        for i in range(num_assets):

            if termination_counter == 10000:
                break

            new_neighbor = Generate_ith_Neighbor(normalized_data, i)
            normalized_neighbor = repair_solution(new_neighbor)
            neighbor_fitness_value = compute_fitness_function(normalized_neighbor, expected_returns, risk_free_rate,covariance_matrix)


            if neighbor_fitness_value > best_fitness:
                best_fitness = neighbor_fitness_value
                best_local_maximum = new_neighbor

            best_fitness_values.append(best_fitness)
            termination.append(termination_counter)


        # After the loop
        if termination_counter == 10000:
            print(f"best_local_maximum:{best_local_maximum}")
            plt.plot(termination, best_fitness_values, label=f"LS-T{seed}")
    return best_fitness
#End of LS ===================================================================================================


def main():
    print("start")

    # ask the user either insert data or use the provided one 
    choice = input("Do you want to use the provided instance (Y) or insert your own (N)? ").lower()
    
    if choice == 'y':
        budget = 250000
        num_assets = 10
        risk_free_rate = 0.06
        expected_returns = [0.003, 0.003, 0.001, 0.006, 0.002, 0.001, 0.0004, 0.01, 0.004, 0.02]
        covariance_matrix = [
            [0.000229, 0.000028, -0.000005, 0.000015, -0.000017, 0.000000, -0.000004, -0.000089, 0.000022, -0.000003],
            [0.000028, 0.000302, 0.000017, 0.000013, -0.000003, 0.000002, -0.000003, 0.000053, 0.000002, -0.000004],
            [-0.000005, 0.000017, 0.001759, -0.000016, 0.000011, 0.000000, -0.000029, -0.004438, 0.000044, 0.000007],
            [0.000015, 0.000013, -0.000016, 0.015125, 0.000181, 0.003710, -0.000040, -0.000431, 0.000143, 0.000023],
            [-0.000017, -0.000003, 0.000011, 0.000181, 0.000300, 0.000061, -0.000015, 0.000341, -0.000006, 0.000008],
            [0.000000, 0.000002, 0.000000, 0.003710, 0.000061, 0.001209, -0.000012, -0.000092, 0.000033, 0.000018],
            [-0.000004, -0.000003, -0.000029, -0.000040, -0.000015, -0.000012, 0.000305, 0.000158, -0.000004, 0.000025],
            [-0.000089, 0.000053, -0.004438, -0.000431, 0.000341, -0.000092, 0.000158, 0.092052, -0.000040, 0.000019],
            [0.000022, 0.000002, 0.000044, 0.000143, -0.000006, 0.000033, -0.000004, -0.000040, 0.000149, 0.000007],
            [-0.000003, -0.000004, 0.000007, 0.000023, 0.000008, 0.000018, 0.000025, 0.000019, 0.000007, 0.000322]
        ]
    else:
        while True:
            try:
                budget = float(input("Enter the budget: "))
                break
            except ValueError:
                print("Invalid input, please enter a valid number.")

        while True:
            try:
                num_assets = int(input("Enter the number of assets: "))
                break
            except ValueError:
                print("Invalid input, please enter a valid integer.")

        while True:
            try:
                risk_free_rate = float(input("Enter the risk-free rate: "))
                break
            except ValueError:
                print("Invalid input, please enter a valid number.")

        # Get expected returns for each asset
        expected_returns = []
        for i in range(num_assets):
            while True:
                try:
                    expected_return = float(input(f"Enter the expected return for asset {i + 1}: "))
                    expected_returns.append(expected_return)
                    break
                except ValueError:
                    print("Invalid input, please enter a valid number.")

        # Get covariance matrix values
        covariance_matrix = []
        for i in range(num_assets):
            row = []
            for j in range(num_assets):
                while True:
                    try:
                        covariance = float(input(f"Enter the value in covariance matrix between {i + 1} and {j + 1}: "))
                        row.append(covariance)
                        break
                    except ValueError:
                        print("Invalid input, please enter a valid number.")
            covariance_matrix.append(row)



    #EX1
    print ("Experiment 1")
    Genetic_search(10 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,1)
    Genetic_search(10 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,2)
    Genetic_search(10 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,3)
    Genetic_search(10 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,4)
    Genetic_search(10 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,5)
    plt.legend()
    plt.xlabel('fitness evaluation')
    plt.ylabel('population fitness average')
    plt.title("Experiment 1\nPS:10 , CO:1 , M:1")
    plt.show()

    print ("Experiment 2")
    # #EX2
    Genetic_search(100 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,1)
    Genetic_search(100 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,2)
    Genetic_search(100 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,3)
    Genetic_search(100 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,4)
    Genetic_search(100 ,1,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,5)
    plt.legend()
    plt.xlabel('fitness evaluation')
    plt.ylabel('population fitness average')
    plt.title("Experiment 2\nPS:100, CO:1, M:1")
    plt.show()

    #EX3
    print ("Experiment 3")
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,1)
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,2)
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,3)
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,4)
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,5)
    plt.legend()
    plt.xlabel('fitness evaluation')
    plt.ylabel('population fitness average')
    plt.title("Experiment 3\nPS:10, CO:1, M:5")
    plt.show()

    #EX4
    print ("Experiment 4")
    Genetic_search(100 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,1)
    Genetic_search(100 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,2)
    Genetic_search(100 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,3)
    Genetic_search(100 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,4)
    Genetic_search(100 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,5)
    plt.legend()
    plt.xlabel('fitness evaluation')
    plt.ylabel('population fitness average')
    plt.title("Experiment 4\nPS:100, CO:1, M:5")
    plt.show()

    #EX5
    print ("Experiment 5")
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,False,1)
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,False,2)
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,False,3)
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,False,4)
    Genetic_search(10 ,5,num_assets,expected_returns, risk_free_rate, covariance_matrix,False,5)
    plt.legend()
    plt.xlabel('fitness evaluation')
    plt.ylabel('population fitness average')
    plt.title("Experiment 5\nPS:10, CO:0, M:5")
    plt.show()

    #EX6
    print ("Experiment 6")
    Genetic_search(10 ,0,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,1)
    Genetic_search(10 ,0,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,2)
    Genetic_search(10 ,0,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,3)
    Genetic_search(10 ,0,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,4)
    Genetic_search(10 ,0,num_assets,expected_returns, risk_free_rate, covariance_matrix,True,5)
    plt.legend()
    plt.xlabel('fitness evaluation')
    plt.ylabel('population fitness average')
    plt.title("Experiment 6\nPS:10, CO:1, M:0")
    plt.show()

    #Local Search
    print ("Local Search")
    best_fitness = Local_Search(1,num_assets,expected_returns, risk_free_rate, covariance_matrix,seed = 1)
    print(f"best_fitness_LS_T1: {best_fitness}")
    print("-" * 50)

    best_fitness = Local_Search(1,num_assets,expected_returns, risk_free_rate, covariance_matrix,seed = 2)
    print(f"best_fitness_LS_T2: {best_fitness}")
    print("-" * 50)

    best_fitness = Local_Search(1,num_assets,expected_returns, risk_free_rate, covariance_matrix,seed = 3)
    print(f"best_fitness_LS_T3: {best_fitness}")
    print("-" * 50)

    best_fitness = Local_Search(1,num_assets,expected_returns, risk_free_rate, covariance_matrix,seed = 4)
    print(f"best_fitness_LS_T4: {best_fitness}")
    print("-" * 50)

    best_fitness = Local_Search(1,num_assets,expected_returns, risk_free_rate, covariance_matrix,seed = 5)
    print(f"best_fitness_LS_T5: {best_fitness}")
    print("-" * 50)

    plt.legend()
    plt.legend()
    plt.ylabel('best fitness evaluation')
    plt.xlabel('termination counter')
    plt.title("LS")
    plt.show()
    plt.show()

if __name__ == "__main__":
    main()