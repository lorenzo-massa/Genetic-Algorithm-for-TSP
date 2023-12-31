import Reporter
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class r0978639:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename, verbose=False, parallelization=False):
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

		# Your code here.

        # PARAMETERS
        self.n_cities = distanceMatrix.shape[0]
        self.lambda_= 25 if self.n_cities < 501 else 12
        if parallelization:
            self.lambda_ = self.lambda_ * 2
        self.mu=self.lambda_*2
        self.alphaList = np.array([0.7, 0.7, 0.7, 0.7]) 
        self.k_for_selection = np.array([2, 3, 4, 5])
        self.max_iterations= 10000
        self.MAX_DIFFERENT_BEST_SOLUTIONS = 1000

        # ISLAND MODEL
        self.n_population = 4
        self.iterationIneraction = 50

        # Initialize the n_population populations  
        populations = []
        for i in range(self.n_population):
            populations.append(initialize(self.alphaList[i], distanceMatrix, self.n_cities, self.lambda_))
            populations[i] = two_opt(populations[i], distanceMatrix)


        # Store the progress of each population
        meanObjectiveList = [[] for i in range(self.n_population)]
        bestObjectiveList = [[] for i in range(self.n_population)]
        bestSolutionList = [[] for i in range(self.n_population)]
        alphaMeanList = [[] for i in range(self.n_population)]

        # Termination criteria
        differentBestSolutions = [0] * self.n_population

        # Tuning parameter for the local search
        #tuning = [2] * n_population

        # Set up the loop
        iteration = 0
        yourConvergenceTestsHere = True

        # Set up the executor
        executor = ProcessPoolExecutor(max_workers=self.n_population)

        while( yourConvergenceTestsHere ):
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])

			# Your code here.

            if parallelization:
                populations = list(executor.map(process_island, populations, 
                                                    [distanceMatrix]*self.n_population, 
                                                    [self.n_cities]*self.n_population, 
                                                    [self.lambda_]*self.n_population, 
                                                    [self.mu]*self.n_population, 
                                                    self.k_for_selection, 
                                                    self.alphaList))
                            

            # For each population
            for i in range(self.n_population):      

                if not parallelization:
                    populations[i] = process_island(populations[i], distanceMatrix, self.n_cities, self.lambda_, self.mu, self.k_for_selection[i], self.alphaList[i])      

                # Compute and save progress
                fvals = objf_pop(populations[i], distanceMatrix, self.n_cities)                                    # Compute the objective function value for each individual
                meanObjectiveList[i].append(np.mean(fvals))                         # Compute the mean objective value of the population           
                bestObjectiveList[i].append(np.min(fvals))                          # Store the new best objective
                bestSolutionList[i].append(populations[i][np.argmin(fvals), :])     # Store the new best solution
                alphaMeanList[i].append(np.mean(populations[i][:, -1]))             # Mean of alpha values

                # Adaptation of parameters
                if iteration > 1 and bestObjectiveList[i][-1] == bestObjectiveList[i][-2] and bestObjectiveList[i][-1] != np.inf:
                    differentBestSolutions[i] += 1
                else:
                    differentBestSolutions[i] = 0

            # Interact the populations 
            if iteration % self.iterationIneraction == 0:
                # Join the n_population populations
                completePopulation = np.vstack(populations)
                # Shuffle the complete population
                np.random.shuffle(completePopulation)
                # Create n_population from the complete population
                for i in range(self.n_population):
                    populations[i] = completePopulation[i*self.lambda_:(i+1)*self.lambda_, :]


            # Find the best solution among the n_population populations
            bestSolution = bestSolutionList[0][-1][:-1]
            bestObjective = bestObjectiveList[0][-1]
            meanObjective = meanObjectiveList[0][-1]
                
            for i in range(1,self.n_population):
                if bestObjectiveList[i][-1] < bestObjective:
                    bestObjective = bestObjectiveList[i][-1]
                    bestSolution = bestSolutionList[i][-1][:-1]
                    meanObjective = meanObjectiveList[i][-1]

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution.astype(np.int16))
            if timeLeft < 0:
                if verbose:
                    print("Terminated because of time limit")
                break

            # Print progress
            if verbose == True and iteration % 5 == 0:
                print(
                    "Iteration: %d, Mean: %f, Best: %f, Time left: %f, Diff Best: %d"
                    % (iteration, meanObjective, bestObjective, timeLeft, min(differentBestSolutions))
                )

            if min(differentBestSolutions) >= self.MAX_DIFFERENT_BEST_SOLUTIONS:
                yourConvergenceTestsHere = False
                print(
                    "Terminated because of %d equal best solutions"
                    % min(differentBestSolutions)
                )
                break

            iteration += 1

        # Return the best solution
        bestSolution = bestSolution.astype(np.int16)
        return bestSolution, bestObjective, bestObjectiveList, bestSolutionList, meanObjectiveList, alphaMeanList, iteration+1

"""
def optimize_island(verbose=False, testMode=False):

    start_time = time.time()

    # Initialize the n_population populations  
    populations = []
    for i in range(n_population):
        populations.append(initialize(alphaList[i]))
        populations[i] = two_opt(populations[i])


    # Store the progress of each population
    meanObjectiveList = [[] for i in range(n_population)]
    bestObjectiveList = [[] for i in range(n_population)]
    bestSolutionList = [[] for i in range(n_population)]
    alphaMeanList = [[] for i in range(n_population)]

    # Termination criteria
    differentBestSolutions = [0] * n_population

    # Tuning parameter for the local search
    #tuning = [2] * n_population

    # Set up the loop
    iteration = 0
    yourConvergenceTestsHere = True

    # Set up the executor
    executor = ProcessPoolExecutor(max_workers=n_population)
    
    while yourConvergenceTestsHere:

        populations = list(executor.map(process_island, populations))

        # For each population
        for i in range(n_population):            

            # Compute and save progress
            fvals = objf_pop(populations[i])                                    # Compute the objective function value for each individual
            meanObjectiveList[i].append(np.mean(fvals))                         # Compute the mean objective value of the population           
            bestObjectiveList[i].append(np.min(fvals))                          # Store the new best objective
            bestSolutionList[i].append(populations[i][np.argmin(fvals), :])     # Store the new best solution
            alphaMeanList[i].append(np.mean(populations[i][:, -1]))             # Mean of alpha values

            # Adaptation of parameters
            if iteration > 1 and bestObjectiveList[i][-1] == bestObjectiveList[i][-2] and bestObjectiveList[i][-1] != np.inf:
                differentBestSolutions[i] += 1
            else:
                differentBestSolutions[i] = 0

            #if differentBestSolutions[i] > 0 and differentBestSolutions[i] % 8 == 0:
            #    tuning[i] += 1

            #if tuning[i] > 5:
            #    tuning[i] = 3


        # Interact the populations 
        if iteration % iterationIneraction == 0:
            # Join the n_population populations
            completePopulation = np.vstack(populations)
            # Shuffle the complete population
            np.random.shuffle(completePopulation)
            # Create n_population from the complete population
            for i in range(n_population):
                populations[i] = completePopulation[i*lambda_:(i+1)*lambda_, :]


        # Find the best solution among the n_population populations
        bestSolution = bestSolutionList[0][-1][:-1]
        bestObjective = bestObjectiveList[0][-1]
        meanObjective = meanObjectiveList[0][-1]
                
        for i in range(1,n_population):
            if bestObjectiveList[i][-1] < bestObjective:
                bestObjective = bestObjectiveList[i][-1]
                bestSolution = bestSolutionList[i][-1][:-1]
                meanObjective = meanObjectiveList[i][-1]

        # Call the reporter with:
        #  - the mean objective function value of the population
        #  - the best objective function value of the population
        #  - a 1D numpy array in the cycle notation containing the best solution
        #    with city numbering starting from 0
        timeLeft = reporter.report(meanObjective, bestObjective, bestSolution.astype(np.int16))

        # Print progress
        if verbose == True and iteration % 5 == 0:
            print(
                "Iteration: %d, Mean: %f, Best: %f, Time left: %f, Diff Best: %d"
                % (iteration, meanObjective, bestObjective, timeLeft, min(differentBestSolutions))
            )

        if min(differentBestSolutions) >= MAX_DIFFERENT_BEST_SOLUTIONS:
            yourConvergenceTestsHere = False
            if not testMode:
                print(
                    "Terminated because of %d equal best solutions"
                    % min(differentBestSolutions)
                )
            break

        if timeLeft < 0 and not testMode:
            yourConvergenceTestsHere = False
            if not testMode:
                print("Terminated because of time limit")
            break
        elif (time.time() - start_time) > 300 and testMode:
            yourConvergenceTestsHere = False
            if not testMode:
                print("Terminated because of time limit")
            break

        iteration += 1

    # Return the best solution
    bestSolution = bestSolution.astype(np.int16)
    return bestSolution, bestObjective, bestObjectiveList, bestSolutionList, meanObjectiveList, alphaMeanList, iteration+1
"""

def process_island(population, distanceMatrix, n_cities, lambda_, mu, k_for_selection, alpha):

    # Selection
    selected = selection(population, k_for_selection, distanceMatrix, n_cities, mu)

    # Crossover
    offspring = crossover(selected, n_cities, lambda_, mu)

    # Join the population and the offspring
    joinedPopulation = np.vstack((population, offspring))

    # Mutation on the joined population without the 3 best solutions
    sorted_population = joinedPopulation[np.argsort(objf_pop(joinedPopulation, distanceMatrix, n_cities))]
    joinedPopulation = np.vstack((sorted_population[:lambda_//6, :], mutation(sorted_population[lambda_//6:, :], n_cities)))

    # Local search
    joinedPopulation = two_opt(joinedPopulation, distanceMatrix)
    
    # Elimination
    population = elimination_pro(joinedPopulation, distanceMatrix, n_cities, lambda_, alpha)

    return population

# UTILITY FUNCTIONS

@njit
def tourIsValid(tour: np.ndarray, n_cities: int):
    # Check if the tour is valid
    tour = tour.astype(np.int16)
    if len(np.unique(tour[:n_cities])) != n_cities:
        return False
    else:
        return True

@njit
def objf_pop(population : np.ndarray, distanceMatrix: np.ndarray, n_cities: int):
    sum_distance = np.zeros(population.shape[0])

    for i in range(population.shape[0]):
        sum_distance[i] = objf(population[i,:n_cities], distanceMatrix)
    
    return sum_distance

@njit
def objf(tour : np.ndarray, distanceMatrix: np.ndarray):

    # Convert float64 indices to integers
    tour = tour.astype(np.int16)

    # Apply the objective function to each row of the cities array
    sum_distance = 0

    for ii in range(tour.shape[0] - 1):
        # Sum the distances between the cities
        sum_distance += distanceMatrix[tour[ii], tour[ii + 1]]

        if sum_distance == np.inf:
            return np.inf

    # Add the distance between the last and first city
    sum_distance += distanceMatrix[tour[- 1], tour[0]]

    return sum_distance

# INIZIALIZATION

@njit
def generate_individual_greedy(distanceMatrix: np.ndarray, n_cities: int):
    # Create an individual choosing always the nearest city
    individual = np.zeros(n_cities).astype(np.int16)
    individual[0] = 0

    not_visited = np.arange(1, n_cities)

    for ii in range(1, n_cities):
        # Select the nearest city
        nearest_city = np.argmin(distanceMatrix[individual[ii - 1], not_visited])
        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)

    return individual
    
@njit
def generate_individual_greedy_reverse(distanceMatrix: np.ndarray, n_cities: int):
    # Create an individual choosing always the nearest city
    individual = np.zeros(n_cities).astype(np.int16)
    individual[0] = 0

    # Last city is random
    individual[n_cities-1] = np.random.randint(1, n_cities)

    not_visited = np.arange(1, n_cities)
    not_visited = np.delete(not_visited, np.where(not_visited == individual[n_cities-1])[0][0])

    for ii in range(n_cities-2, 0 , -1):
        # Select the nearest city backwards
        nearest_city = np.argmin(distanceMatrix[not_visited, individual[ii]])
        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)

    return individual

@njit
def generate_individual_nearest_neighbor(distanceMatrix: np.ndarray, n_cities: int):
    # Create an individual choosing always the nearest city , second city is random

    # Create an individual starting from zero
    individual = np.zeros(n_cities).astype(np.int16)
    individual[0] = 0

    # Second city is random from the city accessible from the first city 
    accessible_cities = np.argsort(distanceMatrix[0, :])
    individual[1] = accessible_cities[np.random.randint(1, accessible_cities.shape[0])]

    not_visited = np.arange(1, n_cities)
    not_visited = np.delete(not_visited, np.where(not_visited == individual[1])[0][0])

    for ii in range(2, n_cities):
        # Select the nearest city
        nearest_city = np.argmin(distanceMatrix[individual[ii - 1], not_visited])
        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)

    return individual

@njit
def generate_individual_nearest_neighbor_more_index(k: int, distanceMatrix: np.ndarray, n_cities: int):
    # Create an individual choosing always the nearest city , but choosing randomly each k cities
    k = (n_cities + 1)//k

    # Create an individual starting from zero
    individual = np.zeros(n_cities).astype(np.int16)
    individual[0] = 0
    # Second city is random from the city accessible from the first city
    accessible_cities = np.argsort(distanceMatrix[0, :])
    individual[1] = accessible_cities[np.random.randint(1, accessible_cities.shape[0])]

    not_visited = np.arange(1, n_cities)
    not_visited = np.delete(not_visited, np.where(not_visited == individual[1])[0][0])

    for ii in range(2, n_cities):

        if ii % k == 0:
            # Select randomly a city from the not visited list
            nearest_city= np.random.randint(0, not_visited.shape[0])
        else:
            # Select the nearest city
            nearest_city = np.argmin(distanceMatrix[individual[ii - 1], not_visited])

        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)


    return individual

@njit
def generate_individual_random(n_cities: int):
    r = np.zeros(n_cities).astype(np.int16)
    r[0] = 0
    r[1:] = np.random.permutation(np.arange(1, n_cities)).astype(np.int16)

    return r


def initialize(my_alpha: float, distanceMatrix: np.ndarray, n_cities: int, lambda_: int) -> np.ndarray:
    # Create a matrix of random individuals
    population = np.zeros((lambda_, n_cities + 1)) # +1 for the alpha value

    not_inf = 0

    for i in range(lambda_):

        if i == 0:
            new_individual = generate_individual_greedy(distanceMatrix, n_cities)
        elif i == 1:
            new_individual = generate_individual_greedy_reverse(distanceMatrix, n_cities)
        elif i == 2:
            new_individual = generate_individual_nearest_neighbor(distanceMatrix, n_cities)
        elif i == 3:
            new_individual = generate_individual_nearest_neighbor_more_index(5,distanceMatrix, n_cities)
        else:
            new_individual = generate_individual_random(n_cities)

        # Evaluate the individual with the objective function
        obj = objf(new_individual, distanceMatrix)
        # Check if the individual is valid for at most n times
        max_tries = n_cities
        while obj == np.inf and max_tries > 0:
            # Create a random individual
            new_individual = generate_individual_random(n_cities)
            # Evaluate the individual with the objective function
            obj = objf(new_individual, distanceMatrix)
            max_tries -= 1

        # Create alpha with gaussian distribution
        alpha = np.array([np.random.normal(my_alpha, 0.03)])
        # Concatenate the alpha value to the individual
        new_individual = np.concatenate((new_individual, alpha))

        if obj == objf(new_individual[:-1], distanceMatrix) != np.inf:
            not_inf += 1

        population[i, :] = new_individual

    return population

# SELECTION

@njit
def selection(population: np.ndarray, k: int, distanceMatrix: np.ndarray, n_cities: int, mu: int):

    # Select  parents with elitism
    elite_dim = mu//8
    elite = np.zeros((elite_dim, n_cities + 1))  # +1 for the alpha value
    elite[:,:-1] = elite[:,:-1].astype(np.int16)

    # Select the best individuals
    elite[:elite_dim, :] = population[np.argsort(objf_pop(population, distanceMatrix, n_cities))[:elite_dim], :]

    # Create a matrix of selected parents
    selected = np.zeros((mu - elite_dim, n_cities + 1))  # +1 for the alpha value
    selected[:,:-1] = selected[:,:-1].astype(np.int16)

    # Selecting the parents
    for ii in range(mu - elite_dim):
        # Select k random individuals from the population
        ri = np.random.choice(np.arange(1, population.shape[0] - 1), k, replace=False)

        # Select the best individual from the k random individuals
        best = np.argmin(objf_pop(population[ri, :], distanceMatrix, n_cities))

        # Add the selected individual to the matrix of selected parents
        selected[ii, :] = population[ri[best], :]

    # Join the elite and the selected parents
    selected = np.vstack((elite, selected))

    if selected.shape[0] != mu:
        raise ValueError("The number of selected parents must be equal to mu")

    return selected
    
"""
def selection_roulette(population: np.ndarray):

    # Calculate the fitness values
    fitness_values = 1.0 / objf_pop(population)

    # Normalize the fitness values to create probabilities
    probabilities = fitness_values / np.sum(fitness_values)

    # Perform roulette wheel selection
    selected_indices = np.random.choice(np.arange(population.shape[0]), size=mu, p=probabilities, replace=True)

    # Create the matrix of selected parents
    selected = population[selected_indices, :]

    return selected
"""

# LOCAL SEARCH

@njit
def one_opt(population: np.ndarray, k, distanceMatrix: np.ndarray, n_cities: int):

    if k < 1 or k > population.shape[1] - 1:
        raise ValueError("k must be between 2 and n-1")

    for i in range(population.shape[0]):

        # For each individual in the population
        best_tour = population[i, :]
        best_obj = objf(best_tour[:-1], distanceMatrix)

        # Select k random indices
        k_list = sorted(np.random.choice(np.arange(1, population.shape[1] - 2), k, replace=False))

        for ri in k_list:
            # Swap the ri-th and ri+1-th cities
            tour = population[i, :].copy()
            tour[ri], tour[ri+1] = tour[ri+1], tour[ri]

            # Evaluate the new tour
            new_obj = objf(tour[:-1], distanceMatrix)

            # Check if the new tour is better
            if new_obj < best_obj:
                best_tour = tour
                best_obj = new_obj

        population[i, :] = best_tour

    return population

@njit
def build_cumulatives(order: np.ndarray, 
                      length: int, distanceMatrix: np.ndarray):
    order = order.astype(np.int16)

    cum_from_0_to_first = np.zeros((length))
    cum_from_second_to_end = np.zeros((length))
    cum_from_second_to_end[length - 1] = distanceMatrix[order[-1], order[0]]
    for i in range(1, length - 1):
        cum_from_0_to_first[i] = cum_from_0_to_first[i - 1] \
            + distanceMatrix[order[i-1], order[i]]        
        cum_from_second_to_end[length - 1 - i] = cum_from_second_to_end[length - i] \
            + distanceMatrix[order[length -1 - i], order[length - i]]
    return cum_from_0_to_first, cum_from_second_to_end

@njit
def two_opt(population: np.ndarray, distanceMatrix: np.ndarray):
    # Apply the local search operator to each row of the population array
    for ii in range(population.shape[0]):
        new = operator_2_opt(population[ii, :-1], distanceMatrix)
        if new is not None:
                population[ii, :-1] = new
    
    return population

@njit
def operator_2_opt(order: np.ndarray, distanceMatrix: np.ndarray):
    """Local search operator, which makes use of 2-opt. Swap two edges within a cycle."""

    order = order.astype(np.int16)

    best_fitness = objf(order, distanceMatrix)
    length = len(order)
    best_combination = (0, 0)

    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(order, length, distanceMatrix)
    if cum_from_second_to_end[-1] > np.inf:
        return None

    for first in range(1, length - 2):
        fit_first_part = cum_from_0_to_first[first-1]
        if fit_first_part > np.inf or fit_first_part > best_fitness:
            break
        fit_middle_part = 0.0
        for second in range(first + 2, length):
            fit_middle_part += distanceMatrix[order[second-1],order[second-2]]
            if fit_middle_part > np.inf:
                break
            
            fit_last_part = cum_from_second_to_end[second]
            if fit_last_part > np.inf:
                continue

            bridge_first = distanceMatrix[order[first-1],order[second-1]]
            bridge_second = distanceMatrix[order[first],order[second]]
            temp = fit_first_part + fit_middle_part
            if temp > best_fitness:
                continue
            new_fitness = temp + fit_last_part + bridge_first + bridge_second
            
            if new_fitness < best_fitness:
                best_combination = (first, second)
                best_fitness = new_fitness
    best_first, best_second = best_combination
    if best_first == 0: # Initial individual was best
        return None
    new_order = np.copy(order)
    new_order[best_first:best_second] = new_order[best_first:best_second][::-1]

    return new_order

"""
def local_search_best_subset(population: np.ndarray, k: int):
    # Apply the best subset solution to each row of the population array
    for ii in range(population.shape[0]):
        population[ii, :-1] = best_subset_solution(population[ii, :-1], k)
    
    return population

def best_subset_solution(tour: np.ndarray, k: int):

    tour = tour.astype(np.int16)
    
    # Generate one random index
    ri = np.random.randint(1, n_cities - k)

    # Find the best subset solution from ri to ri+k using brute force

    # Initialize the best tour and objective function value
    best_tour = tour.copy()
    best_obj = objf_perm(tour[ri-1:ri+k+1])

    # Generate all the possible permutations of the cities from ri to ri+k
    if k <= 6:
        permutations = np.array(list(itertools.permutations(tour[ri:ri+k]))) # Faster with small k
    else:
        permutations = generate_permutations(tour[ri:ri+k]) # Numba version
    
    # Add tour[ri-1] and tour[ri+k] to the permutations
    permutations = np.concatenate((np.ones((permutations.shape[0], 1)).astype(np.int16) * tour[ri-1], permutations), axis=1)
    permutations = np.concatenate((permutations, np.ones((permutations.shape[0], 1)).astype(np.int16) * tour[ri+k]), axis=1)

    # Evaluate the objective function for each permutation
    objs = objf_permutation(permutations)
    best = np.argmin(objs)

    # Check if the best permutation is better than the original tour
    if objs[best] < best_obj:

        # Update the best tour and objective function value
        best_tour[ri:ri+k] = permutations[best, 1:-1]

        if not tourIsValid(best_tour):
            print("best_tour: ", best_tour)
            raise ValueError("Invalid tour during best subset solution")
        
        if objf(tour) < objf(best_tour):
            print("objf(tour): ", objf(tour))
            print("objf(best_tour): ", objf(best_tour))
            raise ValueError("Best subset solution is worse than the original tour")

    return best_tour

def _factorial(n):
    if n == 1:
        return n
    else:
        return n * _factorial(n-1)

def generate_permutations(arr: np.ndarray):
    d = arr.shape[0]
    d_fact = _factorial(d)
    c = np.zeros(d, dtype=np.int32)
    res = np.zeros(shape=(d_fact, d))
    counter = 0
    i = 0
    res[counter] = arr
    counter += 1
    while i < d:
        if c[i] < i:
            if i % 2 == 0:
                arr[0], arr[i] = arr[i], arr[0]
            else:
                arr[c[i]], arr[i] = arr[i], arr[c[i]]
            # Swap has occurred ending the for-loop. Simulate the increment of the for-loop counter
            c[i] += 1
            res[counter] = arr
            counter += 1
            # Simulate recursive call reaching the base case by bringing setting i to the base case
            i = 0
        else:
            # Calling the func(i+1, A) has ended as the for-loop terminated. 
            # Reset the state and increment i.
            c[i] = 0
            i += 1
    # Return array of d! rows and d columns (all permutations)
    return res   

def objf_perm(tour: np.ndarray):
        
        # Convert float64 indices to integers
        tour = tour.astype(np.int16)
    
        # Apply the objective function to each row of the cities array
        sum_distance = 0
    
        for ii in range(tour.shape[0] - 1):
            # Sum the distances between the cities
            sum_distance += distanceMatrix[tour[ii], tour[ii + 1]]

            if sum_distance == np.inf:
                return np.inf
    
        return sum_distance

def objf_permutation(permutations: np.ndarray):
        # Apply the objective function to each row of the permutations array
        obj = np.zeros(permutations.shape[0])
    
        for ii in range(permutations.shape[0]):
            obj[ii] = objf_perm(permutations[ii, :])
    
        return obj
"""
# CROSSOVER

@njit
def uox(parent1: np.ndarray, parent2: np.ndarray, n_cities: int):
    # Create two children
    child1 = np.ones(n_cities).astype(np.int16)
    child2 = np.ones(n_cities).astype(np.int16)

    child1 = child1 * -1
    child2 = child2 * -1

    child1[0] = 0
    child2[0] = 0


    # Choose two random crossover points
    point1, point2 = sorted(np.random.choice(np.arange(1, n_cities - 1), 2, replace=False))

    # Copy the genes from the first parent between the crossover points
    child1[point1:point2] = parent1[point1:point2]
    child2[point1:point2] = parent2[point1:point2]

    # Fill in the remaining positions with genes from the second parent in order
    remaining_positions1 = [i for i in parent2[:-1] if i not in child1]
    remaining_positions2 = [i for i in parent1[:-1] if i not in child2]

    idx1 = 0
    idx2 = 0
    for i in range(n_cities):
        if child1[i] == -1:
            child1[i] = remaining_positions1[idx1]
            idx1 += 1
        if child2[i] == -1:
            child2[i] = remaining_positions2[idx2]
            idx2 += 1

        

    # Add the alpha value to the children (average of the parents' alpha values)
    new_alpha1 = np.array([parent1[-1]])
    new_alpha2 = np.array([parent2[-1]])

    # Concatenate the children to the alpha values
    child1 = np.concatenate((child1, new_alpha1))
    child2 = np.concatenate((child2, new_alpha2))

    return child1, child2

@njit
def crossover(selected: np.ndarray, n_cities: int, lambda_: int, mu: int):
    # Create a matrix of offspring
    offspring = np.zeros((mu, n_cities + 1)) # +1 for the alpha value
    offspring[:,:-1] = offspring[:,:-1].astype(np.int16)

    for ii in range(lambda_):
        # Select two random parents
        ri = sorted(np.random.choice(np.arange(1, lambda_), 2, replace=False))

        # Perform crossover
        #if np.random.rand() < 0.5:
        #    offspring[ii, :], offspring[ii + lambda_, :] = pmx(selected[ri[0], :], selected[ri[1], :])
        #else:
        offspring[ii, :], offspring[ii + lambda_, :] = uox(selected[ri[0], :], selected[ri[1], :], n_cities)

    return offspring

"""
@njit
def scx(parent1: np.ndarray, parent2: np.ndarray):

    # Create a child
    child1 = np.ones(n_cities).astype(np.int16)
    child2 = np.ones(n_cities).astype(np.int16)

    child1 = child1 * -1
    child2 = child2 * -1

    child1[0] = 0
    child2[0] = 0

    # Helper function to find the next city in the sequence
    def find_next_city(parent: np.ndarray, child: np.ndarray, idx):
        current_city = int(child[idx])
        remaining_cities = np.array([i for i in parent[:-1] if i not in child]).astype(np.int16)
        distances = np.array([distanceMatrix[current_city, city] for city in remaining_cities])
        next_city = remaining_cities[np.argmin(distances)]
        return next_city

    # Construct the rest of the child sequences
    for idx in range(1, n_cities):
        child1[idx] = find_next_city(parent1, child1, idx)
        child2[idx] = find_next_city(parent2, child2, idx)

    # Add the alpha value to the child (average of the parents' alpha values)
    new_alpha1 = np.array([parent1[-1]])
    new_alpha2 = np.array([parent2[-1]])

    # Concatenate the child to the alpha value
    child1 = np.concatenate((child1, new_alpha1))
    child2 = np.concatenate((child2, new_alpha2))

    if not tourIsValid(child1[:-1]):
        raise ValueError("Invalid tour during scx crossover")
    if not tourIsValid(child2[:-1]):
        raise ValueError("Invalid tour during scx crossover")

    return child1, child2
"""

# MUTATION

@njit
def swap_mutation(tour, n_cities: int):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 2, replace=False))

    tour[ri[0]], tour[ri[1]] = tour[ri[1]], tour[ri[0]]

    return tour
    
@njit
def insert_mutation(tour, n_cities: int):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 2, replace=False))

    removed = tour[ri[1]]
    tour = np.delete(tour, ri[1])
    tour = np.concatenate((tour[:ri[0]], np.array([removed]), tour[ri[0]:]))

    return tour

@njit
def mutation(offspring: np.ndarray, n_cities: int):

    # Apply the mutation to each row of the offspring array
    for ii, _ in enumerate(offspring):

        # Apply the mutation with probability alpha of the individual
        if np.random.rand() < offspring[ii, n_cities]:

            # Add a noise to the alpha value
            offspring[ii, -1] = np.random.normal(offspring[ii, -1], 0.02)

            if np.random.rand() < 0.2:
                offspring[ii, :] = swap_mutation(offspring[ii, :], n_cities)
            else:
                offspring[ii, :] = inversion_mutation(offspring[ii, :], n_cities)
                

    return offspring

@njit
def scramble_mutation(tour, n_cities: int):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 2, replace=False))

    np.random.shuffle(tour[ri[0]:ri[1]])

    return tour

@njit
def inversion_mutation(tour, n_cities: int):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 2, replace=False))

    tour[ri[0]:ri[1]] = tour[ri[0]:ri[1]][::-1]

    return tour

@njit
def thrors_mutation(tour, n_cities: int):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 3, replace=False))

    tour[ri[0]], tour[ri[1]], tour[ri[2]] = tour[ri[1]], tour[ri[2]], tour[ri[0]]

    return tour

# ELIMINATION  

def elimination_pro(joinedPopulation: np.ndarray, distanceMatrix: np.ndarray, n_cities: int, lambda_: int, alpha: float):

    # Apply the objective function to each row of the joinedPopulation array
    fvals = objf_pop(joinedPopulation, distanceMatrix, n_cities)

    # Sort the individuals based on their objective function value
    perm = np.argsort(fvals)

    # Select the best individuals
    n_best = int(lambda_/8)
    best_survivors = joinedPopulation[perm[0 : 3*n_best], :]

    # Select randomly individuals
    random_survivors = joinedPopulation[np.random.choice(perm[n_best:], 4*n_best, replace=False), :]

    #Remove duplicates
    best_survivors = np.unique(best_survivors, axis=0)
    random_survivors = np.unique(random_survivors, axis=0)

    # Generate  individuals randomly
    generate_survivors = np.zeros((lambda_ - best_survivors.shape[0] - random_survivors.shape[0], n_cities + 1))
    for i in range(generate_survivors.shape[0]):
        generate_survivors[i, :-1] = generate_individual_random(n_cities)
        # Take a value from alphaList
        generate_survivors[i, -1] = np.random.normal(alpha, 0.03)

    # Concatenate the best and random survivors
    survivors = np.vstack((best_survivors, random_survivors, generate_survivors))

    if survivors.shape[0] != lambda_:
        raise ValueError("The number of survivors must be equal to lambda_")
    
    return survivors

"""

# VISUALIZATION

def plot_results(meanObjectiveList: list, bestObjectiveList: list, iteration: int):

    #Remove the last MAX_DIFFERENT_BEST_SOLUTIONS values
    iteration = iteration - MAX_DIFFERENT_BEST_SOLUTIONS + 50
    for i in range(n_population):
        bestObjectiveList[i] = bestObjectiveList[i][:iteration]
        meanObjectiveList[i] = meanObjectiveList[i][:iteration]
            
    # Calculate the mean for every iteration
    meanObjectiveList = np.array(meanObjectiveList)
    meanObjectiveList = np.mean(meanObjectiveList, axis=0)
    
    # Plot the mean and the best obj comparing the n_population populations
    t = np.arange(0, iteration, 1)
    
    #Plot just one graph
    fig, axs = plt.subplots(1, 1)

    for i in range(n_population):
        axs.plot(t, bestObjectiveList[i], label='Best ' + str(i))
        axs.set_title('Best objective function value')
        axs.set(xlabel='Iteration', ylabel='Objective function value')

        #Draw a vertical line every iterationIneraction iterations
        for j in range(1,iteration//iterationIneraction + 1):
            axs.axvline(x=iterationIneraction*j, color='grey', linestyle='--')

    # Plot the mean
    #axs.plot(t, meanObjectiveList, label='Mean')
    axs.legend()

    plt.show()

def plot_results_with_alpha(meanObjectiveList: list, bestObjectiveList: list, alphaMeanList: list, iteration: int):
        
        # Calculate the mean for every iteration
        meanObjectiveList = np.array(meanObjectiveList)
        meanObjectiveList = np.mean(meanObjectiveList, axis=0)
        
        # Plot the mean and the best obj comparing the n_population populations
        t = np.arange(0, iteration, 1)

        fig, axs = plt.subplots(1, 2)

        # Set the dimensions of the figure
        fig.set_figheight(10)
        fig.set_figwidth(15)


        for i in range(n_population):
            axs[0].plot(t, bestObjectiveList[i], label='Best ' + str(i))
            axs[0].set_title('Best objective function value')
            axs[0].set(xlabel='Iteration', ylabel='Objective function value')

            axs[1].plot(t, alphaMeanList[i], label='Alpha ' + str(i))
            axs[1].set_title('Mean of alpha values')
            axs[1].set(xlabel='Iteration', ylabel='Alpha value')

            #Draw a vertical line every iterationIneraction iterations
            for j in range(1,iteration//iterationIneraction + 1):
                axs[0].axvline(x=iterationIneraction*j, color='grey', linestyle='--')
                axs[1].axvline(x=iterationIneraction*j, color='grey', linestyle='--')

        # Plot the mean
        axs[0].plot(t, meanObjectiveList, label='Mean')
        axs[0].legend()
        axs[1].legend()

        plt.show()

# TEST
        
def test():
    n_runs = 500
    for i in range(n_runs):
        print("Run: ", i)
        best_tour, best_obj, bestObjectiveList, bestSolutionList, meanObjectiveList, alphaMeanList, iteration =  optimize_island(verbose=False, testMode=True)
        if not tourIsValid(best_tour):
            print("Invalid tour!")
        results.append(best_obj)

    plt.hist(results, bins=20, color='c', edgecolor='k', alpha=0.65)
    plt.title("Histogram of the best objective function values")
    plt.xlabel("Objective function value")
    plt.ylabel("Frequency")
    plt.show()

    # Compute the mean of the best objective function values
    print("Mean of the best objective function values: ", np.mean(results))
    print("Standard deviation of the best objective function values: ", np.std(results))

def individual_test():
    start_time = time.time()
    best_tour, best_obj, bestObjectiveList, bestSolutionList, meanObjectiveList, alphaMeanList, iteration =  optimize_island(verbose=True)
    end_time = time.time()

    print("Best tour: ", best_tour)
    print("Best objective function value: ", best_obj, " Checked: ", objf(best_tour), " Valid: ", tourIsValid(best_tour))
    print("Time: ", end_time - start_time)

    #plot_results_with_alpha(meanObjectiveList, bestObjectiveList, alphaMeanList, iteration)
    plot_results(meanObjectiveList, bestObjectiveList, iteration)

"""
########################################################################################
        


# Main 
if __name__ == "__main__":
    file_paths = ["tour50.csv", "tour100.csv", "tour200.csv", "tour500.csv", "tour750.csv", "tour1000.csv"]
    filename = file_paths[3]
    print("File: ", filename)
    a = r0978639()
    a. optimize (filename, verbose=True, parallelization=True)


# Benchmark results:
# tour50: simple greedy heuristic 27723       (BEST 25668)
# tour100: simple greedy heuristic 90851      (BEST 77936)   
# tour200: simple greedy heuristic 39745      (BEST 36985)   
# tour500: simple greedy heuristic 157034     (BEST 129135)  
# tour750: simple greedy heuristic 197541     (BEST 171437)
# tour1000: simple greedy heuristic 195848    (BEST 194852)